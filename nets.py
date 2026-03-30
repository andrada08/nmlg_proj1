import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeLayerSkipNet(nn.Module):
    """
    Original architecture:
      - layer1 -> layer2
      - layer2 + layer1 skip into layer3 (output)
    
    Supports both feedforward (linear) and convolutional layers via layer_types config.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, int, int],
        output_size: int,
        activation: str = "relu",
        layer_types: dict[str, str] | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        h1, h2, h3 = hidden_sizes
        
        # Default to all linear if layer_types not provided
        if layer_types is None:
            layer_types = {}
        
        # Get layer types with defaults
        # layer3_from_1 and layer3_from_2 are always linear (projection layers for skip connections)
        # layer3 is always linear (final output layer)
        layer1_type = layer_types.get("layer1", "linear")
        layer2_type = layer_types.get("layer2", "linear")
        
        # Store layer types for forward pass
        self.layer1_type = layer1_type
        self.layer2_type = layer2_type
        
        # Default conv parameters
        conv_kernel_size = 3
        conv_stride = 1
        conv_padding = 1
        
        # Create layer1
        if layer1_type == "conv":
            # For MNIST: input is 1 channel, 28x28
            # hidden_sizes[0] is number of output channels
            self.layer1 = nn.Conv2d(1, h1, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
            # With padding=1, spatial size remains 28x28
            self.layer1_spatial_size = 28
            self.layer1_output_size = h1 * 28 * 28
        else:  # linear
            self.layer1 = nn.Linear(input_size, h1)
            self.layer1_output_size = h1
        
        # Create layer2
        if layer2_type == "conv":
            if layer1_type == "conv":
                # Conv -> Conv: h1 channels -> h2 channels
                self.layer2 = nn.Conv2d(h1, h2, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
                self.layer2_spatial_size = 28  # Still 28x28 with padding=1
            else:  # linear -> conv
                # Need to reshape: linear output of size h1 to spatial dimensions
                # Find spatial dimensions that multiply to h1 (prefer square-ish)
                spatial_dim = int(h1 ** 0.5)
                if spatial_dim * spatial_dim == h1:
                    # Perfect square
                    self.layer2_spatial_size = spatial_dim
                else:
                    # Find best factor pair (closest to square)
                    best_h, best_w = 1, h1
                    min_diff = abs(1 - h1)
                    for h in range(1, int(h1 ** 0.5) + 1):
                        if h1 % h == 0:
                            w = h1 // h
                            diff = abs(h - w)
                            if diff < min_diff:
                                min_diff = diff
                                best_h, best_w = h, w
                    # Use the larger dimension as spatial_size (we'll reshape to best_h x best_w)
                    self.layer2_spatial_size = max(best_h, best_w)
                    self.layer2_spatial_h = best_h
                    self.layer2_spatial_w = best_w
                self.layer2 = nn.Conv2d(1, h2, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)
                # Calculate output size based on actual spatial dimensions
                if hasattr(self, 'layer2_spatial_h') and hasattr(self, 'layer2_spatial_w'):
                    # Non-square: use actual dimensions (with padding=1, spatial size stays the same)
                    self.layer2_output_size = h2 * self.layer2_spatial_h * self.layer2_spatial_w
                else:
                    # Square case
                    self.layer2_output_size = h2 * self.layer2_spatial_size * self.layer2_spatial_size
        else:  # linear
            if layer1_type == "conv":
                # Conv -> Linear: need to flatten
                self.layer2 = nn.Linear(self.layer1_output_size, h2)
            else:  # linear -> linear
                self.layer2 = nn.Linear(h1, h2)
            self.layer2_output_size = h2
        
        # Create layer3_from_1 (skip connection from layer1) - always linear
        self.layer3_from_1 = nn.Linear(self.layer1_output_size, h3)
        
        # Create layer3_from_2 (skip connection from layer2) - always linear
        self.layer3_from_2 = nn.Linear(self.layer2_output_size, h3)
        
        # layer3 is always linear (final output)
        # Both skip connections output h3, so combined input is h3
        self.layer3_input_size = h3
        
        self.layer3 = nn.Linear(self.layer3_input_size, output_size)
        self.activation = getattr(F, activation)

    def forward(self, x):
        # Handle input shape
        if self.layer1_type == "conv":
            # Input should be (batch, 1, 28, 28) for MNIST
            if x.dim() == 2:
                # Flattened input, reshape to image
                batch_size = x.shape[0]
                x = x.view(batch_size, 1, 28, 28)
        else:  # linear
            # Flatten input
            x = x.view(-1, self.input_size)
        
        # Layer1 - preserve x1 for skip connection
        x1 = self.layer1(x)
        x1 = self.activation(x1)
        x1_for_skip = x1  # Keep original for skip connection
        
        # Layer2 - preserve x2 for skip connection
        if self.layer2_type == "conv":
            if self.layer1_type == "linear":
                # Reshape linear output to spatial for conv
                batch_size = x1.shape[0]
                h1_size = x1.shape[1]
                # Use the spatial dimensions calculated in __init__
                if hasattr(self, 'layer2_spatial_h') and hasattr(self, 'layer2_spatial_w'):
                    # Non-square case: use the calculated dimensions
                    x1_reshaped = x1.view(batch_size, 1, self.layer2_spatial_h, self.layer2_spatial_w)
                else:
                    # Square case: use spatial_size x spatial_size
                    spatial_size = self.layer2_spatial_size
                    if h1_size == spatial_size * spatial_size:
                        x1_reshaped = x1.view(batch_size, 1, spatial_size, spatial_size)
                    else:
                        # Fallback: recalculate (shouldn't happen if __init__ is correct)
                        spatial_dim = int(h1_size ** 0.5)
                        if spatial_dim * spatial_dim == h1_size:
                            x1_reshaped = x1.view(batch_size, 1, spatial_dim, spatial_dim)
                        else:
                            # Find factors
                            best_h, best_w = 1, h1_size
                            for h in range(1, int(h1_size ** 0.5) + 1):
                                if h1_size % h == 0:
                                    w = h1_size // h
                                    if abs(h - w) < abs(best_h - best_w):
                                        best_h, best_w = h, w
                            x1_reshaped = x1.view(batch_size, 1, best_h, best_w)
            else:
                x1_reshaped = x1
            x2 = self.activation(self.layer2(x1_reshaped))
        else:  # linear
            if self.layer1_type == "conv":
                # Flatten conv output
                batch_size = x1.shape[0]
                x1_flat = x1.view(batch_size, -1)
            else:
                x1_flat = x1
            x2 = self.activation(self.layer2(x1_flat))
        x2_for_skip = x2  # Keep original for skip connection
        
        # Skip connections to layer3 (always linear projection layers)
        # Process layer3_from_1 (using preserved x1_for_skip)
        if self.layer1_type == "conv":
            # Flatten conv output
            batch_size = x1_for_skip.shape[0]
            x1_flat = x1_for_skip.view(batch_size, -1)
        else:
            x1_flat = x1_for_skip
        x3_from_1 = self.layer3_from_1(x1_flat)
        
        # Process layer3_from_2 (using preserved x2_for_skip)
        if self.layer2_type == "conv":
            # Flatten conv output
            batch_size = x2_for_skip.shape[0]
            x2_flat = x2_for_skip.view(batch_size, -1)
        else:
            x2_flat = x2_for_skip
        x3_from_2 = self.layer3_from_2(x2_flat)
        
        # Combine skip connections
        x3_combined = x3_from_1 + x3_from_2
        
        # Final layer (always linear)
        output = self.layer3(x3_combined)
        return output

    def get_layer_names(self) -> list[str]:
        return ["layer1", "layer2", "layer3_from_1", "layer3_from_2", "layer3"]


class FourLayerIntegratingNet(nn.Module):
    """
    Four-layer integrating architecture:
      - layer1 -> layer2
      - layer1, layer2 -> layer3
      - layer2, layer3 -> layer4
      - layer4 -> output
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, int, int, int],
        output_size: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_size = input_size
        h1, h2, h3, h4 = hidden_sizes

        self.layer1 = nn.Linear(input_size, h1)
        self.layer2 = nn.Linear(h1, h2)

        self.layer3_from_1 = nn.Linear(h1, h3)
        self.layer3_from_2 = nn.Linear(h2, h3)
        self.layer3 = nn.Linear(h3, h3)

        self.layer4_from_2 = nn.Linear(h2, h4)
        self.layer4_from_3 = nn.Linear(h3, h4)

        self.layer4 = nn.Linear(h4, output_size)

        self.activation = getattr(F, activation)

    def forward(self, x):
        x = x.view(-1, self.input_size)

        x1 = self.activation(self.layer1(x))
        x2 = self.activation(self.layer2(x1))

        x3_input = self.layer3_from_1(x1) + self.layer3_from_2(x2)
        x3 = self.activation(self.layer3(x3_input))

        x4_input = self.layer4_from_2(x2) + self.layer4_from_3(x3)
        x4 = self.activation(x4_input)

        output = self.layer4(x4)
        return output

    def get_layer_names(self) -> list[str]:
        return [
            "layer1",
            "layer2",
            "layer3_from_1",
            "layer3_from_2",
            "layer3",
            "layer4_from_2",
            "layer4_from_3",
            "layer4",
        ]


class FourLayerSequentialNet(nn.Module):
    """
    Four-layer sequential architecture:
      - layer1 -> layer2
      - layer1, layer2 -> layer3 (integration point)
      - layer3 -> layer4 (sequential, no skip from layer2)
      - layer4 -> output
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, int, int, int],
        output_size: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_size = input_size
        h1, h2, h3, h4 = hidden_sizes

        self.layer1 = nn.Linear(input_size, h1)
        self.layer2 = nn.Linear(h1, h2)

        self.layer3_from_1 = nn.Linear(h1, h3)
        self.layer3_from_2 = nn.Linear(h2, h3)
        self.layer3 = nn.Linear(h3, h3)

        self.layer4_from_3 = nn.Linear(h3, h4)

        self.layer4 = nn.Linear(h4, output_size)

        self.activation = getattr(F, activation)

    def forward(self, x):
        x = x.view(-1, self.input_size)

        x1 = self.activation(self.layer1(x))
        x2 = self.activation(self.layer2(x1))

        x3_input = self.layer3_from_1(x1) + self.layer3_from_2(x2)
        x3 = self.activation(self.layer3(x3_input))

        x4_input = self.layer4_from_3(x3)
        x4 = self.activation(x4_input)

        output = self.layer4(x4)
        return output

    def get_layer_names(self) -> list[str]:
        return [
            "layer1",
            "layer2",
            "layer3_from_1",
            "layer3_from_2",
            "layer3",
            "layer4_from_3",
            "layer4",
        ]


MODEL_REGISTRY = {
    "three_layer_skip": ThreeLayerSkipNet,
    "four_layer_integrating": FourLayerIntegratingNet,
    "four_layer_sequential": FourLayerSequentialNet,
}


def build_model(
    architecture: str,
    input_size: int,
    hidden_sizes: tuple[int, ...],
    output_size: int,
    activation: str = "relu",
    layer_types: dict[str, str] | None = None,
) -> nn.Module:
    try:
        model_cls = MODEL_REGISTRY[architecture]
    except KeyError as exc:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available options: {', '.join(sorted(MODEL_REGISTRY))}"
        ) from exc

    # Only ThreeLayerSkipNet supports layer_types for now
    if architecture == "three_layer_skip":
        return model_cls(
            input_size=input_size,
            hidden_sizes=hidden_sizes,  # type: ignore[arg-type]
            output_size=output_size,
            activation=activation,
            layer_types=layer_types,
        )
    else:
        return model_cls(
            input_size=input_size,
            hidden_sizes=hidden_sizes,  # type: ignore[arg-type]
            output_size=output_size,
            activation=activation,
        )