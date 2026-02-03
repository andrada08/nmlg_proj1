import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeLayerSkipNet(nn.Module):
    """
    Original architecture:
      - layer1 -> layer2
      - layer2 + layer1 skip into layer3 (output)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, int, int],
        output_size: int,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_size = input_size
        h1, h2, h3 = hidden_sizes
        self.layer1 = nn.Linear(input_size, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.layer3_from_1 = nn.Linear(h1, h3)  # W13: layer1→layer3
        self.layer3_from_2 = nn.Linear(h2, h3)  # W23: layer2→layer3
        self.layer3 = nn.Linear(h3, output_size)  # Layer3 directly outputs to final size
        self.activation = getattr(F, activation)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x1 = self.activation(self.layer1(x))
        x2 = self.activation(self.layer2(x1))
        x3_combined = self.layer3_from_1(x1) + self.layer3_from_2(x2)
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
) -> nn.Module:
    try:
        model_cls = MODEL_REGISTRY[architecture]
    except KeyError as exc:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available options: {', '.join(sorted(MODEL_REGISTRY))}"
        ) from exc

    return model_cls(
        input_size=input_size,
        hidden_sizes=hidden_sizes,  # type: ignore[arg-type]
        output_size=output_size,
        activation=activation,
    )