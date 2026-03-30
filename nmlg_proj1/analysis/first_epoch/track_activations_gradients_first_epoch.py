#!/usr/bin/env python3
"""
Track activations and gradients for the first epoch (epoch 0) of training.
This script reconstructs the model and training setup from a saved config,
re-runs epoch 0 with the same seed, and tracks activations and gradients per step.

Usage:
    python -m nmlg_proj1.analysis.first_epoch.track_activations_gradients_first_epoch outputs/subfolder/sweep_name/
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from nmlg_proj1.data.load_data import load_data
from nmlg_proj1.models.nets import build_model


def get_activation_points(architecture: str) -> list[str]:
    """Return activation point names for each architecture"""
    if architecture == "three_layer_skip":
        return ["x1", "x2", "x3_combined"]
    elif architecture == "four_layer_integrating":
        return ["x1", "x2", "x3_input", "x3", "x4_input", "x4"]
    elif architecture == "four_layer_sequential":
        return ["x1", "x2", "x3_input", "x3", "x4_input", "x4"]
    else:
        return []


def compute_activation_stats(activation_tensor: torch.Tensor) -> dict:
    """Compute statistics for an activation tensor"""
    with torch.no_grad():
        mean = activation_tensor.mean().item()
        std = activation_tensor.std().item()
        norm = activation_tensor.norm().item()
        max_val = activation_tensor.max().item()
        # Sparsity: fraction of zeros (useful for ReLU)
        sparsity = (activation_tensor == 0).float().mean().item()
    return {"mean": mean, "std": std, "norm": norm, "max": max_val, "sparsity": sparsity}


def track_first_epoch(output_dir: str):
    """
    Track activations and gradients for the first epoch.

    Args:
        output_dir: Path to output directory containing config.json
    """
    output_dir = Path(output_dir)
    config_path = output_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Set same random seed as training
    np.random.seed(0)
    torch.manual_seed(0)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Unpack config
    batch_size = cfg["batch_size"]
    ln_rate = cfg.get("ln_rate", 1e-3)
    opt_class = getattr(optim, cfg["optimizer"])
    input_size = cfg["input_size"]
    hidden_sizes = cfg["hidden_sizes"]
    output_size = cfg["output_size"]
    layer_lns = cfg.get("layer_lns")
    activation = cfg["activation"]
    architecture = cfg.get("architecture", "three_layer_skip")

    # Build model
    print(f"\nBuilding {architecture} model...")
    model = build_model(
        architecture=architecture,
        input_size=input_size,
        hidden_sizes=tuple(hidden_sizes),
        output_size=output_size,
        activation=activation,
    )
    model = model.to(device)

    # Get layer names and activation points
    layer_names = model.get_layer_names()
    activation_points = get_activation_points(architecture)

    print(f"Layer names: {layer_names}")
    print(f"Activation points to track: {activation_points}")

    # Setup optimizer (same as training)
    if layer_lns:
        groups = []
        for name, lr in layer_lns.items():
            if not hasattr(model, name):
                raise AttributeError(f"Layer '{name}' not found in model")
            groups.append({"params": getattr(model, name).parameters(), "lr": lr})
        for layer_name in layer_names:
            if layer_name not in layer_lns:
                groups.append({"params": getattr(model, layer_name).parameters(), "lr": ln_rate})
        optimizer = opt_class(groups)
    else:
        optimizer = opt_class(model.parameters(), lr=ln_rate)

    criterion = nn.CrossEntropyLoss()

    # Load data
    print("\nLoading MNIST data...")
    train_loader, _ = load_data(batch_size)

    # Initialize tracking data structure
    tracking_data = {
        "epoch": 0,
        "steps": [],
        "activations": {
            point: {stat: [] for stat in ["mean", "std", "norm", "max", "sparsity"]}
            for point in activation_points
        },
        "gradients": {name: [] for name in layer_names},
        "loss": [],
        "accuracy": [],
    }

    # Storage for activations during forward pass
    activation_storage = {}

    original_forward = model.forward

    def forward_with_tracking(x):
        """Modified forward that captures activations"""
        x = x.view(-1, model.input_size)

        if architecture == "three_layer_skip":
            x1 = model.activation(model.layer1(x))
            activation_storage["x1"] = x1.detach().cpu()

            x2 = model.activation(model.layer2(x1))
            activation_storage["x2"] = x2.detach().cpu()

            x3_combined = model.layer3_from_1(x1) + model.layer3_from_2(x2)
            activation_storage["x3_combined"] = x3_combined.detach().cpu()

            output = model.layer3(x3_combined)
            return output

        elif architecture == "four_layer_integrating":
            x1 = model.activation(model.layer1(x))
            activation_storage["x1"] = x1.detach().cpu()

            x2 = model.activation(model.layer2(x1))
            activation_storage["x2"] = x2.detach().cpu()

            x3_input = model.layer3_from_1(x1) + model.layer3_from_2(x2)
            activation_storage["x3_input"] = x3_input.detach().cpu()

            x3 = model.activation(model.layer3(x3_input))
            activation_storage["x3"] = x3.detach().cpu()

            x4_input = model.layer4_from_2(x2) + model.layer4_from_3(x3)
            activation_storage["x4_input"] = x4_input.detach().cpu()

            x4 = model.activation(x4_input)
            activation_storage["x4"] = x4.detach().cpu()

            output = model.layer4(x4)
            return output

        elif architecture == "four_layer_sequential":
            x1 = model.activation(model.layer1(x))
            activation_storage["x1"] = x1.detach().cpu()

            x2 = model.activation(model.layer2(x1))
            activation_storage["x2"] = x2.detach().cpu()

            x3_input = model.layer3_from_1(x1) + model.layer3_from_2(x2)
            activation_storage["x3_input"] = x3_input.detach().cpu()

            x3 = model.activation(model.layer3(x3_input))
            activation_storage["x3"] = x3.detach().cpu()

            x4_input = model.layer4_from_3(x3)
            activation_storage["x4_input"] = x4_input.detach().cpu()

            x4 = model.activation(x4_input)
            activation_storage["x4"] = x4.detach().cpu()

            output = model.layer4(x4)
            return output
        else:
            return original_forward(x)

    # Replace forward method
    model.forward = forward_with_tracking

    # Run epoch 0
    print("\nRunning epoch 0 and tracking activations/gradients...")
    model.train()

    step = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Clear activation storage (will be populated during forward pass)
        activation_storage.clear()

        # Force garbage collection periodically to help with memory
        if step % 50 == 0:
            import gc

            gc.collect()

        # Forward pass (captures activations)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Track activations for this step
        for point_name in activation_points:
            if point_name in activation_storage:
                # Activation is already on CPU, compute stats
                stats = compute_activation_stats(activation_storage[point_name])
                for stat_name, stat_value in stats.items():
                    tracking_data["activations"][point_name][stat_name].append(stat_value)
                # Clear from storage to free memory
                del activation_storage[point_name]

        # Backward pass
        loss.backward()

        # Track gradients for this step
        for layer_name in layer_names:
            layer = getattr(model, layer_name)
            if layer.weight is not None and layer.weight.grad is not None:
                grad_norm = layer.weight.grad.norm().item()
                tracking_data["gradients"][layer_name].append(grad_norm)
            else:
                tracking_data["gradients"][layer_name].append(0.0)

        # Update weights
        optimizer.step()

        # Track loss and accuracy
        tracking_data["loss"].append(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100.0 * correct / target.size(0)
        tracking_data["accuracy"].append(accuracy)
        tracking_data["steps"].append(step)

        step += 1

        if (batch_idx + 1) % 100 == 0:
            print(f"  Step {batch_idx + 1}/{len(train_loader)}")

    # Restore original forward
    model.forward = original_forward

    # Determine save location: mirror structure in activation_gradient_analysis_first_epoch folder
    # Input: outputs/gradients_across_training/subfolder/sweep_name/
    # Output: outputs/activation_gradient_analysis_first_epoch/subfolder/sweep_name/
    output_dir_str = str(output_dir)

    # Extract subfolder and sweep_name from path
    if "gradients_across_training" in output_dir_str:
        # Replace gradients_across_training with activation_gradient_analysis_first_epoch
        parts = output_dir_str.split("gradients_across_training")
        if len(parts) == 2:
            save_base = Path("outputs") / "activation_gradient_analysis_first_epoch"
            # Get the relative path after gradients_across_training
            relative_path = parts[1].lstrip("/")
            save_dir = save_base / relative_path
        else:
            # Fallback: try to extract from path components
            path_parts = Path(output_dir).parts
            if "outputs" in path_parts:
                idx = path_parts.index("outputs")
                if idx + 1 < len(path_parts):
                    # Get everything after outputs
                    subfolder = path_parts[idx + 1]
                    sweep_name = path_parts[-1] if len(path_parts) > idx + 2 else "unknown"
                    save_dir = (
                        Path("outputs")
                        / "activation_gradient_analysis_first_epoch"
                        / subfolder
                        / sweep_name
                    )
                else:
                    save_dir = Path("outputs") / "activation_gradient_analysis_first_epoch" / "unknown"
            else:
                save_dir = Path("outputs") / "activation_gradient_analysis_first_epoch" / "unknown"
    else:
        # If path doesn't contain gradients_across_training, try to infer structure
        # Assume format: outputs/subfolder/sweep_name/
        path_parts = Path(output_dir).parts
        if "outputs" in path_parts:
            idx = path_parts.index("outputs")
            if idx + 2 < len(path_parts):
                subfolder = path_parts[idx + 1]
                sweep_name = path_parts[-1]
                save_dir = (
                    Path("outputs")
                    / "activation_gradient_analysis_first_epoch"
                    / subfolder
                    / sweep_name
                )
            else:
                save_dir = Path("outputs") / "activation_gradient_analysis_first_epoch" / "unknown"
        else:
            save_dir = Path("outputs") / "activation_gradient_analysis_first_epoch" / "unknown"

    # Create directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    output_file = save_dir / "activation_gradient_analysis_first_epoch.json"
    with open(output_file, "w") as f:
        json.dump(tracking_data, f, indent=2)

    print("\n✓ Tracking complete!")
    print(f"  Total steps: {len(tracking_data['steps'])}")
    print(f"  Activation points tracked: {len(activation_points)}")
    print(f"  Layers tracked: {len(layer_names)}")
    print(f"  Results saved to: {output_file}")

    return tracking_data


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python -m nmlg_proj1.analysis.first_epoch.track_activations_gradients_first_epoch <output_dir>"
        )
        print(
            "Example: python -m nmlg_proj1.analysis.first_epoch.track_activations_gradients_first_epoch "
            "outputs/gradients_across_training/three_layer_skip_with_50_100_150/sweep_.../"
        )
        sys.exit(1)

    output_dir = sys.argv[1]
    track_first_epoch(output_dir)


if __name__ == "__main__":
    main()

