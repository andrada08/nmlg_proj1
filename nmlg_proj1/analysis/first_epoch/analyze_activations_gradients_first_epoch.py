#!/usr/bin/env python3
"""
Analyze activation and gradient tracking data from the first epoch.
Aggregates per-step data into summary statistics and saves to CSV.

Usage:
    python -m nmlg_proj1.analysis.first_epoch.analyze_activations_gradients_first_epoch --subfolder three_layer_skip_with_50_100_150
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_config(config_path):
    """Load config from a config file"""
    with open(config_path, "r") as f:
        return json.load(f)


def extract_hyperparams(config):
    """Extract key hyperparameters from config"""
    return {
        "n_epochs": config.get("n_epochs", "N/A"),
        "batch_size": config.get("batch_size", "N/A"),
        "optimizer": config.get("optimizer", "N/A"),
        "activation": config.get("activation", "N/A"),
        "architecture": config.get("architecture", "unknown"),
        "input_size": config.get("input_size", "N/A"),
        "hidden_sizes": str(config.get("hidden_sizes", "N/A")),
        "output_size": config.get("output_size", "N/A"),
        "ln_rate": config.get("ln_rate", "N/A"),
        "layer_lns": str(config.get("layer_lns", "N/A")),
    }


def compute_activation_metrics(tracking_data, activation_points):
    """Compute summary statistics for activations"""
    metrics = {}

    for point_name in activation_points:
        if point_name not in tracking_data["activations"]:
            continue

        point_data = tracking_data["activations"][point_name]

        # For each statistic (mean, std, norm, max, sparsity)
        for stat_name in ["mean", "std", "norm", "max", "sparsity"]:
            if stat_name not in point_data:
                continue

            values = point_data[stat_name]
            if not values:
                continue

            values_array = np.array(values)

            # Compute summary statistics
            metrics[f"{point_name}_{stat_name}_initial"] = values[0] if values else np.nan
            metrics[f"{point_name}_{stat_name}_final"] = values[-1] if values else np.nan
            metrics[f"{point_name}_{stat_name}_mean"] = np.mean(values_array)
            metrics[f"{point_name}_{stat_name}_std"] = np.std(values_array)
            metrics[f"{point_name}_{stat_name}_min"] = np.min(values_array)
            metrics[f"{point_name}_{stat_name}_max"] = np.max(values_array)

            # Compute trend (slope of linear fit)
            if len(values) > 1:
                x = np.arange(len(values))
                slope = np.polyfit(x, values_array, 1)[0]
                metrics[f"{point_name}_{stat_name}_trend"] = slope
            else:
                metrics[f"{point_name}_{stat_name}_trend"] = np.nan

            # Compute change (final - initial)
            if len(values) > 0:
                metrics[f"{point_name}_{stat_name}_change"] = values[-1] - values[0]
            else:
                metrics[f"{point_name}_{stat_name}_change"] = np.nan

    return metrics


def compute_gradient_metrics(tracking_data, layer_names):
    """Compute summary statistics for gradients"""
    metrics = {}

    for layer_name in layer_names:
        if layer_name not in tracking_data["gradients"]:
            continue

        values = tracking_data["gradients"][layer_name]
        if not values:
            continue

        values_array = np.array(values)

        # Compute summary statistics
        metrics[f"{layer_name}_grad_initial"] = values[0] if values else np.nan
        metrics[f"{layer_name}_grad_final"] = values[-1] if values else np.nan
        metrics[f"{layer_name}_grad_mean"] = np.mean(values_array)
        metrics[f"{layer_name}_grad_std"] = np.std(values_array)
        metrics[f"{layer_name}_grad_min"] = np.min(values_array)
        metrics[f"{layer_name}_grad_max"] = np.max(values_array)

        # Compute trend (slope of linear fit)
        if len(values) > 1:
            x = np.arange(len(values))
            slope = np.polyfit(x, values_array, 1)[0]
            metrics[f"{layer_name}_grad_trend"] = slope
        else:
            metrics[f"{layer_name}_grad_trend"] = np.nan

        # Compute change (final - initial)
        if len(values) > 0:
            metrics[f"{layer_name}_grad_change"] = values[-1] - values[0]
        else:
            metrics[f"{layer_name}_grad_change"] = np.nan

    return metrics


def compute_loss_accuracy_metrics(tracking_data):
    """Compute summary statistics for loss and accuracy"""
    metrics = {}

    if "loss" in tracking_data and tracking_data["loss"]:
        loss_values = np.array(tracking_data["loss"])
        metrics["loss_initial"] = tracking_data["loss"][0]
        metrics["loss_final"] = tracking_data["loss"][-1]
        metrics["loss_mean"] = np.mean(loss_values)
        metrics["loss_min"] = np.min(loss_values)
        metrics["loss_max"] = np.max(loss_values)
        metrics["loss_change"] = tracking_data["loss"][-1] - tracking_data["loss"][0]

        if len(loss_values) > 1:
            x = np.arange(len(loss_values))
            slope = np.polyfit(x, loss_values, 1)[0]
            metrics["loss_trend"] = slope
        else:
            metrics["loss_trend"] = np.nan

    if "accuracy" in tracking_data and tracking_data["accuracy"]:
        acc_values = np.array(tracking_data["accuracy"])
        metrics["accuracy_initial"] = tracking_data["accuracy"][0]
        metrics["accuracy_final"] = tracking_data["accuracy"][-1]
        metrics["accuracy_mean"] = np.mean(acc_values)
        metrics["accuracy_min"] = np.min(acc_values)
        metrics["accuracy_max"] = np.max(acc_values)
        metrics["accuracy_change"] = (
            tracking_data["accuracy"][-1] - tracking_data["accuracy"][0]
        )

        if len(acc_values) > 1:
            x = np.arange(len(acc_values))
            slope = np.polyfit(x, acc_values, 1)[0]
            metrics["accuracy_trend"] = slope
        else:
            metrics["accuracy_trend"] = np.nan

    return metrics


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


def main():
    # Parse command line arguments
    subfolder = ""

    if "--subfolder" in sys.argv:
        idx = sys.argv.index("--subfolder")
        if idx + 1 < len(sys.argv):
            subfolder = sys.argv[idx + 1]

    if not subfolder:
        print(
            "Error: --subfolder is required (e.g., --subfolder three_layer_skip_with_50_100_150)"
        )
        return

    # Find all tracking result directories
    base_dir = Path("outputs") / "activation_gradient_analysis_first_epoch" / subfolder
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return

    # Find all sweep directories
    tracking_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("sweep_")])

    if not tracking_dirs:
        print(f"No tracking result directories found in: {base_dir}")
        return

    print(f"Found {len(tracking_dirs)} tracking result directories")

    # Create results folder matching outputs structure
    results_folder = Path("results") / "activation_gradient_analysis_first_epoch" / subfolder
    results_folder.mkdir(parents=True, exist_ok=True)

    print(f"Saving analysis results to: {results_folder}")

    # Collect data from each run
    results = []

    for tracking_dir in tracking_dirs:
        tracking_file = tracking_dir / "activation_gradient_analysis_first_epoch.json"

        if not tracking_file.exists():
            print(f"Skipping {tracking_dir.name} - missing tracking file")
            continue

        try:
            # Load tracking data
            with open(tracking_file, "r") as f:
                tracking_data = json.load(f)

            # Load config from original training directory
            # Map: activation_gradient_analysis_first_epoch -> gradients_across_training
            original_dir = Path("outputs") / "gradients_across_training" / subfolder / tracking_dir.name
            config_path = original_dir / "config.json"

            if not config_path.exists():
                print(f"Warning: Config not found for {tracking_dir.name}, using defaults")
                config = {}
            else:
                config = load_config(config_path)

            # Extract hyperparameters
            hyperparams = extract_hyperparams(config)

            # Get architecture to determine activation points
            architecture = config.get("architecture", "three_layer_skip")
            activation_points = get_activation_points(architecture)

            # Get layer names from tracking data
            layer_names = list(tracking_data.get("gradients", {}).keys())

            # Compute metrics
            activation_metrics = compute_activation_metrics(tracking_data, activation_points)
            gradient_metrics = compute_gradient_metrics(tracking_data, layer_names)
            loss_acc_metrics = compute_loss_accuracy_metrics(tracking_data)

            # Create row for this run
            row = {"run_name": tracking_dir.name, **hyperparams, **activation_metrics, **gradient_metrics, **loss_acc_metrics}

            results.append(row)
            print(f"Loaded {tracking_dir.name}")

        except Exception as e:
            print(f"Error loading {tracking_dir.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not results:
        print("No valid results found")
        return

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by run name for consistent ordering
    df = df.sort_values("run_name")

    # Save to CSV
    output_file = results_folder / "activation_gradient_analysis_first_epoch.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\nSummary:")
    print(f"  Total runs analyzed: {len(df)}")
    print(f"  Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()

