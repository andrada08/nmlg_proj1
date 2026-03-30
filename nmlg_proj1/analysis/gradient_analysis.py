#!/usr/bin/env python3
"""Functions for computing gradient metrics from training history"""

from itertools import combinations


def compute_layer_parameter_counts(config):
    """
    Compute parameter counts for each layer based on config.
    Returns a dictionary mapping layer_name -> num_parameters
    """
    param_counts = {}

    architecture = config.get("architecture", "three_layer_skip")
    input_size = config.get("input_size", 784)
    hidden_sizes = config.get("hidden_sizes", [])
    output_size = config.get("output_size", 10)
    layer_types = config.get("layer_types", {})

    if architecture != "three_layer_skip":
        # For other architectures, we'd need to implement their parameter counting
        # For now, return empty dict and skip per-parameter analysis
        return param_counts

    if len(hidden_sizes) < 3:
        return param_counts

    h1, h2, h3 = hidden_sizes[0], hidden_sizes[1], hidden_sizes[2]
    layer1_type = layer_types.get("layer1", "linear")
    layer2_type = layer_types.get("layer2", "linear")

    # Default conv parameters
    conv_kernel_size = 3

    # Layer1
    if layer1_type == "conv":
        param_counts["layer1"] = 1 * h1 * conv_kernel_size * conv_kernel_size
        layer1_output_size = h1 * 28 * 28  # With padding=1, spatial size stays 28x28
    else:  # linear
        param_counts["layer1"] = input_size * h1
        layer1_output_size = h1

    # Layer2
    if layer2_type == "conv":
        if layer1_type == "conv":
            param_counts["layer2"] = h1 * h2 * conv_kernel_size * conv_kernel_size
            layer2_output_size = h2 * 28 * 28
        else:  # linear -> conv
            # This case is complex, approximate
            spatial_dim = 14  # Default assumption
            param_counts["layer2"] = 1 * h2 * conv_kernel_size * conv_kernel_size
            layer2_output_size = h2 * spatial_dim * spatial_dim
    else:  # linear
        if layer1_type == "conv":
            param_counts["layer2"] = layer1_output_size * h2
        else:  # linear -> linear
            param_counts["layer2"] = h1 * h2
        layer2_output_size = h2

    # Skip connections (always linear)
    param_counts["layer3_from_1"] = layer1_output_size * h3
    param_counts["layer3_from_2"] = layer2_output_size * h3

    # Layer3 (always linear, final output)
    param_counts["layer3"] = h3 * output_size

    return param_counts


def compute_gradient_metrics(training_history, config=None):
    """
    Compute gradient metrics from training history data.
    This should be called post-hoc in analyze_results.py

    Args:
        training_history: Dictionary containing 'gradients' with epoch-by-epoch data

    Returns:
        Dictionary of gradient metrics
    """
    metrics = {}

    if "gradients" not in training_history or len(training_history["gradients"]["epoch"]) == 0:
        # Return empty metrics if no gradient data available
        return metrics

    # Tunable parameters (margin/smoothing/windows)
    epsilon_rel = 0.05  # require A >= B*(1+epsilon) to be considered "above"
    smooth_window = 3  # simple moving average window for comparisons
    first_window_frac = 0.3  # use first 30% of epochs as "start" window
    last_window_frac = 0.3  # last 30% as "end" window
    majority_threshold = 0.7  # need at least 70% of window epochs to satisfy condition

    def smooth(vals, w):
        if w <= 1 or len(vals) <= 2:
            return vals
        out = []
        for i in range(len(vals)):
            s = max(0, i - (w - 1) // 2)
            e = min(len(vals), i + (w - 1) // 2 + 1)
            window = vals[s:e]
            out.append(sum(window) / max(1, len(window)))
        return out

    def gt_margin(a, b, eps):
        # a > b with relative margin
        return a > b * (1.0 + eps)

    def fraction_above(a_list, b_list, eps, start_idx, end_idx):
        if end_idx <= start_idx:
            return 0.0
        total = 0
        good = 0
        for i in range(start_idx, end_idx):
            total += 1
            if gt_margin(a_list[i], b_list[i], eps):
                good += 1
        return good / max(1, total)

    gradients = training_history["gradients"]

    # Get main layers (exclude skip connections)
    main_layers = [name for name in gradients if name != "epoch" and "from" not in name]

    # Compute per-parameter gradients post-hoc if config is provided
    if config is not None:
        param_counts = compute_layer_parameter_counts(config)
        if param_counts:
            # Add per-parameter gradients to the gradients dict
            for layer_name in main_layers:
                if layer_name in param_counts and layer_name in gradients:
                    num_params = param_counts[layer_name]
                    if num_params > 0:
                        # Compute per-parameter norm: raw_norm / sqrt(num_params)
                        raw_grads = gradients[layer_name]
                        per_param_grads = [g / (num_params**0.5) for g in raw_grads]
                        gradients[f"{layer_name}_per_param"] = per_param_grads

    # Filter out per_param versions from main_layers (they'll be analyzed separately)
    main_layers = [name for name in main_layers if not name.endswith("_per_param")]

    # Get per-parameter layer names (either from computed or existing)
    main_layers_per_param = [
        name.replace("_per_param", "") for name in gradients if name.endswith("_per_param")
    ]

    if len(main_layers) < 2:
        raise ValueError(
            "Expected at least two primary layers (layer*, without 'from') in gradients"
        )

    def layer_sort_key(name: str):
        digits = "".join(ch for ch in name if ch.isdigit())
        return (int(digits) if digits else float("inf"), name)

    main_layers.sort(key=layer_sort_key)

    grads_by_layer = {layer: gradients[layer] for layer in main_layers}

    # Smooth copies for comparisons
    smoothed_grads = {layer: smooth(vals, smooth_window) for layer, vals in grads_by_layer.items()}

    if any(len(vals) < 2 for vals in smoothed_grads.values()):
        return metrics

    # Large relative drops for each primary layer
    drop_threshold = 0.5  # 50% drop
    drops = {}
    for layer, grads in grads_by_layer.items():
        layer_drops = [
            (grads[i - 1] - grads[i]) / grads[i - 1]
            for i in range(1, len(grads))
            if grads[i - 1] > 0
        ]
        drops[layer] = layer_drops
        metrics[f"{layer}_large_drop"] = 1 if any(drop > drop_threshold for drop in layer_drops) else 0

    # Switches between pairs (margin-aware)
    def switches(a_list, b_list):
        n = min(len(a_list), len(b_list))
        if n < 2:
            return 0
        cnt = 0
        prev = gt_margin(a_list[0], b_list[0], epsilon_rel)
        for i in range(1, n):
            cur = gt_margin(a_list[i], b_list[i], epsilon_rel)
            if cur != prev:
                cnt += 1
            prev = cur
        return cnt

    # Temporal order metric: Check if A above B happened before B above A
    def temporal_order(a_list, b_list):
        """Returns (1, epoch) if A was above B (with margin) before B was above A (with margin), else (0, None).
        epoch is the epoch where B first becomes above A (the switch point)."""
        n = min(len(a_list), len(b_list))
        if n == 0:
            return (0, None)

        first_a_above_b = None
        first_b_above_a = None

        for i in range(n):
            if first_a_above_b is None and gt_margin(a_list[i], b_list[i], epsilon_rel):
                first_a_above_b = i
            if first_b_above_a is None and gt_margin(b_list[i], a_list[i], epsilon_rel):
                first_b_above_a = i
            # If we found both, we can stop
            if first_a_above_b is not None and first_b_above_a is not None:
                break

        # Pattern requires A above B happened before B above A
        if first_a_above_b is not None and first_b_above_a is not None:
            if first_a_above_b < first_b_above_a:
                return (1, first_b_above_a)  # Return the epoch where B becomes above A
            else:
                return (0, None)
        return (0, None)

    # Function to find the largest single-epoch jump in accuracy throughout training
    def check_accuracy_boost(pattern_epoch, test_accuracy_list, boost_threshold=0.5):
        """
        Find the largest single-epoch jump in accuracy throughout the entire training.
        This is independent of the pattern detection epoch.

        Args:
            pattern_epoch: Epoch where pattern was detected (0-indexed) - kept for compatibility but not used
            test_accuracy_list: List of test accuracies per epoch
            boost_threshold: Minimum accuracy increase (in percentage points) to count as boost

        Returns:
            (boost_detected, boost_magnitude, boost_epoch)
            - boost_detected: 1 if boost detected, 0 otherwise
            - boost_magnitude: Largest single-epoch jump found (percentage points)
            - boost_epoch: Epoch where the jump occurs (the epoch AFTER the jump, 0-indexed)
        """
        if not test_accuracy_list or len(test_accuracy_list) < 2:
            return (0, 0.0, None)

        # Find the largest single-epoch jump throughout training
        max_jump = 0.0
        max_jump_epoch = None

        for i in range(1, len(test_accuracy_list)):
            jump = test_accuracy_list[i] - test_accuracy_list[i - 1]
            if jump > max_jump:
                max_jump = jump
                max_jump_epoch = i  # Epoch where the jump occurred (after the increase)

        # Check if jump exceeds threshold
        boost_detected = 1 if max_jump >= boost_threshold else 0

        return (boost_detected, max_jump, max_jump_epoch)

    def compute_boost_pattern_alignment(pattern_epoch, boost_epoch, alignment_window=3):
        """
        Compute alignment metrics between pattern detection epoch and accuracy boost epoch.

        Args:
            pattern_epoch: Epoch where pattern was detected (0-indexed), or None/-1 if no pattern
            boost_epoch: Epoch where accuracy boost occurred (0-indexed), or None/-1 if no boost
            alignment_window: Number of epochs on either side to consider "aligned" (default ±3)

        Returns:
            (epoch_diff, abs_epoch_diff, aligned)
            - epoch_diff: boost_epoch - pattern_epoch (signed, negative means boost before pattern)
            - abs_epoch_diff: Absolute difference
            - aligned: 1 if within ±alignment_window epochs, 0 otherwise
        """
        if (
            pattern_epoch is None
            or pattern_epoch == -1
            or boost_epoch is None
            or boost_epoch == -1
        ):
            return (-999, -999, 0)  # Use -999 as sentinel for missing data

        epoch_diff = boost_epoch - pattern_epoch
        abs_epoch_diff = abs(epoch_diff)
        aligned = 1 if abs_epoch_diff <= alignment_window else 0

        return (epoch_diff, abs_epoch_diff, aligned)

    # Combined patterns for all layer pairs: composite of component metrics (unchanged except margin-aware upstream)
    def check_pattern_composite(
        layerA_above_B,
        layerB_above_A,
        switches_AB,
        layerA_large_drop,
        temporal_order_result,
    ):
        """Pattern occurs if all component metrics are present AND the temporal order is correct (A above B happened before B above A).
        Returns (1, epoch) if pattern detected, else (0, None). epoch is where the pattern switch occurs."""
        temporal_order_AB, pattern_epoch = temporal_order_result
        if (
            layerA_above_B
            and layerB_above_A
            and switches_AB
            and layerA_large_drop
            and temporal_order_AB
        ):
            return (1, pattern_epoch)
        return (0, None)

    # Strict temporal pattern: checks specific epoch-1, epoch, epoch+1 sequence
    # Uses smoothed gradients and margin-aware comparisons
    def check_pattern_strict(layerA_smoothed, layerB_smoothed, layerA_drops):
        """
        Strict pattern: A above B at epoch-1, B above A at epoch, AND A has large drop at epoch-1.
        This is a very specific temporal sequence that must occur consecutively.
        Returns (1, epoch) if pattern detected at epoch i, else (0, None).
        """
        # Need at least 2 epochs to check epoch-1 and epoch
        if len(layerA_smoothed) < 2 or len(layerA_drops) < 1:
            return (0, None)

        # Check for the specific sequence: epoch-1 (A > B), epoch (B > A), and large drop at epoch-1
        for i in range(1, len(layerA_smoothed)):
            # epoch-1: A above B (with margin)
            # epoch: B above A (with margin)
            # A has large drop at epoch-1
            if (
                gt_margin(layerA_smoothed[i - 1], layerB_smoothed[i - 1], epsilon_rel)
                and gt_margin(layerB_smoothed[i], layerA_smoothed[i], epsilon_rel)
                and i - 1 < len(layerA_drops)
                and layerA_drops[i - 1] > drop_threshold
            ):
                return (1, i)  # Pattern detected at epoch i
        return (0, None)

    # Get test accuracy list for accuracy boost analysis
    test_accuracy_list = training_history.get("test_accuracy", [])

    # Pairwise metrics for all unordered layer pairs
    for layer_a, layer_b in combinations(main_layers, 2):
        sm_a = smoothed_grads[layer_a]
        sm_b = smoothed_grads[layer_b]
        n = min(len(sm_a), len(sm_b))
        if n < 2:
            continue

        # Directional comparisons
        metrics[f"{layer_a}_above_{layer_b}"] = (
            1 if any(gt_margin(sm_a[i], sm_b[i], epsilon_rel) for i in range(n)) else 0
        )
        metrics[f"{layer_b}_above_{layer_a}"] = (
            1 if any(gt_margin(sm_b[i], sm_a[i], epsilon_rel) for i in range(n)) else 0
        )

        # Switches
        switches_ab = switches(sm_a, sm_b)
        metrics[f"switches_{layer_a}_{layer_b}"] = 1 if switches_ab > 0 else 0

        # Temporal order (both directions) - now returns (value, epoch)
        temporal_order_ab_result = temporal_order(sm_a, sm_b)
        temporal_order_ba_result = temporal_order(sm_b, sm_a)
        metrics[f"{layer_a}vs{layer_b}_temporal_order"] = temporal_order_ab_result[0]
        metrics[f"{layer_b}vs{layer_a}_temporal_order"] = temporal_order_ba_result[0]

        # Composite patterns for both directions - now returns (value, epoch)
        pattern_ab_result = check_pattern_composite(
            metrics[f"{layer_a}_above_{layer_b}"],
            metrics[f"{layer_b}_above_{layer_a}"],
            metrics[f"switches_{layer_a}_{layer_b}"],
            metrics[f"{layer_a}_large_drop"],
            temporal_order_ab_result,
        )
        pattern_ba_result = check_pattern_composite(
            metrics[f"{layer_b}_above_{layer_a}"],
            metrics[f"{layer_a}_above_{layer_b}"],
            metrics[f"switches_{layer_a}_{layer_b}"],
            metrics[f"{layer_b}_large_drop"],
            temporal_order_ba_result,
        )
        metrics[f"{layer_a}vs{layer_b}_pattern"] = pattern_ab_result[0]
        metrics[f"{layer_b}vs{layer_a}_pattern"] = pattern_ba_result[0]
        # Store pattern detection epochs
        metrics[f"{layer_a}vs{layer_b}_pattern_epoch"] = (
            pattern_ab_result[1] if pattern_ab_result[1] is not None else -1
        )
        metrics[f"{layer_b}vs{layer_a}_pattern_epoch"] = (
            pattern_ba_result[1] if pattern_ba_result[1] is not None else -1
        )

        # Strict patterns - now returns (value, epoch)
        strict_ab_result = check_pattern_strict(sm_a, sm_b, drops[layer_a])
        strict_ba_result = check_pattern_strict(sm_b, sm_a, drops[layer_b])
        metrics[f"{layer_a}vs{layer_b}_strict_pattern"] = strict_ab_result[0]
        metrics[f"{layer_b}vs{layer_a}_strict_pattern"] = strict_ba_result[0]
        # Store strict pattern detection epochs
        metrics[f"{layer_a}vs{layer_b}_strict_pattern_epoch"] = (
            strict_ab_result[1] if strict_ab_result[1] is not None else -1
        )
        metrics[f"{layer_b}vs{layer_a}_strict_pattern_epoch"] = (
            strict_ba_result[1] if strict_ba_result[1] is not None else -1
        )

        # Strict patterns must be subset of composite patterns
        if metrics[f"{layer_a}vs{layer_b}_pattern"] == 0:
            metrics[f"{layer_a}vs{layer_b}_strict_pattern"] = 0
            strict_ab_result = (0, None)
        if metrics[f"{layer_b}vs{layer_a}_pattern"] == 0:
            metrics[f"{layer_b}vs{layer_a}_strict_pattern"] = 0
            strict_ba_result = (0, None)

        # Check for accuracy boost around pattern detection epochs
        # For composite patterns, use the pattern epoch
        if pattern_ab_result[0] == 1 and pattern_ab_result[1] is not None:
            boost_result = check_accuracy_boost(pattern_ab_result[1], test_accuracy_list)
            metrics[f"{layer_a}vs{layer_b}_pattern_accuracy_boost"] = boost_result[0]
            metrics[f"{layer_a}vs{layer_b}_pattern_accuracy_boost_magnitude"] = boost_result[1]
            metrics[f"{layer_a}vs{layer_b}_pattern_accuracy_boost_epoch"] = (
                boost_result[2] if boost_result[2] is not None else -1
            )

            # Compute alignment metrics
            alignment_result = compute_boost_pattern_alignment(
                pattern_ab_result[1],
                boost_result[2] if boost_result[2] is not None else -1,
            )
            metrics[f"{layer_a}vs{layer_b}_pattern_boost_pattern_epoch_diff"] = alignment_result[0]
            metrics[f"{layer_a}vs{layer_b}_pattern_boost_pattern_epoch_abs_diff"] = alignment_result[1]
            metrics[f"{layer_a}vs{layer_b}_pattern_boost_pattern_aligned"] = alignment_result[2]
        else:
            metrics[f"{layer_a}vs{layer_b}_pattern_accuracy_boost"] = 0
            metrics[f"{layer_a}vs{layer_b}_pattern_accuracy_boost_magnitude"] = 0.0
            metrics[f"{layer_a}vs{layer_b}_pattern_accuracy_boost_epoch"] = -1
            metrics[f"{layer_a}vs{layer_b}_pattern_boost_pattern_epoch_diff"] = -999
            metrics[f"{layer_a}vs{layer_b}_pattern_boost_pattern_epoch_abs_diff"] = -999
            metrics[f"{layer_a}vs{layer_b}_pattern_boost_pattern_aligned"] = 0

        if pattern_ba_result[0] == 1 and pattern_ba_result[1] is not None:
            boost_result = check_accuracy_boost(pattern_ba_result[1], test_accuracy_list)
            metrics[f"{layer_b}vs{layer_a}_pattern_accuracy_boost"] = boost_result[0]
            metrics[f"{layer_b}vs{layer_a}_pattern_accuracy_boost_magnitude"] = boost_result[1]
            metrics[f"{layer_b}vs{layer_a}_pattern_accuracy_boost_epoch"] = (
                boost_result[2] if boost_result[2] is not None else -1
            )

            # Compute alignment metrics
            alignment_result = compute_boost_pattern_alignment(
                pattern_ba_result[1],
                boost_result[2] if boost_result[2] is not None else -1,
            )
            metrics[f"{layer_b}vs{layer_a}_pattern_boost_pattern_epoch_diff"] = alignment_result[0]
            metrics[f"{layer_b}vs{layer_a}_pattern_boost_pattern_epoch_abs_diff"] = alignment_result[1]
            metrics[f"{layer_b}vs{layer_a}_pattern_boost_pattern_aligned"] = alignment_result[2]
        else:
            metrics[f"{layer_b}vs{layer_a}_pattern_accuracy_boost"] = 0
            metrics[f"{layer_b}vs{layer_a}_pattern_accuracy_boost_magnitude"] = 0.0
            metrics[f"{layer_b}vs{layer_a}_pattern_accuracy_boost_epoch"] = -1
            metrics[f"{layer_b}vs{layer_a}_pattern_boost_pattern_epoch_diff"] = -999
            metrics[f"{layer_b}vs{layer_a}_pattern_boost_pattern_epoch_abs_diff"] = -999
            metrics[f"{layer_b}vs{layer_a}_pattern_boost_pattern_aligned"] = 0

        # For strict patterns
        if strict_ab_result[0] == 1 and strict_ab_result[1] is not None:
            boost_result = check_accuracy_boost(strict_ab_result[1], test_accuracy_list)
            metrics[f"{layer_a}vs{layer_b}_strict_pattern_accuracy_boost"] = boost_result[0]
            metrics[f"{layer_a}vs{layer_b}_strict_pattern_accuracy_boost_magnitude"] = boost_result[1]
            metrics[f"{layer_a}vs{layer_b}_strict_pattern_accuracy_boost_epoch"] = (
                boost_result[2] if boost_result[2] is not None else -1
            )

            # Compute alignment metrics
            alignment_result = compute_boost_pattern_alignment(
                strict_ab_result[1],
                boost_result[2] if boost_result[2] is not None else -1,
            )
            metrics[f"{layer_a}vs{layer_b}_strict_pattern_boost_pattern_epoch_diff"] = alignment_result[0]
            metrics[f"{layer_a}vs{layer_b}_strict_pattern_boost_pattern_epoch_abs_diff"] = alignment_result[1]
            metrics[f"{layer_a}vs{layer_b}_strict_pattern_boost_pattern_aligned"] = alignment_result[2]
        else:
            metrics[f"{layer_a}vs{layer_b}_strict_pattern_accuracy_boost"] = 0
            metrics[f"{layer_a}vs{layer_b}_strict_pattern_accuracy_boost_magnitude"] = 0.0
            metrics[f"{layer_a}vs{layer_b}_strict_pattern_accuracy_boost_epoch"] = -1
            metrics[f"{layer_a}vs{layer_b}_strict_pattern_boost_pattern_epoch_diff"] = -999
            metrics[f"{layer_a}vs{layer_b}_strict_pattern_boost_pattern_epoch_abs_diff"] = -999
            metrics[f"{layer_a}vs{layer_b}_strict_pattern_boost_pattern_aligned"] = 0

        if strict_ba_result[0] == 1 and strict_ba_result[1] is not None:
            boost_result = check_accuracy_boost(strict_ba_result[1], test_accuracy_list)
            metrics[f"{layer_b}vs{layer_a}_strict_pattern_accuracy_boost"] = boost_result[0]
            metrics[f"{layer_b}vs{layer_a}_strict_pattern_accuracy_boost_magnitude"] = boost_result[1]
            metrics[f"{layer_b}vs{layer_a}_strict_pattern_accuracy_boost_epoch"] = (
                boost_result[2] if boost_result[2] is not None else -1
            )

            # Compute alignment metrics
            alignment_result = compute_boost_pattern_alignment(
                strict_ba_result[1],
                boost_result[2] if boost_result[2] is not None else -1,
            )
            metrics[f"{layer_b}vs{layer_a}_strict_pattern_boost_pattern_epoch_diff"] = alignment_result[0]
            metrics[f"{layer_b}vs{layer_a}_strict_pattern_boost_pattern_epoch_abs_diff"] = alignment_result[1]
            metrics[f"{layer_b}vs{layer_a}_strict_pattern_boost_pattern_aligned"] = alignment_result[2]
        else:
            metrics[f"{layer_b}vs{layer_a}_strict_pattern_accuracy_boost"] = 0
            metrics[f"{layer_b}vs{layer_a}_strict_pattern_accuracy_boost_magnitude"] = 0.0
            metrics[f"{layer_b}vs{layer_a}_strict_pattern_accuracy_boost_epoch"] = -1
            metrics[f"{layer_b}vs{layer_a}_strict_pattern_boost_pattern_epoch_diff"] = -999
            metrics[f"{layer_b}vs{layer_a}_strict_pattern_boost_pattern_epoch_abs_diff"] = -999
            metrics[f"{layer_b}vs{layer_a}_strict_pattern_boost_pattern_aligned"] = 0

    # Extract final accuracies
    if "accuracy" in training_history and training_history["accuracy"]:
        metrics["final_train_accuracy"] = training_history["accuracy"][-1]
    if "test_accuracy" in training_history and training_history["test_accuracy"]:
        metrics["final_test_accuracy"] = training_history["test_accuracy"][-1]

    # Now analyze per-parameter gradients if available
    if main_layers_per_param and len(main_layers_per_param) >= 2:
        # Get per-parameter gradients
        grads_per_param_by_layer = {
            layer: gradients[f"{layer}_per_param"]
            for layer in main_layers_per_param
            if f"{layer}_per_param" in gradients
        }

        if len(grads_per_param_by_layer) >= 2:
            # Sort layers
            main_layers_per_param_sorted = sorted(main_layers_per_param, key=layer_sort_key)

            # Smooth per-parameter gradients
            smoothed_grads_per_param = {
                layer: smooth(vals, smooth_window)
                for layer, vals in grads_per_param_by_layer.items()
            }

            if all(len(vals) >= 2 for vals in smoothed_grads_per_param.values()):
                # Large drops for per-parameter gradients
                drops_per_param = {}
                for layer, grads in grads_per_param_by_layer.items():
                    layer_drops = [
                        (grads[i - 1] - grads[i]) / grads[i - 1]
                        for i in range(1, len(grads))
                        if grads[i - 1] > 0
                    ]
                    drops_per_param[layer] = layer_drops
                    metrics[f"{layer}_per_param_large_drop"] = (
                        1 if any(drop > drop_threshold for drop in layer_drops) else 0
                    )

                # Pairwise metrics for per-parameter gradients
                for layer_a, layer_b in combinations(main_layers_per_param_sorted, 2):
                    if (
                        layer_a not in smoothed_grads_per_param
                        or layer_b not in smoothed_grads_per_param
                    ):
                        continue

                    sm_a = smoothed_grads_per_param[layer_a]
                    sm_b = smoothed_grads_per_param[layer_b]
                    n = min(len(sm_a), len(sm_b))
                    if n < 2:
                        continue

                    # Directional comparisons
                    metrics[f"{layer_a}_per_param_above_{layer_b}"] = (
                        1 if any(gt_margin(sm_a[i], sm_b[i], epsilon_rel) for i in range(n)) else 0
                    )
                    metrics[f"{layer_b}_per_param_above_{layer_a}"] = (
                        1 if any(gt_margin(sm_b[i], sm_a[i], epsilon_rel) for i in range(n)) else 0
                    )

                    # Switches
                    switches_ab = switches(sm_a, sm_b)
                    metrics[f"per_param_switches_{layer_a}_{layer_b}"] = (
                        1 if switches_ab > 0 else 0
                    )

                    # Temporal order
                    temporal_order_ab_result = temporal_order(sm_a, sm_b)
                    temporal_order_ba_result = temporal_order(sm_b, sm_a)
                    metrics[f"{layer_a}_per_param_vs_{layer_b}_temporal_order"] = temporal_order_ab_result[0]
                    metrics[f"{layer_b}_per_param_vs_{layer_a}_temporal_order"] = temporal_order_ba_result[0]

                    # Composite patterns
                    pattern_ab_result = check_pattern_composite(
                        metrics[f"{layer_a}_per_param_above_{layer_b}"],
                        metrics[f"{layer_b}_per_param_above_{layer_a}"],
                        metrics[f"per_param_switches_{layer_a}_{layer_b}"],
                        metrics[f"{layer_a}_per_param_large_drop"],
                        temporal_order_ab_result,
                    )
                    pattern_ba_result = check_pattern_composite(
                        metrics[f"{layer_b}_per_param_above_{layer_a}"],
                        metrics[f"{layer_a}_per_param_above_{layer_b}"],
                        metrics[f"per_param_switches_{layer_a}_{layer_b}"],
                        metrics[f"{layer_b}_per_param_large_drop"],
                        temporal_order_ba_result,
                    )
                    metrics[f"{layer_a}_per_param_vs_{layer_b}_pattern"] = pattern_ab_result[0]
                    metrics[f"{layer_b}_per_param_vs_{layer_a}_pattern"] = pattern_ba_result[0]
                    metrics[f"{layer_a}_per_param_vs_{layer_b}_pattern_epoch"] = (
                        pattern_ab_result[1] if pattern_ab_result[1] is not None else -1
                    )
                    metrics[f"{layer_b}_per_param_vs_{layer_a}_pattern_epoch"] = (
                        pattern_ba_result[1] if pattern_ba_result[1] is not None else -1
                    )

                    # Strict patterns
                    strict_ab_result = check_pattern_strict(
                        sm_a, sm_b, drops_per_param[layer_a]
                    )
                    strict_ba_result = check_pattern_strict(
                        sm_b, sm_a, drops_per_param[layer_b]
                    )
                    metrics[f"{layer_a}_per_param_vs_{layer_b}_strict_pattern"] = strict_ab_result[0]
                    metrics[f"{layer_b}_per_param_vs_{layer_a}_strict_pattern"] = strict_ba_result[0]
                    metrics[f"{layer_a}_per_param_vs_{layer_b}_strict_pattern_epoch"] = (
                        strict_ab_result[1] if strict_ab_result[1] is not None else -1
                    )
                    metrics[f"{layer_b}_per_param_vs_{layer_a}_strict_pattern_epoch"] = (
                        strict_ba_result[1] if strict_ba_result[1] is not None else -1
                    )

                    # Strict patterns must be subset of composite patterns
                    if metrics[f"{layer_a}_per_param_vs_{layer_b}_pattern"] == 0:
                        metrics[f"{layer_a}_per_param_vs_{layer_b}_strict_pattern"] = 0
                        strict_ab_result = (0, None)
                    if metrics[f"{layer_b}_per_param_vs_{layer_a}_pattern"] == 0:
                        metrics[f"{layer_b}_per_param_vs_{layer_a}_strict_pattern"] = 0
                        strict_ba_result = (0, None)

    return metrics

