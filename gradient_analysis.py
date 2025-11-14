#!/usr/bin/env python3
"""Functions for computing gradient metrics from training history"""

from itertools import combinations


def compute_gradient_metrics(training_history):
    """
    Compute gradient metrics from training history data.
    This should be called post-hoc in analyze_results.py
    
    Args:
        training_history: Dictionary containing 'gradients' with epoch-by-epoch data
        
    Returns:
        Dictionary of gradient metrics
    """
    metrics = {}

    if 'gradients' not in training_history or len(training_history['gradients']['epoch']) == 0:
        # Return empty metrics if no gradient data available
        return metrics

    # Tunable parameters (margin/smoothing/windows)
    epsilon_rel = 0.05  # require A >= B*(1+epsilon) to be considered "above"
    smooth_window = 3   # simple moving average window for comparisons
    first_window_frac = 0.3  # use first 30% of epochs as "start" window
    last_window_frac = 0.3   # last 30% as "end" window
    majority_threshold = 0.7 # need at least 70% of window epochs to satisfy condition
    def smooth(vals, w):
        if w <= 1 or len(vals) <= 2:
            return vals
        out = []
        for i in range(len(vals)):
            s = max(0, i - (w - 1)//2)
            e = min(len(vals), i + (w - 1)//2 + 1)
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

    gradients = training_history['gradients']

    main_layers = [
        name
        for name in gradients
        if name != 'epoch' and 'from' not in name
    ]

    if len(main_layers) < 2:
        raise ValueError(
            "Expected at least two primary layers (layer*, without 'from') in gradients"
        )

    def layer_sort_key(name: str):
        digits = ''.join(ch for ch in name if ch.isdigit())
        return (int(digits) if digits else float('inf'), name)

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
        metrics[f'{layer}_large_drop'] = 1 if any(drop > drop_threshold for drop in layer_drops) else 0

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
        """Returns 1 if A was above B (with margin) before B was above A (with margin), else 0"""
        n = min(len(a_list), len(b_list))
        if n == 0:
            return 0

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
            return 1 if first_a_above_b < first_b_above_a else 0
        return 0

    # Combined patterns for all layer pairs: composite of component metrics (unchanged except margin-aware upstream)
    def check_pattern_composite(layerA_above_B, layerB_above_A, switches_AB, layerA_large_drop, temporal_order_AB):
        """Pattern occurs if all component metrics are present AND the temporal order is correct (A above B happened before B above A)"""
        return 1 if (layerA_above_B and layerB_above_A and switches_AB and layerA_large_drop and temporal_order_AB) else 0

    # Strict temporal pattern: checks specific epoch-1, epoch, epoch+1 sequence
    # Uses smoothed gradients and margin-aware comparisons
    def check_pattern_strict(layerA_smoothed, layerB_smoothed, layerA_drops):
        """
        Strict pattern: A above B at epoch-1, B above A at epoch, AND A has large drop at epoch-1.
        This is a very specific temporal sequence that must occur consecutively.
        """
        pattern = 0
        # Need at least 2 epochs to check epoch-1 and epoch
        if len(layerA_smoothed) < 2 or len(layerA_drops) < 1:
            return 0
        
        # Check for the specific sequence: epoch-1 (A > B), epoch (B > A), and large drop at epoch-1
        for i in range(1, len(layerA_smoothed)):
            # epoch-1: A above B (with margin)
            # epoch: B above A (with margin)
            # A has large drop at epoch-1
            if (gt_margin(layerA_smoothed[i-1], layerB_smoothed[i-1], epsilon_rel) and
                gt_margin(layerB_smoothed[i], layerA_smoothed[i], epsilon_rel) and
                i-1 < len(layerA_drops) and layerA_drops[i-1] > drop_threshold):
                pattern = 1
                break
        return pattern
    

    # Pairwise metrics for all unordered layer pairs
    for layer_a, layer_b in combinations(main_layers, 2):
        sm_a = smoothed_grads[layer_a]
        sm_b = smoothed_grads[layer_b]
        n = min(len(sm_a), len(sm_b))
        if n < 2:
            continue

        # Directional comparisons
        metrics[f'{layer_a}_above_{layer_b}'] = 1 if any(gt_margin(sm_a[i], sm_b[i], epsilon_rel) for i in range(n)) else 0
        metrics[f'{layer_b}_above_{layer_a}'] = 1 if any(gt_margin(sm_b[i], sm_a[i], epsilon_rel) for i in range(n)) else 0

        # Switches
        switches_ab = switches(sm_a, sm_b)
        metrics[f'switches_{layer_a}_{layer_b}'] = 1 if switches_ab > 0 else 0

        # Temporal order (both directions)
        metrics[f'{layer_a}vs{layer_b}_temporal_order'] = temporal_order(sm_a, sm_b)
        metrics[f'{layer_b}vs{layer_a}_temporal_order'] = temporal_order(sm_b, sm_a)

        # Composite patterns for both directions
        metrics[f'{layer_a}vs{layer_b}_pattern'] = check_pattern_composite(
            metrics[f'{layer_a}_above_{layer_b}'],
            metrics[f'{layer_b}_above_{layer_a}'],
            metrics[f'switches_{layer_a}_{layer_b}'],
            metrics[f'{layer_a}_large_drop'],
            metrics[f'{layer_a}vs{layer_b}_temporal_order'],
        )
        metrics[f'{layer_b}vs{layer_a}_pattern'] = check_pattern_composite(
            metrics[f'{layer_b}_above_{layer_a}'],
            metrics[f'{layer_a}_above_{layer_b}'],
            metrics[f'switches_{layer_a}_{layer_b}'],
            metrics[f'{layer_b}_large_drop'],
            metrics[f'{layer_b}vs{layer_a}_temporal_order'],
        )

        # Strict patterns
        metrics[f'{layer_a}vs{layer_b}_strict_pattern'] = check_pattern_strict(sm_a, sm_b, drops[layer_a])
        metrics[f'{layer_b}vs{layer_a}_strict_pattern'] = check_pattern_strict(sm_b, sm_a, drops[layer_b])

        # Strict patterns must be subset of composite patterns
        if metrics[f'{layer_a}vs{layer_b}_pattern'] == 0:
            metrics[f'{layer_a}vs{layer_b}_strict_pattern'] = 0
        if metrics[f'{layer_b}vs{layer_a}_pattern'] == 0:
            metrics[f'{layer_b}vs{layer_a}_strict_pattern'] = 0

    # Extract final accuracies
    if 'accuracy' in training_history and training_history['accuracy']:
        metrics['final_train_accuracy'] = training_history['accuracy'][-1]
    if 'test_accuracy' in training_history and training_history['test_accuracy']:
        metrics['final_test_accuracy'] = training_history['test_accuracy'][-1]
    
    return metrics
