#!/usr/bin/env python3
"""Functions for computing gradient metrics from training history"""


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
    
    layer1_grads = training_history['gradients']['layer1']
    layer2_grads = training_history['gradients']['layer2']
    
    # Enforce presence of aggregated layer3 gradients
    if ('layer3' not in training_history['gradients'] or
        not training_history['gradients']['layer3']):
        raise ValueError("Missing 'layer3' gradients in training history; analysis requires gradients['layer3']")
    layer3_grads = training_history['gradients']['layer3']

    # Smooth copies for comparisons
    l1 = smooth(layer1_grads, smooth_window)
    l2 = smooth(layer2_grads, smooth_window)
    l3 = smooth(layer3_grads, smooth_window)

    n = min(len(l1), len(l2), len(l3))
    if n < 2:
        return metrics

    # Pairwise comparisons (margin-aware)
    metrics['layer1_above_layer2'] = 1 if any(gt_margin(a, b, epsilon_rel) for a, b in zip(l1[:n], l2[:n])) else 0
    metrics['layer2_above_layer1'] = 1 if any(gt_margin(b, a, epsilon_rel) for a, b in zip(l1[:n], l2[:n])) else 0
    metrics['layer1_above_layer3'] = 1 if any(gt_margin(a, c, epsilon_rel) for a, c in zip(l1[:n], l3[:n])) else 0
    metrics['layer3_above_layer1'] = 1 if any(gt_margin(c, a, epsilon_rel) for a, c in zip(l1[:n], l3[:n])) else 0
    metrics['layer2_above_layer3'] = 1 if any(gt_margin(b, c, epsilon_rel) for b, c in zip(l2[:n], l3[:n])) else 0
    metrics['layer3_above_layer2'] = 1 if any(gt_margin(c, b, epsilon_rel) for b, c in zip(l2[:n], l3[:n])) else 0
    
    # Switches between pairs (margin-aware)
    def switches(a_list, b_list):
        cnt = 0
        prev = gt_margin(a_list[0], b_list[0], epsilon_rel)
        for i in range(1, n):
            cur = gt_margin(a_list[i], b_list[i], epsilon_rel)
            if cur != prev:
                cnt += 1
            prev = cur
        return cnt
    
    switches_12 = switches(l1, l2)
    switches_13 = switches(l1, l3)
    switches_23 = switches(l2, l3)
    
    metrics['switches_12'] = 1 if switches_12 > 0 else 0
    metrics['switches_13'] = 1 if switches_13 > 0 else 0
    metrics['switches_23'] = 1 if switches_23 > 0 else 0
    
    # Large relative drops for all layers (unchanged)
    drop_threshold = 0.5  # 50% drop
    l1_drops = [(layer1_grads[i-1] - layer1_grads[i]) / layer1_grads[i-1] 
                for i in range(1, len(layer1_grads)) if layer1_grads[i-1] > 0]
    l2_drops = [(layer2_grads[i-1] - layer2_grads[i]) / layer2_grads[i-1] 
                for i in range(1, len(layer2_grads)) if layer2_grads[i-1] > 0]
    l3_drops = [(layer3_grads[i-1] - layer3_grads[i]) / layer3_grads[i-1] 
                for i in range(1, len(layer3_grads)) if layer3_grads[i-1] > 0]
    
    metrics['layer1_large_drop'] = 1 if any(drop > drop_threshold for drop in l1_drops) else 0
    metrics['layer2_large_drop'] = 1 if any(drop > drop_threshold for drop in l2_drops) else 0
    metrics['layer3_large_drop'] = 1 if any(drop > drop_threshold for drop in l3_drops) else 0
    
    # Temporal order metric: Check if A above B happened before B above A
    def temporal_order(a_list, b_list):
        """Returns 1 if A was above B (with margin) before B was above A (with margin), else 0"""
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
    
    # Compute temporal_order for all layer combinations
    # This checks if A above B happened before B above A for each pair
    metrics['layer1vslayer2_temporal_order'] = temporal_order(l1, l2)  # layer1 above layer2 happened before layer2 above layer1
    metrics['layer1vslayer3_temporal_order'] = temporal_order(l1, l3)  # layer1 above layer3 happened before layer3 above layer1
    metrics['layer2vslayer3_temporal_order'] = temporal_order(l2, l3)  # layer2 above layer3 happened before layer3 above layer2
    metrics['layer2vslayer1_temporal_order'] = temporal_order(l2, l1)  # layer2 above layer1 happened before layer1 above layer2
    metrics['layer3vslayer1_temporal_order'] = temporal_order(l3, l1)  # layer3 above layer1 happened before layer1 above layer3
    metrics['layer3vslayer2_temporal_order'] = temporal_order(l3, l2)  # layer3 above layer2 happened before layer2 above layer3

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
    
    # Compute composite patterns
    
    metrics['layer1vslayer2_pattern'] = check_pattern_composite(
        metrics['layer1_above_layer2'], 
        metrics['layer2_above_layer1'],
        metrics['switches_12'], 
        metrics['layer1_large_drop'],
        metrics['layer1vslayer2_temporal_order'])  # layer1 above layer2 happened before layer2 above layer1
    
    metrics['layer1vslayer3_pattern'] = check_pattern_composite(
        metrics['layer1_above_layer3'], 
        metrics['layer3_above_layer1'],
        metrics['switches_13'], 
        metrics['layer1_large_drop'],
        metrics['layer1vslayer3_temporal_order'])  # layer1 above layer3 happened before layer3 above layer1
    
    metrics['layer2vslayer3_pattern'] = check_pattern_composite(
        metrics['layer2_above_layer3'], 
        metrics['layer3_above_layer2'],
        metrics['switches_23'], 
        metrics['layer2_large_drop'],
        metrics['layer2vslayer3_temporal_order'])  # layer2 above layer3 happened before layer3 above layer2
    
    metrics['layer2vslayer1_pattern'] = check_pattern_composite(
        metrics['layer2_above_layer1'], 
        metrics['layer1_above_layer2'],
        metrics['switches_12'], 
        metrics['layer2_large_drop'],
        metrics['layer2vslayer1_temporal_order'])  # layer2 above layer1 happened before layer1 above layer2
    
    metrics['layer3vslayer1_pattern'] = check_pattern_composite(
        metrics['layer3_above_layer1'], 
        metrics['layer1_above_layer3'],
        metrics['switches_13'], 
        metrics['layer3_large_drop'],
        metrics['layer3vslayer1_temporal_order'])  # layer3 above layer1 happened before layer1 above layer3
    
    metrics['layer3vslayer2_pattern'] = check_pattern_composite(
        metrics['layer3_above_layer2'], 
        metrics['layer2_above_layer3'],
        metrics['switches_23'], 
        metrics['layer3_large_drop'],
        metrics['layer3vslayer2_temporal_order'])  # layer3 above layer2 happened before layer2 above layer3
    
    # Compute strict temporal patterns
    # Note: Strict patterns use smoothed gradients and margin-aware comparisons
    # They check for a specific consecutive sequence: epoch-1 (A > B), epoch (B > A), and A drop at epoch-1
    metrics['layer1vslayer2_strict_pattern'] = check_pattern_strict(l1, l2, l1_drops)
    metrics['layer1vslayer3_strict_pattern'] = check_pattern_strict(l1, l3, l1_drops)
    metrics['layer2vslayer3_strict_pattern'] = check_pattern_strict(l2, l3, l2_drops)
    metrics['layer2vslayer1_strict_pattern'] = check_pattern_strict(l2, l1, l2_drops)
    metrics['layer3vslayer1_strict_pattern'] = check_pattern_strict(l3, l1, l3_drops)
    metrics['layer3vslayer2_strict_pattern'] = check_pattern_strict(l3, l2, l3_drops)
    
    # Ensure strict patterns are only 1 if their corresponding composite pattern is also 1
    # Strict patterns should be a subset of composite patterns
    if metrics['layer1vslayer2_pattern'] == 0:
        metrics['layer1vslayer2_strict_pattern'] = 0
    if metrics['layer1vslayer3_pattern'] == 0:
        metrics['layer1vslayer3_strict_pattern'] = 0
    if metrics['layer2vslayer3_pattern'] == 0:
        metrics['layer2vslayer3_strict_pattern'] = 0
    if metrics['layer2vslayer1_pattern'] == 0:
        metrics['layer2vslayer1_strict_pattern'] = 0
    if metrics['layer3vslayer1_pattern'] == 0:
        metrics['layer3vslayer1_strict_pattern'] = 0
    if metrics['layer3vslayer2_pattern'] == 0:
        metrics['layer3vslayer2_strict_pattern'] = 0
    
    # Extract final accuracies
    if 'accuracy' in training_history and training_history['accuracy']:
        metrics['final_train_accuracy'] = training_history['accuracy'][-1]
    if 'test_accuracy' in training_history and training_history['test_accuracy']:
        metrics['final_test_accuracy'] = training_history['test_accuracy'][-1]
    
    return metrics
