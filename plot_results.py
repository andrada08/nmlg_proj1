#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
import re
from pathlib import Path

LAYER_COLOR_MAP = {
    'layer1': '#FFC300',  # yellow/gold
    'layer2': '#FF69B4',  # pink
    'layer3': '#6EC5FF',  # lighter blue
    'layer4': '#2ca02c',  # green
    'layer5': '#9467bd',
    'layer6': '#8c564b',
}
DEFAULT_LAYER_COLORS = list(LAYER_COLOR_MAP.values())


def get_layer_color(layer_name: str) -> str:
    if layer_name in LAYER_COLOR_MAP:
        return LAYER_COLOR_MAP[layer_name]
    # Fallback: assign deterministically based on numeric suffix or hash
    match = re.search(r'layer(\d+)', layer_name)
    if match:
        idx = int(match.group(1)) - 1
        return DEFAULT_LAYER_COLORS[idx % len(DEFAULT_LAYER_COLORS)]
    return DEFAULT_LAYER_COLORS[hash(layer_name) % len(DEFAULT_LAYER_COLORS)]

def smooth_gradients(vals, window=3):
    """Smooth gradient values using a moving average"""
    if window <= 1 or not vals:
        return vals
    out = []
    n = len(vals)
    half = (window - 1) // 2
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        window_vals = vals[s:e]
        out.append(sum(window_vals) / max(1, len(window_vals)))
    return out

def load_results(input_folder="."):
    """Load the gradient analysis results"""
    csv_path = os.path.join(input_folder, 'gradient_analysis.csv')
    if not Path(csv_path).exists():
        print(f"{csv_path} not found. Run analyze_results.py first.")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")
    return df


def _extract_layer_number(name: str) -> int:
    match = re.search(r'layer(\d+)', name)
    return int(match.group(1)) if match else 0


def _get_primary_layers_from_history(history: dict) -> list[str]:
    gradients = history.get('gradients', {})
    primary = [
        name for name in gradients
        if name != 'epoch' and 'from' not in name
    ]
    return sorted(primary, key=_extract_layer_number)


def _build_dynamic_pattern_groups(df: pd.DataFrame) -> dict:
    pattern_groups: dict[str, dict] = {}
    for col in df.columns:
        if not col.endswith('_pattern') or col.endswith('_strict_pattern'):
            continue
        base = col[:-len('_pattern')]
        if 'vs' not in base:
            continue
        layer_a, layer_b = base.split('vs', 1)
        related_metrics = [
            col,
            f'{base}_strict_pattern',
            f'{layer_a}_above_{layer_b}',
            f'{layer_b}_above_{layer_a}',
            f'switches_{layer_a}_{layer_b}',
            f'{layer_a}_large_drop',
            f'{layer_b}_large_drop',
        ]
        pattern_groups[col] = {
            'related_metrics': related_metrics,
            'title': f'{layer_a.title()} vs {layer_b.title()} Pattern'.replace('Layer', 'Layer ')
        }
    return pattern_groups

def create_pattern_specific_plots(df, output_folder=".", title_suffix=""):
    """Create plots for each pattern showing its related metrics"""

    pattern_groups = _build_dynamic_pattern_groups(df)

    for pattern, info in pattern_groups.items():
        if pattern in df.columns:
            # Check if any runs have this pattern
            pattern_runs = df[df[pattern] == 1]
            if len(pattern_runs) > 0:
                print(f"Creating pattern-specific plot for: {pattern} ({len(pattern_runs)} runs)")
                
                # Filter to only existing metrics
                available_metrics = [m for m in info['related_metrics'] if m in df.columns]
                
                if len(available_metrics) > 0:
                    # Create figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    fig.suptitle(f'{info["title"]} Analysis - {title_suffix}', fontsize=14, fontweight='bold')
                    
                    # Left subplot: Metric frequency for this pattern
                    pattern_metric_totals = df[available_metrics].sum().sort_values(ascending=False)
                    bars = ax1.bar(range(len(pattern_metric_totals)), pattern_metric_totals.values, 
                                  color='lightcoral', edgecolor='black')
                    ax1.set_xlabel('Related Metrics')
                    ax1.set_ylabel('Total Occurrences')
                    ax1.set_title(f'Related Metrics Frequency\n({len(pattern_runs)} runs with {pattern})')
                    ax1.set_xticks(range(len(pattern_metric_totals)))
                    ax1.set_xticklabels(pattern_metric_totals.index, rotation=45, ha='right')
                    ax1.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for i, (bar, count) in enumerate(zip(bars, pattern_metric_totals)):
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                                 f'{count}', ha='center', va='bottom', fontsize=9)
                    
                    # Right subplot: Correlation matrix for related metrics
                    if len(available_metrics) > 1:
                        pattern_data = df[available_metrics]
                        correlation_matrix = pattern_data.corr()
                        
                        im = ax2.imshow(correlation_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                        ax2.set_xticks(range(len(correlation_matrix.columns)))
                        ax2.set_yticks(range(len(correlation_matrix.index)))
                        ax2.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
                        ax2.set_yticklabels(correlation_matrix.index)
                        ax2.set_title('Related Metrics Correlation')
                        
                        # Add correlation values as text
                        for i in range(len(correlation_matrix.index)):
                            for j in range(len(correlation_matrix.columns)):
                                text = ax2.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                               ha="center", va="center", color="black", fontsize=8)
                        
                        # Add colorbar
                        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
                        cbar.set_label('Correlation Coefficient')
                    else:
                        ax2.text(0.5, 0.5, 'Need at least 2 metrics\nfor correlation matrix', 
                                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                        ax2.set_title('Related Metrics Correlation')
                    
                    plt.tight_layout()
                    
                    filename = os.path.join(output_folder, f'pattern_analysis_{pattern}_{title_suffix}.png')
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Pattern analysis plot saved to: {filename}")
                else:
                    print(f"No related metrics found for pattern: {pattern}")

def create_comprehensive_metrics_plots(df, output_folder=".", title_suffix=""):
    """Create comprehensive metric visualizations with frequency and correlation"""
    
    # Select metric columns
    metric_cols = [col for col in df.columns if col not in [
        'run_name', 'n_epochs', 'batch_size', 'optimizer', 'activation',
        'architecture', 'input_size', 'hidden_sizes', 'output_size', 'ln_rate', 'layer_lns',
        'final_test_accuracy', 'final_train_accuracy', 'total_score'
    ]]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Comprehensive Metrics Analysis - {title_suffix}', fontsize=16, fontweight='bold')
    
    # Left subplot: Metric frequency bar chart
    # Sort metrics so layer 1 related metrics are at the end
    def sort_metrics(metric_list):
        """Sort metrics by the layer numbers they reference, then alphabetically."""
        def metric_key(name: str):
            layers = re.findall(r'layer(\d+)', name)
            layer_nums = tuple(int(num) for num in layers) if layers else (float('inf'),)
            return (layer_nums, name)
        return sorted(metric_list, key=metric_key)
    
    sorted_metrics = sort_metrics(metric_cols)
    metric_totals = df[sorted_metrics].sum().sort_values(ascending=False)
    bars = ax1.bar(range(len(metric_totals)), metric_totals.values, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Total Occurrences')
    ax1.set_title('Metric Frequency Across All Runs')
    ax1.set_xticks(range(len(metric_totals)))
    ax1.set_xticklabels(metric_totals.index, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, metric_totals)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{count}', ha='center', va='bottom', fontsize=9)
    
    # Right subplot: Metric correlation heatmap
    if len(metric_cols) > 1:  # Need at least 2 metrics for correlation
        # Use the same sorted metrics for consistency
        metric_data = df[sorted_metrics]
        correlation_matrix = metric_data.corr()
        
        im = ax2.imshow(correlation_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(correlation_matrix.columns)))
        ax2.set_yticks(range(len(correlation_matrix.index)))
        ax2.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax2.set_yticklabels(correlation_matrix.index)
        ax2.set_title('Gradient Metrics Correlation Matrix')
        
        # Add correlation values as text
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                text = ax2.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
        cbar.set_label('Correlation Coefficient')
    else:
        ax2.text(0.5, 0.5, 'Need at least 2 metrics\nfor correlation matrix', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Gradient Metrics Correlation Matrix')
    
    plt.tight_layout()
    
    filename = os.path.join(output_folder, f'comprehensive_metrics_{title_suffix}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive metrics plot saved to: {filename}")

def create_pattern_hyperparameter_heatmaps(df, output_folder=".", title_suffix=""):
    """Create heatmaps showing pattern frequency by hyperparameter combinations"""
    
    # Get pattern columns (excluding strict patterns for cleaner visualization)
    pattern_columns = [col for col in df.columns if col.endswith('_pattern') and not col.endswith('_strict_pattern')]
    
    # Check required columns
    required_cols = ['hidden_sizes', 'layer_lns']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns {required_cols}. Skipping hyperparameter heatmaps.")
        return
    
    import ast
    import re
    
    for pattern in pattern_columns:
        # Only create plots for patterns that actually occur
        if pattern not in df.columns or df[pattern].sum() == 0:
            continue
        
        print(f"Creating hyperparameter heatmap for: {pattern}")
        
        # ===== LEFT PANEL: Original heatmap by full combinations =====
        pattern_counts = df.groupby(['hidden_sizes', 'layer_lns'])[pattern].sum().reset_index()
        pivot_table = pattern_counts.pivot(index='hidden_sizes', columns='layer_lns', values=pattern)
        pivot_table = pivot_table.fillna(0).astype(int)
        
        # ===== MIDDLE & RIGHT PANELS: Cumulative by individual layer sizes and learning rates =====
        # Extract individual layer sizes and learning rates for BOTH pattern and no-pattern
        cumulative_data_pattern = []
        cumulative_data_no_pattern = []
        
        for idx, row in df.iterrows():
            # Parse hidden_sizes (format: "[100, 150, 200]" or similar)
            try:
                hidden_sizes_str = str(row['hidden_sizes']).strip()
                if hidden_sizes_str.startswith('['):
                    sizes = ast.literal_eval(hidden_sizes_str)
                else:
                    # Fallback: try to extract numbers
                    sizes = [int(x) for x in re.findall(r'\d+', hidden_sizes_str)]
            except:
                continue
            
            # Parse layer_lns (format: "{'layer1': 1e-4, 'layer2': 2e-4, 'layer3': 4e-4}" or similar)
            try:
                layer_lns_str = str(row['layer_lns']).strip()
                if layer_lns_str.startswith('{'):
                    lrs = ast.literal_eval(layer_lns_str)
                else:
                    # Fallback: try to extract as dict
                    lrs = {}
                    layer_keys = [f'layer{i+1}' for i in range(len(sizes))]
                    for layer in layer_keys:
                        match = re.search(rf"'{layer}':\s*([\d.e+-]+)", layer_lns_str)
                        if match:
                            lrs[layer] = float(match.group(1))
            except:
                continue
            
            # Add entries for each layer (both pattern and no-pattern)
            layer_sequence = [f'layer{i+1}' for i in range(len(sizes))]
            for layer_idx, (layer_name, size) in enumerate(zip(layer_sequence, sizes), 1):
                if layer_name in lrs:
                    if row[pattern] == 1:
                        cumulative_data_pattern.append({
                            'layer': layer_idx,
                            'size': size,
                            'lr': lrs[layer_name],
                            'count': 1
                        })
                    else:
                        cumulative_data_no_pattern.append({
                            'layer': layer_idx,
                            'size': size,
                            'lr': lrs[layer_name],
                            'count': 1
                        })
        
        if not cumulative_data_pattern and not cumulative_data_no_pattern:
            print(f"  Warning: Could not parse hyperparameters for cumulative view. Skipping additional panels.")
            # Create single panel figure
            fig, ax = plt.subplots(1, 1, figsize=(max(8, len(pivot_table.columns) * 0.8), 
                                                  max(6, len(pivot_table.index) * 0.6)))
            axes = [ax]
            has_cumulative = False
        else:
            has_cumulative = True
            # Create cumulative pivot tables for pattern
            if cumulative_data_pattern:
                cum_df_pattern = pd.DataFrame(cumulative_data_pattern)
                cum_pivot_pattern = cum_df_pattern.groupby(['layer', 'size', 'lr'])['count'].sum().reset_index()
            else:
                cum_pivot_pattern = pd.DataFrame(columns=['layer', 'size', 'lr', 'count'])
            
            # Create cumulative pivot tables for no-pattern
            if cumulative_data_no_pattern:
                cum_df_no_pattern = pd.DataFrame(cumulative_data_no_pattern)
                cum_pivot_no_pattern = cum_df_no_pattern.groupby(['layer', 'size', 'lr'])['count'].sum().reset_index()
            else:
                cum_pivot_no_pattern = pd.DataFrame(columns=['layer', 'size', 'lr', 'count'])
            
            # Get all unique combinations for consistent structure
            pattern_layers = cum_pivot_pattern['layer'].tolist() if len(cum_pivot_pattern) > 0 else []
            no_pattern_layers = cum_pivot_no_pattern['layer'].tolist() if len(cum_pivot_no_pattern) > 0 else []
            all_layers = sorted(set(pattern_layers + no_pattern_layers))
            
            pattern_sizes = cum_pivot_pattern['size'].tolist() if len(cum_pivot_pattern) > 0 else []
            no_pattern_sizes = cum_pivot_no_pattern['size'].tolist() if len(cum_pivot_no_pattern) > 0 else []
            all_sizes = sorted(set(pattern_sizes + no_pattern_sizes))
            
            pattern_lrs = cum_pivot_pattern['lr'].tolist() if len(cum_pivot_pattern) > 0 else []
            no_pattern_lrs = cum_pivot_no_pattern['lr'].tolist() if len(cum_pivot_no_pattern) > 0 else []
            all_lrs = sorted(set(pattern_lrs + no_pattern_lrs))
            
            # Build pivot tables with proper structure
            def build_pivot_table(df_pivot, all_layers, all_sizes, all_lrs):
                # Create index: layer_size combinations, sorted by layer then size
                index_labels = []
                for layer in all_layers:
                    # Get sizes that actually exist for this layer
                    layer_sizes_in_data = sorted(df_pivot[df_pivot['layer'] == layer]['size'].unique().tolist())
                    # Use all sizes if we have data, otherwise use all_sizes
                    layer_sizes = layer_sizes_in_data if len(layer_sizes_in_data) > 0 else all_sizes
                    for size in layer_sizes:
                        index_labels.append((layer, size))
                
                # Create columns: layer_lr combinations, sorted by layer then lr
                col_labels = []
                for layer in all_layers:
                    # Get lrs that actually exist for this layer
                    layer_lrs_in_data = sorted(df_pivot[df_pivot['layer'] == layer]['lr'].unique().tolist())
                    # Use all lrs if we have data, otherwise use all_lrs
                    layer_lrs = layer_lrs_in_data if len(layer_lrs_in_data) > 0 else all_lrs
                    for lr in layer_lrs:
                        col_labels.append((layer, lr))
                
                # Build pivot table
                pivot_data = {}
                for (row_layer, row_size) in index_labels:
                    pivot_data[(row_layer, row_size)] = {}
                    for (col_layer, col_lr) in col_labels:
                        # Only count if same layer
                        if row_layer == col_layer:
                            count = df_pivot[(df_pivot['layer'] == row_layer) & 
                                           (df_pivot['size'] == row_size) & 
                                           (df_pivot['lr'] == col_lr)]['count'].sum()
                            pivot_data[(row_layer, row_size)][(col_layer, col_lr)] = int(count)
                        else:
                            pivot_data[(row_layer, row_size)][(col_layer, col_lr)] = 0
                
                # Convert to DataFrame
                pivot_df = pd.DataFrame(pivot_data).T
                pivot_df.columns = col_labels
                return pivot_df, index_labels, col_labels
            
            # Build pattern pivot table first to get the structure
            cum_pivot_table_pattern, row_labels, col_labels = build_pivot_table(cum_pivot_pattern, all_layers, all_sizes, all_lrs)
            
            # Build no-pattern pivot table using the SAME structure (row_labels and col_labels)
            # This ensures both tables have identical structure for comparison
            pivot_data_no_pattern = {}
            for (row_layer, row_size) in row_labels:
                pivot_data_no_pattern[(row_layer, row_size)] = {}
                for (col_layer, col_lr) in col_labels:
                    # Only count if same layer
                    if row_layer == col_layer:
                        if len(cum_pivot_no_pattern) > 0:
                            count = cum_pivot_no_pattern[(cum_pivot_no_pattern['layer'] == row_layer) & 
                                                       (cum_pivot_no_pattern['size'] == row_size) & 
                                                       (cum_pivot_no_pattern['lr'] == col_lr)]['count'].sum()
                            pivot_data_no_pattern[(row_layer, row_size)][(col_layer, col_lr)] = int(count)
                        else:
                            pivot_data_no_pattern[(row_layer, row_size)][(col_layer, col_lr)] = 0
                    else:
                        pivot_data_no_pattern[(row_layer, row_size)][(col_layer, col_lr)] = 0
            
            # Convert to DataFrame with same structure
            cum_pivot_table_no_pattern = pd.DataFrame(pivot_data_no_pattern).T
            cum_pivot_table_no_pattern.columns = col_labels
            
            # Create three-panel figure
            fig = plt.figure(figsize=(24, max(8, len(row_labels) * 0.3)))
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            axes = [ax1, ax2, ax3]
        
        # ===== LEFT PANEL: Full combinations =====
        ax1 = axes[0]
        im1 = ax1.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto', vmin=0)
        ax1.set_xticks(range(len(pivot_table.columns)))
        ax1.set_yticks(range(len(pivot_table.index)))
        ax1.set_xticklabels([str(x)[:30] + '...' if len(str(x)) > 30 else str(x) 
                             for x in pivot_table.columns], rotation=45, ha='right', fontsize=8)
        ax1.set_yticklabels([str(x)[:30] + '...' if len(str(x)) > 30 else str(x) 
                             for x in pivot_table.index], fontsize=8)
        ax1.set_xlabel('Learning Rates (layer_lns)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Layer Sizes (hidden_sizes)', fontsize=11, fontweight='bold')
        ax1.set_title(f'By Full Combinations\n(Count of runs with pattern)', fontsize=12, fontweight='bold')
        
        max_count1 = pivot_table.values.max()
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                count = int(pivot_table.iloc[i, j])
                text_color = 'white' if max_count1 > 0 and count > max_count1 * 0.6 else 'black'
                ax1.text(j, i, str(count), ha='center', va='center', 
                        color=text_color, fontsize=9, fontweight='bold')
        
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Number of Runs', rotation=270, labelpad=15, fontsize=9)
        
        # ===== MIDDLE PANEL: Cumulative by individual components (WITH PATTERN) =====
        if has_cumulative:
            ax2 = axes[1]
            im2 = ax2.imshow(cum_pivot_table_pattern.values, cmap='YlOrRd', aspect='auto', vmin=0)
            
            # Create grouped tick labels with separators
            def create_grouped_labels(labels, label_type='size'):
                """Create labels grouped by layer with separators"""
                # Group labels by layer
                layer_groups = {}
                for i, (layer, value) in enumerate(labels):
                    if layer not in layer_groups:
                        layer_groups[layer] = []
                    layer_groups[layer].append((i, value))
                
                # Create tick labels (just numbers)
                tick_labels = []
                tick_positions = []
                group_boundaries = []
                
                for layer in sorted(layer_groups.keys()):
                    group_boundaries.append(len(tick_positions))
                    for pos, value in layer_groups[layer]:
                        tick_positions.append(pos)
                        if label_type == 'size':
                            tick_labels.append(str(int(value)))
                        else:  # lr
                            tick_labels.append(f'{value:.2e}')
                    group_boundaries.append(len(tick_positions))
                
                return tick_labels, tick_positions, layer_groups, group_boundaries
            
            row_tick_labels, row_tick_pos, row_layer_groups, row_boundaries = create_grouped_labels(row_labels, 'size')
            col_tick_labels, col_tick_pos, col_layer_groups, col_boundaries = create_grouped_labels(col_labels, 'lr')
            
            ax2.set_xticks(col_tick_pos)
            ax2.set_yticks(row_tick_pos)
            ax2.set_xticklabels(col_tick_labels, rotation=45, ha='right', fontsize=7)
            ax2.set_yticklabels(row_tick_labels, fontsize=7)
            
            # Add layer group labels on axes
            # Y-axis: Layer names on the left (move further out to avoid tick overlap)
            for layer in sorted(row_layer_groups.keys()):
                layer_positions = [pos for pos, _ in row_layer_groups[layer]]
                if layer_positions:
                    mid_idx = (min(layer_positions) + max(layer_positions)) / 2
                    # Move further left (x=-0.08) to avoid overlapping with tick labels
                    ax2.text(-0.08, mid_idx, f'L{layer}', 
                            transform=ax2.get_yaxis_transform(), ha='right', va='center',
                            fontsize=9, fontweight='bold', rotation=0)
            
            # X-axis: Layer names below ticks but above x-axis label
            for layer in sorted(col_layer_groups.keys()):
                layer_positions = [pos for pos, _ in col_layer_groups[layer]]
                if layer_positions:
                    mid_idx = (min(layer_positions) + max(layer_positions)) / 2
                    # Position below ticks (y=-0.09) but above x-axis label
                    ax2.text(mid_idx, -0.09, f'L{layer}', 
                            transform=ax2.get_xaxis_transform(), ha='center', va='top',
                            fontsize=9, fontweight='bold')
            
            # Add vertical separators between layer groups (columns)
            for i, boundary in enumerate(col_boundaries[1:-1], 1):
                ax2.axvline(boundary - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            # Add horizontal separators between layer groups (rows)
            for i, boundary in enumerate(row_boundaries[1:-1], 1):
                ax2.axhline(boundary - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            ax2.set_xlabel('Learning Rate', fontsize=11, fontweight='bold', labelpad=25)
            ax2.set_ylabel('Layer Size', fontsize=11, fontweight='bold', labelpad=25)
            ax2.set_title(f'With Pattern\n(Sum across all layers)', fontsize=12, fontweight='bold')
            
            max_count2 = cum_pivot_table_pattern.values.max()
            for i in range(len(cum_pivot_table_pattern.index)):
                for j in range(len(cum_pivot_table_pattern.columns)):
                    count = int(cum_pivot_table_pattern.iloc[i, j])
                    if count > 0:
                        text_color = 'white' if max_count2 > 0 and count > max_count2 * 0.6 else 'black'
                        ax2.text(j, i, str(count), ha='center', va='center', 
                                color=text_color, fontsize=8, fontweight='bold')
            
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Number of Runs', rotation=270, labelpad=15, fontsize=9)
            
            # ===== RIGHT PANEL: Cumulative by individual components (NO PATTERN) =====
            ax3 = axes[2]
            im3 = ax3.imshow(cum_pivot_table_no_pattern.values, cmap='YlOrRd', aspect='auto', vmin=0)
            
            ax3.set_xticks(col_tick_pos)
            ax3.set_yticks(row_tick_pos)
            ax3.set_xticklabels(col_tick_labels, rotation=45, ha='right', fontsize=7)
            ax3.set_yticklabels(row_tick_labels, fontsize=7)
            
            # Add layer group labels on axes (same as ax2, with same spacing)
            for layer in sorted(row_layer_groups.keys()):
                layer_positions = [pos for pos, _ in row_layer_groups[layer]]
                if layer_positions:
                    mid_idx = (min(layer_positions) + max(layer_positions)) / 2
                    # Move further left (x=-0.08) to avoid overlapping with tick labels
                    ax3.text(-0.08, mid_idx, f'L{layer}', 
                            transform=ax3.get_yaxis_transform(), ha='right', va='center',
                            fontsize=9, fontweight='bold', rotation=0)
            
            for layer in sorted(col_layer_groups.keys()):
                layer_positions = [pos for pos, _ in col_layer_groups[layer]]
                if layer_positions:
                    mid_idx = (min(layer_positions) + max(layer_positions)) / 2
                    # Position below ticks (y=-0.09) but above x-axis label
                    ax3.text(mid_idx, -0.09, f'L{layer}', 
                            transform=ax3.get_xaxis_transform(), ha='center', va='top',
                            fontsize=9, fontweight='bold')
            
            # Add separators
            for boundary in col_boundaries[1:-1]:
                ax3.axvline(boundary - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            for boundary in row_boundaries[1:-1]:
                ax3.axhline(boundary - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            ax3.set_xlabel('Learning Rate', fontsize=11, fontweight='bold', labelpad=25)
            ax3.set_ylabel('Layer Size', fontsize=11, fontweight='bold', labelpad=25)
            ax3.set_title(f'Without Pattern\n(Sum across all layers)', fontsize=12, fontweight='bold')
            
            max_count3 = cum_pivot_table_no_pattern.values.max()
            for i in range(len(cum_pivot_table_no_pattern.index)):
                for j in range(len(cum_pivot_table_no_pattern.columns)):
                    count = int(cum_pivot_table_no_pattern.iloc[i, j])
                    if count > 0:
                        text_color = 'white' if max_count3 > 0 and count > max_count3 * 0.6 else 'black'
                        ax3.text(j, i, str(count), ha='center', va='center', 
                                color=text_color, fontsize=8, fontweight='bold')
            
            cbar3 = plt.colorbar(im3, ax=ax3)
            cbar3.set_label('Number of Runs', rotation=270, labelpad=15, fontsize=9)
        
        # Overall title
        pattern_name = pattern.replace('_pattern', '').replace('_', ' ')
        fig.suptitle(f'Pattern Frequency: {pattern_name}', fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save plot
        pattern_name_file = pattern.replace('_pattern', '').replace('_', '')
        filename = os.path.join(output_folder, f'pattern_hyperparameter_{pattern_name_file}_{title_suffix}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Pattern hyperparameter heatmap saved to: {filename}")

def main():
    # Parse command line arguments
    pattern_analysis = '--pattern-analysis' in sys.argv
    comprehensive_metrics = '--comprehensive-metrics' in sys.argv
    pattern_specific = '--pattern-specific' in sys.argv
    no_pattern_examples = '--no-pattern-examples' in sys.argv
    pattern_hyperparameter = '--pattern-hyperparameter' in sys.argv
    clean_plots = '--clean-plots' in sys.argv
    input_folder = "."
    subfolder = ""
    
    if '--input-folder' in sys.argv:
        idx = sys.argv.index('--input-folder')
        if idx + 1 < len(sys.argv):
            input_folder = sys.argv[idx + 1]
    
    # Require explicit input folder
    if not ('--input-folder' in sys.argv and os.path.isdir(input_folder)):
        print("Error: --input-folder is required and must be an existing directory (e.g., results/your_experiment)")
        return
    
    # Verify expected CSV exists
    expected_csv = os.path.join(input_folder, 'gradient_analysis.csv')
    if not os.path.isfile(expected_csv):
        print(f"Error: Expected CSV not found at {expected_csv}. Run analyze_results.py first for this folder.")
        return

    # Determine analysis name from input folder
    if input_folder.startswith('results/'):
        # Extract analysis name from results path (e.g., "results/test" -> "test")
        analysis_name = input_folder.split('/')[-1]
    else:
        analysis_name = "plots"  # Default fallback
    
    # Create results folder structure
    results_folder = "results"
    plots_folder = os.path.join(results_folder, analysis_name)
    os.makedirs(plots_folder, exist_ok=True)
    
    print(f"Saving plots to: {plots_folder}")

    # Clean up old plot images in this folder (do not remove CSV) if requested
    if clean_plots:
        try:
            import glob
            cleaned_count = 0
            for old_png in glob.glob(os.path.join(plots_folder, '*.png')):
                try:
                    os.remove(old_png)
                    cleaned_count += 1
                except OSError:
                    pass
            if cleaned_count > 0:
                print(f"Cleaned {cleaned_count} old plot image(s) from {plots_folder}")
        except Exception as e:
            print(f"Warning: could not clean plot images in {plots_folder}: {e}")
    
    # Backward compatibility: check for subfolder as first argument
    if '--subfolder' in sys.argv:
        idx = sys.argv.index('--subfolder')
        if idx + 1 < len(sys.argv):
            subfolder = sys.argv[idx + 1]
            if not input_folder or input_folder == ".":
                input_folder = f"outputs/{subfolder}"

    # Optional explicit override for output subfolder
    if '--output-subfolder' in sys.argv:
        idx = sys.argv.index('--output-subfolder')
        if idx + 1 < len(sys.argv):
            subfolder = sys.argv[idx + 1]

    # If no subfolder specified but we have an input_folder, try to infer it
    if not subfolder and input_folder and input_folder != ".":
        # If input_folder contains 'outputs/', extract the subfolder
        if 'outputs/' in input_folder:
            subfolder = input_folder.split('outputs/')[-1].strip('/').split('/')[0]
        # If input_folder is a results folder, map results/{name} -> outputs/{name}
        elif input_folder.startswith('results/'):
            subfolder = input_folder.split('/')[-1]
            print(f"Inferred subfolder from results folder: {subfolder}")
        else:
            # As a last resort, look for sweep directories in outputs/ to determine subfolder
            import glob
            sweep_dirs = glob.glob('outputs/*/sweep_*')
            if sweep_dirs:
                subfolder = sweep_dirs[0].split('/')[1]  # outputs/subfolder/sweep_*
                print(f"Inferred subfolder via scan: {subfolder}")
            else:
                subfolder = 'test'  # fallback
    
    # If no flags are specified, create all plots by default
    create_all = not (pattern_analysis or comprehensive_metrics or pattern_specific or no_pattern_examples or pattern_hyperparameter)
    if create_all:
        pattern_analysis = True
        comprehensive_metrics = True
        pattern_specific = True
        no_pattern_examples = True
        pattern_hyperparameter = True
        print("No plot flags specified. Creating all plots by default.")
    
    # Create pattern analysis plots
    print("Loading gradient analysis results...")
    df = load_results(input_folder)
    if df is None:
        return
    
    if pattern_analysis:
        # Get only the pattern columns (metrics that end with '_pattern')
        pattern_columns = [col for col in df.columns if col.endswith('_pattern')]
        
        print(f"Creating pattern analysis plots for {len(df)} runs...")
        create_pattern_plots(df, pattern_columns, plots_folder, input_folder, subfolder)
    
    if comprehensive_metrics:
        print(f"Creating comprehensive metrics plots for {len(df)} runs...")
        title_suffix = analysis_name
        create_comprehensive_metrics_plots(df, plots_folder, title_suffix)
    
    if pattern_specific:
        print(f"Creating pattern-specific plots for {len(df)} runs...")
        title_suffix = analysis_name
        create_pattern_specific_plots(df, plots_folder, title_suffix)
    
    if no_pattern_examples:
        print(f"Creating no-pattern examples for {len(df)} runs...")
        title_suffix = analysis_name
        create_no_pattern_examples(df, plots_folder, input_folder, subfolder, title_suffix=title_suffix, n_examples=6)
    
    if pattern_hyperparameter:
        print(f"Creating pattern hyperparameter heatmaps for {len(df)} runs...")
        title_suffix = analysis_name
        create_pattern_hyperparameter_heatmaps(df, plots_folder, title_suffix)
    
    print("\nAll plots complete!")

def create_pattern_plots(df, pattern_columns, output_folder=".", input_folder="", subfolder=""):
    """Create visualizations for pattern analysis"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    def make_frequency_plot(dataframe, suffix):
        if dataframe.empty:
            return
        pattern_counts = dataframe[pattern_columns].sum().sort_values(ascending=False)
        if pattern_counts.empty:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        pattern_percentages = (pattern_counts / len(dataframe)) * 100

        bars = ax.bar(range(len(pattern_counts)), pattern_percentages)
        ax.set_xlabel('Patterns')
        ax.set_ylabel('Percentage of Runs (%)')
        ax.set_title(f'Pattern Frequency Across All Runs - {suffix}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(pattern_counts)))
        ax.set_xticklabels(pattern_counts.index, rotation=45, ha='right')

        for i, (bar, count) in enumerate(zip(bars, pattern_counts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{count}/{len(dataframe)}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        filename = os.path.join(output_folder, f'pattern_frequency_{suffix}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Pattern frequency plot saved to: {filename}")

    base_suffix = input_folder.replace('outputs/', '').replace('/', '_') if input_folder else "current"
    make_frequency_plot(df, base_suffix)

    if 'architecture' in df.columns:
        for arch in sorted(df['architecture'].dropna().unique()):
            subset = df[df['architecture'] == arch]
            if subset.empty:
                continue
            make_frequency_plot(subset, f'{base_suffix}_{arch}')
    
    create_pattern_examples(df, pattern_columns, output_folder, input_folder, subfolder)

def create_pattern_examples(df, pattern_columns, output_folder=".", input_folder="", subfolder=""):
    """Create separate plots for each pattern showing gradient examples"""
    
    import os
    import json
    import matplotlib.pyplot as plt
    
    for pattern in pattern_columns:
        # Find runs that have this pattern
        pattern_mask = df[pattern] == 1
        # If a matching temporal_order exists, require it to be 1 to enforce directionality
        temporal_order_col = None
        if pattern.endswith('_pattern'):
            base = pattern.replace('_pattern', '')
            temporal_order_col = f'{base}_temporal_order'
        if temporal_order_col and temporal_order_col in df.columns:
            pattern_mask = pattern_mask & (df[temporal_order_col] == 1)
        pattern_runs = df[pattern_mask]['run_name'].tolist()
        
        if not pattern_runs:
            print(f"No runs found with pattern: {pattern}")
            continue
            
        print(f"Creating examples for pattern: {pattern} ({len(pattern_runs)} runs)")
        
        # Create subplot for this pattern
        n_examples = min(6, len(pattern_runs))  # Show up to 6 examples
        fig, axes = plt.subplots(3, n_examples, figsize=(4*n_examples, 14))
        fig.suptitle(f'Pattern Examples: {pattern}', fontsize=16, fontweight='bold')
        
        # axes[0] accuracy, axes[1] raw gradients, axes[2] smoothed gradients
        
        # Check if this pattern has a corresponding strict pattern
        strict_pattern_col = pattern.replace('_pattern', '_strict_pattern')
        has_strict = strict_pattern_col in df.columns
        
        # Create title with strict pattern info
        title = f'Pattern Examples: {pattern}'
        if has_strict:
            strict_count = df[strict_pattern_col].sum()
            title += f' (Strict: {strict_count}/{len(df)})'
        
        for i in range(n_examples):
            run_name = pattern_runs[i]
            
            # Find the run directory
            # The run directories are always in outputs/, not in the input_folder
            # input_folder is only used for the CSV location
            if subfolder:
                run_dir = os.path.join('outputs', subfolder, run_name)
            else:
                run_dir = os.path.join('outputs', run_name)

            # Fallback: if computed path doesn't exist, try to locate the run anywhere under outputs/
            if not os.path.isdir(run_dir):
                import glob
                candidates = glob.glob(os.path.join('outputs', '**', run_name), recursive=True)
                if candidates:
                    run_dir = candidates[0]
                    print(f"Resolved run directory via fallback: {run_dir}")
                else:
                    print(f"Warning: Could not locate run directory for {run_name} under outputs/.")

            # Try to load training history first, fall back to PNG if not available
            history_file = os.path.join(run_dir, 'training_history.json')
            gradient_png = os.path.join(run_dir, 'gradients.png')
            
            if os.path.exists(history_file):
                # Use raw data from training history
                try:
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    
                    epochs = list(range(1, len(history['accuracy']) + 1))
                    
                    # Check if this run meets strict criteria
                    is_strict = False
                    if has_strict:
                        run_data = df[df['run_name'] == run_name]
                        if len(run_data) > 0:
                            is_strict = run_data.iloc[0][strict_pattern_col] == 1
                    
                    # Plot accuracy (top row)
                    if n_examples == 1:
                        axes[0].plot(epochs, history['accuracy'], label='Train Acc', linewidth=2, color='blue')
                        if 'test_accuracy' in history and history['test_accuracy']:
                            axes[0].plot(epochs, history['test_accuracy'], label='Test Acc', linewidth=2, color='red')
                        axes[0].set_xlabel('Epoch')
                        axes[0].set_ylabel('Accuracy (%)')
                        title = f'{run_name}\nAccuracy'
                        if is_strict:
                            title += ' [STRICT]'
                        axes[0].set_title(title, fontsize=10, color='red' if is_strict else 'black')
                        axes[0].legend(fontsize=8)
                        axes[0].grid(True, alpha=0.3)
                    else:
                        axes[0, i].plot(epochs, history['accuracy'], label='Train Acc', linewidth=2, color='blue')
                        if 'test_accuracy' in history and history['test_accuracy']:
                            axes[0, i].plot(epochs, history['test_accuracy'], label='Test Acc', linewidth=2, color='red')
                        axes[0, i].set_xlabel('Epoch')
                        axes[0, i].set_ylabel('Accuracy (%)')
                        title = f'{run_name}\nAccuracy'
                        if is_strict:
                            title += ' [STRICT]'
                        axes[0, i].set_title(title, fontsize=10, color='red' if is_strict else 'black')
                        axes[0, i].legend(fontsize=8)
                        axes[0, i].grid(True, alpha=0.3)
                    
                    # Plot raw and smoothed gradients (middle and bottom rows)
                    gradient_epochs = history['gradients']['epoch']
                    layer_names = _get_primary_layers_from_history(history)

                    # Raw gradients
                    for layer in layer_names:
                        if layer in history['gradients']:
                            series = history['gradients'][layer]
                            if n_examples == 1:
                                axes[1].plot(
                                    gradient_epochs,
                                    series,
                                    label=layer,
                                    linewidth=2,
                                    color=get_layer_color(layer),
                                )
                            else:
                                axes[1, i].plot(
                                    gradient_epochs,
                                    series,
                                    label=layer,
                                    linewidth=2,
                                    color=get_layer_color(layer),
                                )

                    # Smoothed gradients
                    for layer in layer_names:
                        if layer in history['gradients']:
                            series = history['gradients'][layer]
                            series_s = smooth_gradients(series, 3)
                            if n_examples == 1:
                                axes[2].plot(
                                    gradient_epochs,
                                    series_s,
                                    label=layer,
                                    linewidth=2,
                                    color=get_layer_color(layer),
                                )
                            else:
                                axes[2, i].plot(
                                    gradient_epochs,
                                    series_s,
                                    label=layer,
                                    linewidth=2,
                                    color=get_layer_color(layer),
                                )

                    if n_examples == 1:
                        axes[1].set_xlabel('Epoch')
                        axes[1].set_ylabel('Gradient Norm')
                        title_mid = 'Gradients (raw)'
                        if is_strict:
                            title_mid += ' [STRICT]'
                        axes[1].set_title(title_mid, fontsize=10, color='red' if is_strict else 'black')
                        axes[1].legend(fontsize=8)
                        axes[1].grid(True, alpha=0.3)

                        axes[2].set_xlabel('Epoch')
                        axes[2].set_ylabel('Gradient Norm')
                        title_bot = 'Gradients (smoothed)'
                        if is_strict:
                            title_bot += ' [STRICT]'
                        axes[2].set_title(title_bot, fontsize=10, color='red' if is_strict else 'black')
                        axes[2].legend(fontsize=8)
                        axes[2].grid(True, alpha=0.3)
                    else:
                        axes[1, i].set_xlabel('Epoch')
                        axes[1, i].set_ylabel('Gradient Norm')
                        title_mid = 'Gradients (raw)'
                        if is_strict:
                            title_mid += ' [STRICT]'
                        axes[1, i].set_title(title_mid, fontsize=10, color='red' if is_strict else 'black')
                        axes[1, i].legend(fontsize=8)
                        axes[1, i].grid(True, alpha=0.3)

                        axes[2, i].set_xlabel('Epoch')
                        axes[2, i].set_ylabel('Gradient Norm')
                        title_bot = 'Gradients (smoothed)'
                        if is_strict:
                            title_bot += ' [STRICT]'
                        axes[2, i].set_title(title_bot, fontsize=10, color='red' if is_strict else 'black')
                        axes[2, i].legend(fontsize=8)
                        axes[2, i].grid(True, alpha=0.3)
                    
                except Exception as e:
                    if n_examples == 1:
                        axes[0].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', 
                                    ha='center', va='center', transform=axes[0].transAxes)
                        axes[0].set_title(run_name, fontsize=10)
                        axes[1].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', 
                                    ha='center', va='center', transform=axes[1].transAxes)
                        axes[1].set_title('Gradients', fontsize=10)
                        axes[2].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', 
                                    ha='center', va='center', transform=axes[2].transAxes)
                        axes[2].set_title('Gradients', fontsize=10)
                    else:
                        axes[0, i].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', 
                                        ha='center', va='center', transform=axes[0, i].transAxes)
                        axes[0, i].set_title(run_name, fontsize=10)
                        axes[1, i].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', 
                                        ha='center', va='center', transform=axes[1, i].transAxes)
                        axes[1, i].set_title('Gradients', fontsize=10)
                        axes[2, i].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', 
                                        ha='center', va='center', transform=axes[2, i].transAxes)
                        axes[2, i].set_title('Gradients', fontsize=10)
                    
            elif os.path.exists(gradient_png):
                # Fall back to existing PNG for gradients only
                try:
                    img = plt.imread(gradient_png)
                    if n_examples == 1:
                        axes[0].text(0.5, 0.5, f'No accuracy data\nfor {run_name}', 
                                    ha='center', va='center', transform=axes[0].transAxes)
                        axes[0].set_title(f'{run_name}\nAccuracy', fontsize=10)
                        axes[1].imshow(img)
                        axes[1].set_title('Gradients', fontsize=10)
                        axes[1].axis('off')
                        axes[2].imshow(img)
                        axes[2].set_title('Gradients', fontsize=10)
                        axes[2].axis('off')
                    else:
                        axes[0, i].text(0.5, 0.5, f'No accuracy data\nfor {run_name}', 
                                        ha='center', va='center', transform=axes[0, i].transAxes)
                        axes[0, i].set_title(f'{run_name}\nAccuracy', fontsize=10)
                        axes[1, i].imshow(img)
                        axes[1, i].set_title('Gradients', fontsize=10)
                        axes[1, i].axis('off')
                        axes[2, i].imshow(img)
                        axes[2, i].set_title('Gradients', fontsize=10)
                        axes[2, i].axis('off')
                except Exception as e:
                    if n_examples == 1:
                        axes[0].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', 
                                    ha='center', va='center', transform=axes[0].transAxes)
                        axes[0].set_title(f'{run_name}\nAccuracy', fontsize=10)
                        axes[1].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', 
                                    ha='center', va='center', transform=axes[1].transAxes)
                        axes[1].set_title('Gradients', fontsize=10)
                        axes[2].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', 
                                    ha='center', va='center', transform=axes[2].transAxes)
                        axes[2].set_title('Gradients', fontsize=10)
                    else:
                        axes[0, i].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', 
                                        ha='center', va='center', transform=axes[0, i].transAxes)
                        axes[0, i].set_title(f'{run_name}\nAccuracy', fontsize=10)
                        axes[1, i].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', 
                                        ha='center', va='center', transform=axes[1, i].transAxes)
                        axes[1, i].set_title('Gradients', fontsize=10)
                        axes[2, i].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', 
                                        ha='center', va='center', transform=axes[2, i].transAxes)
                        axes[2, i].set_title('Gradients', fontsize=10)
            else:
                # No data available
                if n_examples == 1:
                    axes[0].text(0.5, 0.5, f'No data\nfor {run_name}', 
                                ha='center', va='center', transform=axes[0].transAxes)
                    axes[0].set_title(f'{run_name}\nAccuracy', fontsize=10)
                    axes[1].text(0.5, 0.5, f'No data\nfor {run_name}', 
                                ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title('Gradients', fontsize=10)
                    axes[2].text(0.5, 0.5, f'No data\nfor {run_name}', 
                                ha='center', va='center', transform=axes[2].transAxes)
                    axes[2].set_title('Gradients', fontsize=10)
                else:
                    axes[0, i].text(0.5, 0.5, f'No data\nfor {run_name}', 
                                    ha='center', va='center', transform=axes[0, i].transAxes)
                    axes[0, i].set_title(f'{run_name}\nAccuracy', fontsize=10)
                    axes[1, i].text(0.5, 0.5, f'No data\nfor {run_name}', 
                                    ha='center', va='center', transform=axes[1, i].transAxes)
                    axes[1, i].set_title('Gradients', fontsize=10)
                    axes[2, i].text(0.5, 0.5, f'No data\nfor {run_name}', 
                                    ha='center', va='center', transform=axes[2, i].transAxes)
                    axes[2, i].set_title('Gradients', fontsize=10)
        
        # Hide unused subplots if we have fewer than 6 examples
        if n_examples < 6 and len(axes.shape) > 1:
            for i in range(n_examples, axes.shape[1]):
                axes[0, i].set_visible(False)
                axes[1, i].set_visible(False)
                axes[2, i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        title_suffix = input_folder.replace('outputs/', '').replace('/', '_') if input_folder else "current"
        filename = os.path.join(output_folder, f'pattern_examples_{pattern}_{title_suffix}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Pattern examples plot saved to: {filename}")

def create_no_pattern_examples(df, output_folder=".", input_folder="", subfolder="", title_suffix="", n_examples=6):
    """Create a plot of runs that have no patterns at all (all *_pattern == 0)."""
    import os
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    import glob

    # Identify pattern columns (includes strict patterns as well)
    pattern_columns = [col for col in df.columns if col.endswith('_pattern')]
    if not pattern_columns:
        print("No pattern columns found; skipping no-pattern examples plot.")
        return

    # Select runs with zero across all pattern columns
    no_pattern_df = df[df[pattern_columns].sum(axis=1) == 0]
    if no_pattern_df.empty:
        print("No runs without patterns; skipping no-pattern examples plot.")
        return

    run_names = no_pattern_df['run_name'].tolist()
    n_examples = min(n_examples, len(run_names))
    print(f"Creating examples for runs without any pattern ({len(run_names)} found, showing {n_examples})")

    # Build figure: 3 rows (accuracy, raw gradients, smoothed gradients) x n_examples columns
    fig, axes = plt.subplots(3, n_examples, figsize=(4*n_examples, 14))
    fig.suptitle(f'No-Pattern Examples: {title_suffix}', fontsize=16, fontweight='bold')

    # Normalize axes access for n_examples==1
    if n_examples == 1:
        axes = np.array(axes).reshape(3, 1)

    for i in range(n_examples):
        run_name = run_names[i]

        # Locate run directory under outputs
        if subfolder:
            run_dir = os.path.join('outputs', subfolder, run_name)
        else:
            run_dir = os.path.join('outputs', run_name)

        if not os.path.isdir(run_dir):
            cands = glob.glob(os.path.join('outputs', '**', run_name), recursive=True)
            if cands:
                run_dir = cands[0]
                print(f"Resolved run directory via fallback: {run_dir}")
            else:
                print(f"Warning: Could not locate run directory for {run_name} under outputs/.")

        history_file = os.path.join(run_dir, 'training_history.json')

        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)

                epochs = list(range(1, len(history.get('accuracy', [])) + 1))

                # Accuracy plot (top)
                if epochs:
                    axes[0, i].plot(epochs, history.get('accuracy', []), label='Train Acc', linewidth=2, color='blue')
                    if 'test_accuracy' in history and history['test_accuracy']:
                        axes[0, i].plot(epochs, history['test_accuracy'], label='Test Acc', linewidth=2, color='red')
                    axes[0, i].set_xlabel('Epoch')
                    axes[0, i].set_ylabel('Accuracy (%)')
                    axes[0, i].set_title(f'{run_name}\nAccuracy', fontsize=10)
                    axes[0, i].legend(fontsize=8)
                    axes[0, i].grid(True, alpha=0.3)
                else:
                    axes[0, i].text(0.5, 0.5, 'No accuracy data', ha='center', va='center', transform=axes[0, i].transAxes)
                    axes[0, i].set_title(f'{run_name}\n(No accuracy)')

                # Gradients plots: raw (middle), smoothed (bottom)
                if 'gradients' in history and len(history['gradients'].get('epoch', [])) > 0:
                    grad_epochs = history['gradients']['epoch']
                    layer_names = _get_primary_layers_from_history(history)
                    # Raw
                    for layer in layer_names:
                        if layer in history['gradients']:
                            axes[1, i].plot(
                                grad_epochs,
                                history['gradients'][layer],
                                label=layer,
                                linewidth=2,
                                color=get_layer_color(layer),
                            )
                    axes[1, i].set_xlabel('Epoch')
                    axes[1, i].set_ylabel('Gradient Norm')
                    axes[1, i].set_title('Gradients (raw)', fontsize=10)
                    axes[1, i].legend(fontsize=8)
                    axes[1, i].grid(True, alpha=0.3)

                    # Smoothed (3-epoch)
                    for layer in layer_names:
                        if layer in history['gradients']:
                            series = history['gradients'][layer]
                            axes[2, i].plot(
                                grad_epochs,
                                smooth_gradients(series, 3),
                                label=layer,
                                linewidth=2,
                                color=get_layer_color(layer),
                            )
                    axes[2, i].set_xlabel('Epoch')
                    axes[2, i].set_ylabel('Gradient Norm')
                    axes[2, i].set_title('Gradients (smoothed)', fontsize=10)
                    axes[2, i].legend(fontsize=8)
                    axes[2, i].grid(True, alpha=0.3)
                else:
                    axes[1, i].text(0.5, 0.5, 'No gradient data', ha='center', va='center', transform=axes[1, i].transAxes)
                    axes[1, i].set_title('Gradients (No data)', fontsize=10)
                    axes[2, i].text(0.5, 0.5, 'No gradient data', ha='center', va='center', transform=axes[2, i].transAxes)
                    axes[2, i].set_title('Gradients (No data)', fontsize=10)
            except Exception as e:
                axes[0, i].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', ha='center', va='center', transform=axes[0, i].transAxes)
                axes[0, i].set_title(run_name, fontsize=10)
                axes[1, i].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', ha='center', va='center', transform=axes[1, i].transAxes)
                axes[1, i].set_title('Gradients', fontsize=10)
                axes[2, i].text(0.5, 0.5, f'Error loading\n{run_name}\n{str(e)}', ha='center', va='center', transform=axes[2, i].transAxes)

    plt.tight_layout()
    filename = os.path.join(output_folder, f'no_pattern_examples_{title_suffix}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"No-pattern examples plot saved to: {filename}")

if __name__ == '__main__':
    main()
