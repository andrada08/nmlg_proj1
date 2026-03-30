#!/usr/bin/env python3
"""
Plot activation and gradient tracking data from the first epoch.
Creates visualizations of activation/gradient evolution and comparisons.

Usage:
    python plot_activations_gradients_first_epoch.py --subfolder three_layer_skip_with_50_100_150
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import sys
import os
from pathlib import Path


def load_analysis_csv(input_folder):
    """Load the analysis CSV"""
    csv_path = Path(input_folder) / 'activation_gradient_analysis_first_epoch.csv'
    if not csv_path.exists():
        print(f"{csv_path} not found. Run analyze_activations_gradients_first_epoch.py first.")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")
    return df


def load_tracking_data(subfolder, run_name):
    """Load the per-step tracking data for a specific run"""
    tracking_file = Path('outputs') / 'activation_gradient_analysis_first_epoch' / subfolder / run_name / 'activation_gradient_analysis_first_epoch.json'
    if not tracking_file.exists():
        return None
    
    with open(tracking_file, 'r') as f:
        return json.load(f)


def plot_activation_evolution(tracking_data, output_path, run_name):
    """Plot activation evolution over steps for a single run"""
    activation_points = list(tracking_data['activations'].keys())
    steps = tracking_data['steps']
    
    fig, axes = plt.subplots(len(activation_points), 2, figsize=(14, 4 * len(activation_points)))
    if len(activation_points) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, point_name in enumerate(activation_points):
        point_data = tracking_data['activations'][point_name]
        
        # Plot norm and mean (with dual y-axes due to scale difference)
        ax1 = axes[idx, 0]
        if 'norm' in point_data:
            ax1.plot(steps, point_data['norm'], label='Norm', linewidth=1.5, color='blue')
            ax1.set_ylabel('Norm', color='blue', fontsize=10)
            ax1.tick_params(axis='y', labelcolor='blue')
        if 'mean' in point_data:
            ax1_twin = ax1.twinx()
            ax1_twin.plot(steps, point_data['mean'], label='Mean', linewidth=1.5, color='red', alpha=0.7)
            ax1_twin.set_ylabel('Mean', color='red', fontsize=10)
            ax1_twin.tick_params(axis='y', labelcolor='red')
        ax1.set_xlabel('Step')
        ax1.set_title(f'{point_name}: Norm and Mean')
        ax1.grid(True, alpha=0.3)
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        if 'mean' in point_data:
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(loc='upper left')
        
        # Plot sparsity and std
        ax2 = axes[idx, 1]
        if 'sparsity' in point_data:
            ax2.plot(steps, point_data['sparsity'], label='Sparsity', linewidth=1.5, color='orange')
        if 'std' in point_data:
            ax2_twin = ax2.twinx()
            ax2_twin.plot(steps, point_data['std'], label='Std', linewidth=1.5, color='green', alpha=0.7)
            ax2_twin.set_ylabel('Std', color='green')
            ax2_twin.tick_params(axis='y', labelcolor='green')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Sparsity', color='orange')
        ax2.set_title(f'{point_name}: Sparsity and Std')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.grid(True, alpha=0.3)
        if 'std' in point_data:
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
        else:
            ax2.legend()
    
    plt.suptitle(f'Activation Evolution: {run_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


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


def plot_gradient_evolution(tracking_data, output_path, run_name):
    """Plot gradient evolution over steps for a single run (raw and smoothed)"""
    all_layer_names = list(tracking_data['gradients'].keys())
    # Filter to only main layers (exclude "from" layers like layer3_from_1, layer3_from_2)
    layer_names = [name for name in all_layer_names if '_from_' not in name]
    steps = tracking_data['steps']
    
    # Create two subplots: raw on top, smoothed on bottom
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(layer_names)))
    
    # Plot raw gradients on top
    for idx, layer_name in enumerate(layer_names):
        if layer_name in tracking_data['gradients']:
            values = tracking_data['gradients'][layer_name]
            ax1.plot(steps, values, label=layer_name, linewidth=1.5, color=colors[idx], alpha=0.8)
    
    ax1.set_ylabel('Gradient Norm (Raw)', fontsize=12)
    ax1.set_title(f'Gradient Evolution: {run_name}', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot smoothed gradients on bottom
    for idx, layer_name in enumerate(layer_names):
        if layer_name in tracking_data['gradients']:
            values = tracking_data['gradients'][layer_name]
            smoothed = smooth_gradients(values, window=3)
            ax2.plot(steps, smoothed, label=layer_name, linewidth=1.5, color=colors[idx], alpha=0.8)
    
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Gradient Norm (Smoothed)', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_activation_gradient_correlation(tracking_data, output_path, run_name):
    """Plot correlation between activations and gradients"""
    activation_points = list(tracking_data['activations'].keys())
    all_layer_names = list(tracking_data['gradients'].keys())
    # Filter to only main layers (exclude "from" layers)
    layer_names = [name for name in all_layer_names if '_from_' not in name]
    
    # Create a grid of subplots
    n_points = len(activation_points)
    n_layers = len(layer_names)
    
    fig, axes = plt.subplots(n_points, n_layers, figsize=(4 * n_layers, 4 * n_points))
    if n_points == 1:
        axes = axes.reshape(1, -1)
    if n_layers == 1:
        axes = axes.reshape(-1, 1)
    
    steps = tracking_data['steps']
    
    for i, point_name in enumerate(activation_points):
        for j, layer_name in enumerate(layer_names):
            ax = axes[i, j]
            
            # Get activation norm and gradient norm
            if 'norm' in tracking_data['activations'][point_name]:
                act_norms = tracking_data['activations'][point_name]['norm']
                grad_norms = tracking_data['gradients'][layer_name]
                
                # Scatter plot
                ax.scatter(act_norms, grad_norms, alpha=0.3, s=10)
                
                # Compute correlation
                if len(act_norms) > 1 and len(grad_norms) > 1:
                    corr = np.corrcoef(act_norms, grad_norms)[0, 1]
                    ax.set_title(f'{point_name} vs {layer_name}\n(corr={corr:.3f})', fontsize=10)
                else:
                    ax.set_title(f'{point_name} vs {layer_name}', fontsize=10)
                
                ax.set_xlabel(f'{point_name} Norm')
                ax.set_ylabel(f'{layer_name} Grad Norm')
                ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Activation-Gradient Correlations: {run_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_meaningful_correlations(df, subfolder, output_folder, n_runs=6):
    """Plot most meaningful activation-gradient correlations for multiple runs"""
    # Extract architecture name from subfolder
    arch_parts = subfolder.split('_')
    if 'three' in arch_parts and 'layer' in arch_parts and 'skip' in arch_parts:
        arch_name = "Three Layer Skip"
        # Most meaningful correlations for three_layer_skip:
        # - x1 vs layer3 gradient (skip connection)
        # - x2 vs layer3 gradient (skip connection)
        # - x3_combined vs layer3 gradient (direct input)
        # - x1 vs layer2 gradient (sequential flow)
        # - x2 vs layer2 gradient (self-relationship)
        meaningful_pairs = [
            ('x1', 'layer3'),
            ('x2', 'layer3'),
            ('x3_combined', 'layer3'),
            ('x1', 'layer2'),
            ('x2', 'layer2'),
        ]
    elif 'four' in arch_parts and 'layer' in arch_parts and 'integrating' in arch_parts:
        arch_name = "Four Layer Integrating"
        # Most meaningful for four_layer_integrating
        meaningful_pairs = [
            ('x1', 'layer3'), ('x2', 'layer3'), ('x3', 'layer4'),
            ('x2', 'layer4'), ('x4', 'layer4'),
        ]
    elif 'four' in arch_parts and 'layer' in arch_parts and 'sequential' in arch_parts:
        arch_name = "Four Layer Sequential"
        # Most meaningful for four_layer_sequential
        meaningful_pairs = [
            ('x1', 'layer3'), ('x2', 'layer3'), ('x3', 'layer4'),
            ('x4', 'layer4'),
        ]
    else:
        arch_name = subfolder.replace('_', ' ').title()
        meaningful_pairs = []
    
    # Select same runs as other combined plots
    pattern_match = df['run_name'].str.extract(r'_(low|high)_(balanced|uniform|extreme)')
    df['lr_pattern'] = pattern_match.apply(lambda x: f"{x[0]}_{x[1]}" if pd.notna(x[0]) and pd.notna(x[1]) else 'unknown', axis=1)
    
    selected_runs = []
    for pattern in df['lr_pattern'].unique():
        pattern_runs = df[df['lr_pattern'] == pattern]
        if len(pattern_runs) > 0 and len(selected_runs) < n_runs:
            selected_runs.append(pattern_runs.iloc[0])
    
    if len(selected_runs) < n_runs:
        remaining = df[~df.index.isin([r.name for r in selected_runs])]
        if len(remaining) > 0:
            additional = remaining.sample(min(n_runs - len(selected_runs), len(remaining)), random_state=42)
            selected_runs.extend([additional.iloc[i] for i in range(len(additional))])
    
    selected_runs = selected_runs[:n_runs]
    
    if not meaningful_pairs:
        print("No meaningful pairs defined for this architecture")
        return
    
    # Create plot: len(meaningful_pairs) rows x n_runs columns
    fig, axes = plt.subplots(len(meaningful_pairs), n_runs, 
                            figsize=(3 * n_runs, 3 * len(meaningful_pairs)), sharex=True, sharey='row')
    if len(meaningful_pairs) == 1:
        axes = axes.reshape(1, -1)
    
    import re
    
    for pair_idx, (act_point, layer_name) in enumerate(meaningful_pairs):
        for run_idx, row in enumerate(selected_runs):
            run_name = row['run_name']
            tracking_data = load_tracking_data(subfolder, run_name)
            
            if not tracking_data:
                continue
            
            ax = axes[pair_idx, run_idx]
            
            # Extract short title
            sizes_match = re.search(r'l(\d+)-(\d+)-(\d+)', run_name)
            lr_match = re.search(r'lr([\de-]+)-([\de-]+)-([\de-]+)', run_name)
            short_title = ""
            if sizes_match:
                sizes = '-'.join(sizes_match.groups())
                short_title = f"L:{sizes}"
            if lr_match:
                lrs = '-'.join([lr.replace('e-', 'e') for lr in lr_match.groups()])
                if short_title:
                    short_title += f"\nLR:{lrs}"
                else:
                    short_title = f"LR:{lrs}"
            if not short_title:
                short_title = run_name[:20]
            
            # Get activation norm and gradient norm
            if (act_point in tracking_data['activations'] and 
                'norm' in tracking_data['activations'][act_point] and
                layer_name in tracking_data['gradients']):
                
                act_norms = tracking_data['activations'][act_point]['norm']
                grad_norms = tracking_data['gradients'][layer_name]
                
                # Scatter plot
                ax.scatter(act_norms, grad_norms, alpha=0.3, s=8)
                
                # Compute correlation
                if len(act_norms) > 1 and len(grad_norms) > 1:
                    corr = np.corrcoef(act_norms, grad_norms)[0, 1]
                    ax.set_title(f'{short_title}\n(corr={corr:.3f})', fontsize=7, fontweight='bold')
                else:
                    ax.set_title(short_title, fontsize=7, fontweight='bold')
                
                if run_idx == 0:
                    ax.set_ylabel(f'{act_point} vs {layer_name}\nGrad Norm', fontsize=9)
                if pair_idx == len(meaningful_pairs) - 1:
                    ax.set_xlabel(f'{act_point} Norm', fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(short_title, fontsize=7, fontweight='bold')
    
    plt.suptitle(f'{arch_name}: Meaningful Activation-Gradient Correlations', fontsize=16, fontweight='bold', y=0.995)
    plt.subplots_adjust(top=0.93)
    plt.tight_layout()
    output_path = Path(output_folder) / 'activation_gradient_correlations_meaningful.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def compute_rolling_correlation(x, y, window_size=50):
    """Compute rolling correlation between two time series"""
    if len(x) != len(y) or len(x) < window_size:
        return [], []
    
    correlations = []
    centers = []
    
    for i in range(window_size, len(x) + 1):
        window_x = x[i - window_size:i]
        window_y = y[i - window_size:i]
        
        if len(window_x) > 1 and len(window_y) > 1:
            corr = np.corrcoef(window_x, window_y)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
                centers.append(i - window_size // 2)  # Center of window
    
    return centers, correlations


def plot_combined_correlations_over_time(df, subfolder, output_folder, n_runs=6, window_size=50):
    """Plot how activation-gradient correlations evolve over time for multiple runs"""
    # Extract architecture name from subfolder
    arch_parts = subfolder.split('_')
    if 'three' in arch_parts and 'layer' in arch_parts and 'skip' in arch_parts:
        arch_name = "Three Layer Skip"
        meaningful_pairs = [
            ('x1', 'layer3'),
            ('x2', 'layer3'),
            ('x3_combined', 'layer3'),
            ('x1', 'layer2'),
            ('x2', 'layer2'),
        ]
    elif 'four' in arch_parts and 'layer' in arch_parts and 'integrating' in arch_parts:
        arch_name = "Four Layer Integrating"
        meaningful_pairs = [
            ('x1', 'layer3'), ('x2', 'layer3'), ('x3', 'layer4'),
            ('x2', 'layer4'), ('x4', 'layer4'),
        ]
    elif 'four' in arch_parts and 'layer' in arch_parts and 'sequential' in arch_parts:
        arch_name = "Four Layer Sequential"
        meaningful_pairs = [
            ('x1', 'layer3'), ('x2', 'layer3'), ('x3', 'layer4'),
            ('x4', 'layer4'),
        ]
    else:
        arch_name = subfolder.replace('_', ' ').title()
        meaningful_pairs = []
    
    # Select same runs as other combined plots
    pattern_match = df['run_name'].str.extract(r'_(low|high)_(balanced|uniform|extreme)')
    df['lr_pattern'] = pattern_match.apply(lambda x: f"{x[0]}_{x[1]}" if pd.notna(x[0]) and pd.notna(x[1]) else 'unknown', axis=1)
    
    selected_runs = []
    for pattern in df['lr_pattern'].unique():
        pattern_runs = df[df['lr_pattern'] == pattern]
        if len(pattern_runs) > 0 and len(selected_runs) < n_runs:
            selected_runs.append(pattern_runs.iloc[0])
    
    if len(selected_runs) < n_runs:
        remaining = df[~df.index.isin([r.name for r in selected_runs])]
        if len(remaining) > 0:
            additional = remaining.sample(min(n_runs - len(selected_runs), len(remaining)), random_state=42)
            selected_runs.extend([additional.iloc[i] for i in range(len(additional))])
    
    selected_runs = selected_runs[:n_runs]
    
    if not meaningful_pairs:
        print("No meaningful pairs defined for this architecture")
        return
    
    # Create plot: len(meaningful_pairs) rows x n_runs columns
    fig, axes = plt.subplots(len(meaningful_pairs), n_runs, 
                            figsize=(3 * n_runs, 3 * len(meaningful_pairs)), sharex=True, sharey='row')
    if len(meaningful_pairs) == 1:
        axes = axes.reshape(1, -1)
    
    import re
    
    for pair_idx, (act_point, layer_name) in enumerate(meaningful_pairs):
        for run_idx, row in enumerate(selected_runs):
            run_name = row['run_name']
            tracking_data = load_tracking_data(subfolder, run_name)
            
            if not tracking_data:
                continue
            
            ax = axes[pair_idx, run_idx]
            
            # Extract short title
            sizes_match = re.search(r'l(\d+)-(\d+)-(\d+)', run_name)
            lr_match = re.search(r'lr([\de-]+)-([\de-]+)-([\de-]+)', run_name)
            short_title = ""
            if sizes_match:
                sizes = '-'.join(sizes_match.groups())
                short_title = f"L:{sizes}"
            if lr_match:
                lrs = '-'.join([lr.replace('e-', 'e') for lr in lr_match.groups()])
                if short_title:
                    short_title += f"\nLR:{lrs}"
                else:
                    short_title = f"LR:{lrs}"
            if not short_title:
                short_title = run_name[:20]
            
            # Get activation norm and gradient norm
            if (act_point in tracking_data['activations'] and 
                'norm' in tracking_data['activations'][act_point] and
                layer_name in tracking_data['gradients']):
                
                act_norms = np.array(tracking_data['activations'][act_point]['norm'])
                grad_norms = np.array(tracking_data['gradients'][layer_name])
                
                # Compute rolling correlation
                steps_centers, correlations = compute_rolling_correlation(act_norms, grad_norms, window_size=window_size)
                
                if len(correlations) > 0:
                    ax.plot(steps_centers, correlations, linewidth=1.5, alpha=0.8)
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
                    ax.set_title(short_title, fontsize=7, fontweight='bold')
                    
                    if run_idx == 0:
                        ax.set_ylabel(f'{act_point} vs {layer_name}\nCorrelation', fontsize=9)
                    if pair_idx == len(meaningful_pairs) - 1:
                        ax.set_xlabel('Step', fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(-1, 1)  # Correlation range
                else:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(short_title, fontsize=7, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(short_title, fontsize=7, fontweight='bold')
    
    plt.suptitle(f'{arch_name}: Activation-Gradient Correlations Over Time', fontsize=16, fontweight='bold', y=0.995)
    plt.subplots_adjust(top=0.93)
    plt.tight_layout()
    output_path = Path(output_folder) / 'activation_gradient_correlations_over_time.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_comparisons(df, output_folder, subfolder):
    """Create summary comparison plots across all runs"""
    
    # Plot 1: Activation norm trends by learning rate pattern
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract learning rate pattern from run_name
    pattern_match = df['run_name'].str.extract(r'_(low|high)_(balanced|uniform|extreme)')
    df['lr_pattern'] = pattern_match.apply(lambda x: f"{x[0]}_{x[1]}" if pd.notna(x[0]) and pd.notna(x[1]) else 'unknown', axis=1)
    
    # Activation norm trends
    activation_cols = [c for c in df.columns if '_norm_trend' in c and any(ap in c for ap in ['x1', 'x2', 'x3'])]
    
    if activation_cols:
        ax = axes[0, 0]
        data_to_plot = []
        for col in activation_cols[:3]:  # Limit to first 3
            for pattern in df['lr_pattern'].unique():
                subset = df[df['lr_pattern'] == pattern][col].dropna()
                if len(subset) > 0:
                    data_to_plot.append({
                        'Metric': col.replace('_norm_trend', ''),
                        'LR Pattern': pattern,
                        'Trend': subset.mean()
                    })
        
        if data_to_plot:
            plot_df = pd.DataFrame(data_to_plot)
            sns.barplot(data=plot_df, x='LR Pattern', y='Trend', hue='Metric', ax=ax)
            ax.set_title('Activation Norm Trends by LR Pattern')
            ax.tick_params(axis='x', rotation=45)
    
    # Gradient trends
    grad_cols = [c for c in df.columns if '_grad_trend' in c]
    if grad_cols:
        ax = axes[0, 1]
        data_to_plot = []
        for col in grad_cols[:5]:  # Limit to first 5 layers
            for pattern in df['lr_pattern'].unique():
                subset = df[df['lr_pattern'] == pattern][col].dropna()
                if len(subset) > 0:
                    data_to_plot.append({
                        'Layer': col.replace('_grad_trend', ''),
                        'LR Pattern': pattern,
                        'Trend': subset.mean()
                    })
        
        if data_to_plot:
            plot_df = pd.DataFrame(data_to_plot)
            sns.barplot(data=plot_df, x='LR Pattern', y='Trend', hue='Layer', ax=ax)
            ax.set_title('Gradient Trends by LR Pattern')
            ax.tick_params(axis='x', rotation=45)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Activation sparsity comparison
    sparsity_cols = [c for c in df.columns if '_sparsity_mean' in c and any(ap in c for ap in ['x1', 'x2', 'x3'])]
    if sparsity_cols:
        ax = axes[1, 0]
        data_to_plot = []
        for col in sparsity_cols[:3]:
            for pattern in df['lr_pattern'].unique():
                subset = df[df['lr_pattern'] == pattern][col].dropna()
                if len(subset) > 0:
                    data_to_plot.append({
                        'Metric': col.replace('_sparsity_mean', ''),
                        'LR Pattern': pattern,
                        'Sparsity': subset.mean()
                    })
        
        if data_to_plot:
            plot_df = pd.DataFrame(data_to_plot)
            sns.barplot(data=plot_df, x='LR Pattern', y='Sparsity', hue='Metric', ax=ax)
            ax.set_title('Activation Sparsity by LR Pattern')
            ax.tick_params(axis='x', rotation=45)
    
    # Loss trend
    if 'loss_trend' in df.columns:
        ax = axes[1, 1]
        data_to_plot = []
        for pattern in df['lr_pattern'].unique():
            subset = df[df['lr_pattern'] == pattern]['loss_trend'].dropna()
            if len(subset) > 0:
                data_to_plot.append({
                    'LR Pattern': pattern,
                    'Loss Trend': subset.mean()
                })
        
        if data_to_plot:
            plot_df = pd.DataFrame(data_to_plot)
            sns.barplot(data=plot_df, x='LR Pattern', y='Loss Trend', ax=ax)
            ax.set_title('Loss Trend by LR Pattern')
            ax.tick_params(axis='x', rotation=45)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'Summary Comparisons: {subfolder}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = Path(output_folder) / 'summary_comparisons.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_gradient_evolution(df, subfolder, output_folder, n_runs=6):
    """Plot gradient evolution for multiple runs in a single combined plot"""
    # Extract architecture name from subfolder and convert to readable format
    # e.g., "three_layer_skip_with_50_100_150" -> "Three Layer Skip"
    # e.g., "four_layer_integrating_all_sweeps" -> "Four Layer Integrating"
    arch_parts = subfolder.split('_')
    if 'three' in arch_parts and 'layer' in arch_parts and 'skip' in arch_parts:
        arch_name = "Three Layer Skip"
    elif 'four' in arch_parts and 'layer' in arch_parts and 'integrating' in arch_parts:
        arch_name = "Four Layer Integrating"
    elif 'four' in arch_parts and 'layer' in arch_parts and 'sequential' in arch_parts:
        arch_name = "Four Layer Sequential"
    else:
        # Fallback: capitalize and format
        arch_name = subfolder.replace('_', ' ').title()
    
    # Select diverse examples (one from each LR pattern if possible)
    pattern_match = df['run_name'].str.extract(r'_(low|high)_(balanced|uniform|extreme)')
    df['lr_pattern'] = pattern_match.apply(lambda x: f"{x[0]}_{x[1]}" if pd.notna(x[0]) and pd.notna(x[1]) else 'unknown', axis=1)
    
    # Try to get one example from each pattern, then fill remaining slots randomly
    selected_runs = []
    for pattern in df['lr_pattern'].unique():
        pattern_runs = df[df['lr_pattern'] == pattern]
        if len(pattern_runs) > 0 and len(selected_runs) < n_runs:
            selected_runs.append(pattern_runs.iloc[0])
    
    # Fill remaining slots randomly if needed
    if len(selected_runs) < n_runs:
        remaining = df[~df.index.isin([r.name for r in selected_runs])]
        if len(remaining) > 0:
            additional = remaining.sample(min(n_runs - len(selected_runs), len(remaining)), random_state=42)
            selected_runs.extend([additional.iloc[i] for i in range(len(additional))])
    
    # Limit to n_runs
    selected_runs = selected_runs[:n_runs]
    
    # Create combined plot: top row = accuracy (all runs), middle row = raw gradients (all runs), bottom row = smoothed gradients (all runs)
    # 3 rows x n_runs columns
    fig, axes = plt.subplots(3, len(selected_runs), figsize=(4 * len(selected_runs), 12), sharex=True)
    if len(selected_runs) == 1:
        axes = axes.reshape(-1, 1)
    
    # Get layer names and colors (assuming all runs have same architecture)
    first_tracking_data = None
    for row in selected_runs:
        tracking_data = load_tracking_data(subfolder, row['run_name'])
        if tracking_data:
            first_tracking_data = tracking_data
            break
    
    if first_tracking_data:
        all_layer_names = list(first_tracking_data['gradients'].keys())
        layer_names = [name for name in all_layer_names if '_from_' not in name]
        colors = plt.cm.tab10(np.linspace(0, 1, len(layer_names)))
    else:
        layer_names = []
        colors = []
    
    for idx, row in enumerate(selected_runs):
        run_name = row['run_name']
        tracking_data = load_tracking_data(subfolder, run_name)
        
        if not tracking_data:
            continue
        
        steps = tracking_data['steps']
        
        # Extract short title (layer sizes and learning rates)
        import re
        sizes_match = re.search(r'l(\d+)-(\d+)-(\d+)', run_name)
        lr_match = re.search(r'lr([\de-]+)-([\de-]+)-([\de-]+)', run_name)
        short_title = ""
        if sizes_match:
            sizes = '-'.join(sizes_match.groups())
            short_title = f"L:{sizes}"
        if lr_match:
            lrs = '-'.join([lr.replace('e-', 'e') for lr in lr_match.groups()])
            if short_title:
                short_title += f"\nLR:{lrs}"
            else:
                short_title = f"LR:{lrs}"
        if not short_title:
            short_title = run_name[:30]  # Fallback
        
        # Accuracy (top row)
        ax0 = axes[0, idx]
        if 'accuracy' in tracking_data and tracking_data['accuracy']:
            ax0.plot(steps, tracking_data['accuracy'], linewidth=2, color='purple', alpha=0.8)
            if idx == 0:
                ax0.set_ylabel('Accuracy (%)', fontsize=10)
            ax0.set_title(short_title, fontsize=8, fontweight='bold')
            ax0.grid(True, alpha=0.3)
        else:
            ax0.text(0.5, 0.5, 'No accuracy data', ha='center', va='center', transform=ax0.transAxes)
            ax0.set_title(short_title, fontsize=8, fontweight='bold')
        
        # Raw gradients (middle row)
        ax1 = axes[1, idx]
        for j, layer_name in enumerate(layer_names):
            if layer_name in tracking_data['gradients']:
                values = tracking_data['gradients'][layer_name]
                ax1.plot(steps, values, label=layer_name, linewidth=1.5, color=colors[j], alpha=0.8)
        
        if idx == 0:
            ax1.set_ylabel('Gradient Norm (Raw)', fontsize=10)
        if idx == 0:
            ax1.legend(fontsize=7, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Smoothed gradients (bottom row)
        ax2 = axes[2, idx]
        for j, layer_name in enumerate(layer_names):
            if layer_name in tracking_data['gradients']:
                values = tracking_data['gradients'][layer_name]
                smoothed = smooth_gradients(values, window=3)
                ax2.plot(steps, smoothed, label=layer_name, linewidth=1.5, color=colors[j], alpha=0.8)
        
        if idx == 0:
            ax2.set_ylabel('Gradient Norm (Smoothed)', fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    # Set x-label only on bottom row
    for idx in range(len(selected_runs)):
        axes[2, idx].set_xlabel('Step', fontsize=12)
    
    # Add row titles
    fig.text(0.02, 0.83, 'Accuracy', ha='center', fontsize=12, fontweight='bold', rotation=90)
    fig.text(0.02, 0.5, 'Raw Gradients', ha='center', fontsize=12, fontweight='bold', rotation=90)
    fig.text(0.02, 0.17, 'Smoothed Gradients', ha='center', fontsize=12, fontweight='bold', rotation=90)
    
    plt.suptitle(f'{arch_name}: Gradient Evolution - Multiple Runs', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(left=0.08, top=0.95)
    plt.tight_layout()
    output_path = Path(output_folder) / 'gradient_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_activation_evolution(df, subfolder, output_folder, n_runs=6):
    """Plot activation evolution for multiple runs in combined plots"""
    # Extract architecture name from subfolder
    arch_parts = subfolder.split('_')
    if 'three' in arch_parts and 'layer' in arch_parts and 'skip' in arch_parts:
        arch_name = "Three Layer Skip"
    elif 'four' in arch_parts and 'layer' in arch_parts and 'integrating' in arch_parts:
        arch_name = "Four Layer Integrating"
    elif 'four' in arch_parts and 'layer' in arch_parts and 'sequential' in arch_parts:
        arch_name = "Four Layer Sequential"
    else:
        arch_name = subfolder.replace('_', ' ').title()
    
    # Select same runs as gradient evolution (one from each LR pattern if possible)
    pattern_match = df['run_name'].str.extract(r'_(low|high)_(balanced|uniform|extreme)')
    df['lr_pattern'] = pattern_match.apply(lambda x: f"{x[0]}_{x[1]}" if pd.notna(x[0]) and pd.notna(x[1]) else 'unknown', axis=1)
    
    selected_runs = []
    for pattern in df['lr_pattern'].unique():
        pattern_runs = df[df['lr_pattern'] == pattern]
        if len(pattern_runs) > 0 and len(selected_runs) < n_runs:
            selected_runs.append(pattern_runs.iloc[0])
    
    if len(selected_runs) < n_runs:
        remaining = df[~df.index.isin([r.name for r in selected_runs])]
        if len(remaining) > 0:
            additional = remaining.sample(min(n_runs - len(selected_runs), len(remaining)), random_state=42)
            selected_runs.extend([additional.iloc[i] for i in range(len(additional))])
    
    selected_runs = selected_runs[:n_runs]
    
    # Get activation points from first run
    first_tracking_data = None
    for row in selected_runs:
        tracking_data = load_tracking_data(subfolder, row['run_name'])
        if tracking_data:
            first_tracking_data = tracking_data
            break
    
    if not first_tracking_data:
        print("No tracking data found for selected runs")
        return
    
    activation_points = list(first_tracking_data['activations'].keys())
    steps = first_tracking_data['steps']
    
    # Create plot: 3 rows (activation points) x 2 columns (norm/mean, sparsity/std)
    # Each column has n_runs subplots side-by-side
    fig, axes = plt.subplots(len(activation_points), 2, figsize=(4 * n_runs, 4 * len(activation_points)), sharex=True)
    if len(activation_points) == 1:
        axes = axes.reshape(1, -1)
    
    # Extract short titles for runs
    import re
    
    for point_idx, point_name in enumerate(activation_points):
        # Left column: Norm and Mean
        for run_idx, row in enumerate(selected_runs):
            run_name = row['run_name']
            tracking_data = load_tracking_data(subfolder, run_name)
            
            if not tracking_data or point_name not in tracking_data['activations']:
                continue
            
            ax = axes[point_idx, 0]
            # Create subplot for this run
            if run_idx == 0:
                # First run uses the main axes
                main_ax = ax
            else:
                # Other runs need to be on the same axes or we need a different approach
                # Actually, we need separate subplots for each run
                pass
        
        # Actually, let me restructure this - we need n_runs columns for each metric
        # So: len(activation_points) rows x (n_runs * 2) columns
        # Or better: len(activation_points) rows x 2 "groups" of n_runs columns each
    
    # Let me restructure: create separate subplots for each run
    # Structure: len(activation_points) rows x (n_runs * 2) columns
    fig, axes = plt.subplots(len(activation_points), n_runs * 2, 
                            figsize=(3 * n_runs * 2, 4 * len(activation_points)), sharex=True)
    if len(activation_points) == 1:
        axes = axes.reshape(1, -1)
    
    for point_idx, point_name in enumerate(activation_points):
        for run_idx, row in enumerate(selected_runs):
            run_name = row['run_name']
            tracking_data = load_tracking_data(subfolder, run_name)
            
            if not tracking_data or point_name not in tracking_data['activations']:
                continue
            
            point_data = tracking_data['activations'][point_name]
            
            # Extract short title
            sizes_match = re.search(r'l(\d+)-(\d+)-(\d+)', run_name)
            lr_match = re.search(r'lr([\de-]+)-([\de-]+)-([\de-]+)', run_name)
            short_title = ""
            if sizes_match:
                sizes = '-'.join(sizes_match.groups())
                short_title = f"L:{sizes}"
            if lr_match:
                lrs = '-'.join([lr.replace('e-', 'e') for lr in lr_match.groups()])
                if short_title:
                    short_title += f"\nLR:{lrs}"
                else:
                    short_title = f"LR:{lrs}"
            if not short_title:
                short_title = run_name[:20]
            
            # Left subplot: Norm and Mean
            ax_left = axes[point_idx, run_idx * 2]
            if 'norm' in point_data:
                ax_left.plot(steps, point_data['norm'], label='Norm', linewidth=1.5, color='blue', alpha=0.8)
                ax_left.set_ylabel('Norm', color='blue', fontsize=9)
                ax_left.tick_params(axis='y', labelcolor='blue')
            if 'mean' in point_data:
                ax_left_twin = ax_left.twinx()
                ax_left_twin.plot(steps, point_data['mean'], label='Mean', linewidth=1.5, color='red', alpha=0.7)
                ax_left_twin.set_ylabel('Mean', color='red', fontsize=9)
                ax_left_twin.tick_params(axis='y', labelcolor='red')
            if run_idx == 0:
                ax_left.text(-0.15, 0.5, point_name, transform=ax_left.transAxes, 
                           rotation=90, ha='center', va='center', fontsize=10, fontweight='bold')
            ax_left.set_title(short_title, fontsize=7, fontweight='bold')
            ax_left.grid(True, alpha=0.3)
            if run_idx == 0 and point_idx == 0:
                lines1, labels1 = ax_left.get_legend_handles_labels()
                if 'mean' in point_data:
                    lines2, labels2 = ax_left_twin.get_legend_handles_labels()
                    ax_left.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='upper left')
            
            # Right subplot: Sparsity and Std
            ax_right = axes[point_idx, run_idx * 2 + 1]
            if 'sparsity' in point_data:
                ax_right.plot(steps, point_data['sparsity'], label='Sparsity', linewidth=1.5, color='orange', alpha=0.8)
                ax_right.set_ylabel('Sparsity', color='orange', fontsize=9)
                ax_right.tick_params(axis='y', labelcolor='orange')
            if 'std' in point_data:
                ax_right_twin = ax_right.twinx()
                ax_right_twin.plot(steps, point_data['std'], label='Std', linewidth=1.5, color='green', alpha=0.7)
                ax_right_twin.set_ylabel('Std', color='green', fontsize=9)
                ax_right_twin.tick_params(axis='y', labelcolor='green')
            ax_right.set_title(short_title, fontsize=7, fontweight='bold')
            ax_right.grid(True, alpha=0.3)
            if run_idx == 0 and point_idx == 0:
                if 'std' in point_data:
                    lines1, labels1 = ax_right.get_legend_handles_labels()
                    lines2, labels2 = ax_right_twin.get_legend_handles_labels()
                    ax_right.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='upper left')
    
    # Set x-labels only on bottom row
    for run_idx in range(n_runs):
        axes[-1, run_idx * 2].set_xlabel('Step', fontsize=10)
        axes[-1, run_idx * 2 + 1].set_xlabel('Step', fontsize=10)
    
    plt.suptitle(f'{arch_name}: Activation Evolution - Multiple Runs', fontsize=16, fontweight='bold', y=0.995)
    plt.subplots_adjust(top=0.93)
    plt.tight_layout()
    output_path = Path(output_folder) / 'activation_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_example_runs(df, subfolder, output_folder, n_examples=6):
    """Plot detailed evolution for a few example runs"""
    # Select diverse examples
    examples = df.sample(min(n_examples, len(df)), random_state=42)
    
    for idx, row in examples.iterrows():
        run_name = row['run_name']
        tracking_data = load_tracking_data(subfolder, run_name)
        
        if tracking_data:
            # Correlation
            corr_path = Path(output_folder) / f'activation_gradient_correlation_{run_name}.png'
            plot_activation_gradient_correlation(tracking_data, corr_path, run_name)
            
            # Note: Individual activation and gradient evolution plots removed - see combined plots


def main():
    # Parse command line arguments
    subfolder = ""
    
    if '--subfolder' in sys.argv:
        idx = sys.argv.index('--subfolder')
        if idx + 1 < len(sys.argv):
            subfolder = sys.argv[idx + 1]
    
    if not subfolder:
        print("Error: --subfolder is required (e.g., --subfolder three_layer_skip_with_50_100_150)")
        return
    
    # Load analysis CSV (matching outputs structure)
    input_folder = Path('results') / 'activation_gradient_analysis_first_epoch' / subfolder
    df = load_analysis_csv(input_folder)
    
    if df is None:
        return
    
    # Create output folder for plots
    output_folder = input_folder / 'plots'
    output_folder.mkdir(exist_ok=True)
    
    print(f"Saving plots to: {output_folder}")
    
    # Create summary comparison plots
    print("\nCreating summary comparison plots...")
    plot_summary_comparisons(df, output_folder, subfolder)
    
    # Create combined gradient evolution plot
    print("\nCreating combined gradient evolution plot...")
    plot_combined_gradient_evolution(df, subfolder, output_folder, n_runs=6)
    
    # Create combined activation evolution plot
    print("\nCreating combined activation evolution plot...")
    plot_combined_activation_evolution(df, subfolder, output_folder, n_runs=6)
    
    # Create combined meaningful correlations plot
    print("\nCreating combined meaningful correlations plot...")
    plot_combined_meaningful_correlations(df, subfolder, output_folder, n_runs=6)
    
    # Create combined correlations over time plot
    print("\nCreating combined correlations over time plot...")
    plot_combined_correlations_over_time(df, subfolder, output_folder, n_runs=6, window_size=50)
    
    # Create example run plots (correlation only)
    print("\nCreating example run plots...")
    plot_example_runs(df, subfolder, output_folder, n_examples=6)
    
    print(f"\n✓ All plots saved to: {output_folder}")


if __name__ == '__main__':
    main()

