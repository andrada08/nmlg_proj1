#!/usr/bin/env python3
import os
import json
import pandas as pd
import glob
import sys
from pathlib import Path
from gradient_analysis import compute_gradient_metrics

def load_config(config_path):
    """Load config from a config file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_metrics(metrics_path):
    """Load gradient metrics from a metrics file"""
    with open(metrics_path, 'r') as f:
        return json.load(f)

def extract_hyperparams(config):
    """Extract key hyperparameters from config"""
    return {
        'n_epochs': config.get('n_epochs', 'N/A'),
        'batch_size': config.get('batch_size', 'N/A'),
        'optimizer': config.get('optimizer', 'N/A'),
        'activation': config.get('activation', 'N/A'),
        'architecture': config.get('architecture', 'unknown'),
        'input_size': config.get('input_size', 'N/A'),
        'hidden_sizes': str(config.get('hidden_sizes', 'N/A')),
        'output_size': config.get('output_size', 'N/A'),
        'ln_rate': config.get('ln_rate', 'N/A'),
        'layer_lns': str(config.get('layer_lns', 'N/A'))
    }

def main():
    # Parse command line arguments
    input_folder = ""
    output_subfolder = ""  # For organizing sweeps in outputs/
    
    if '--input-folder' in sys.argv:
        idx = sys.argv.index('--input-folder')
        if idx + 1 < len(sys.argv):
            input_folder = sys.argv[idx + 1]
    
    if '--output-subfolder' in sys.argv:
        idx = sys.argv.index('--output-subfolder')
        if idx + 1 < len(sys.argv):
            output_subfolder = sys.argv[idx + 1]
    
    # Require one of input_folder or output_subfolder
    if not input_folder and not output_subfolder:
        print("Error: Provide either --input-folder outputs/{subfolder} or --output-subfolder {subfolder}.")
        return

    # Validate input_folder if provided
    if input_folder and not os.path.isdir(input_folder):
        print(f"Error: --input-folder path does not exist: {input_folder}")
        return

    # Determine analysis name from input folder or output subfolder
    if input_folder:
        # Extract subfolder name from input path (e.g., "outputs/test" -> "test")
        if input_folder.startswith('outputs/'):
            analysis_name = input_folder.split('/')[-1]
        else:
            analysis_name = os.path.basename(input_folder)
    elif output_subfolder:
        analysis_name = output_subfolder
    else:
        analysis_name = "analysis"  # Default fallback
    
    # Create results folder structure
    results_folder = "results"
    analysis_folder = os.path.join(results_folder, analysis_name)
    os.makedirs(analysis_folder, exist_ok=True)
    
    print(f"Saving analysis results to: {analysis_folder}")
    
    # Find all output directories
    if input_folder:
        search_path = os.path.join(input_folder, 'sweep_*')
        print(f"Looking for results in: {search_path}")
    elif output_subfolder:
        search_path = f'outputs/{output_subfolder}/sweep_*'
        print(f"Looking for results in: {search_path}")
    else:
        # Backward compatibility: check for subfolder as first argument
        if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
            input_folder = sys.argv[1]
            search_path = f'outputs/{input_folder}/sweep_*'
            print(f"Looking for results in: {search_path}")
        else:
            search_path = 'outputs/sweep_*'
            print(f"Looking for results in: {search_path}")
    
    output_dirs = glob.glob(search_path)
    
    if not output_dirs:
        print("No output directories found in outputs/")
        return
    
    print(f"Found {len(output_dirs)} output directories")
    
    # Collect data from each run
    results = []
    
    for output_dir in sorted(output_dirs):
        config_path = os.path.join(output_dir, 'config.json')
        training_history_path = os.path.join(output_dir, 'training_history.json')
        
        if not os.path.exists(config_path):
            print(f"Skipping {output_dir} - missing config file")
            continue
        
        if not os.path.exists(training_history_path):
            print(f"Skipping {output_dir} - missing training_history.json")
            continue
        
        try:
            # Load config and training history
            config = load_config(config_path)
            with open(training_history_path, 'r') as f:
                training_history = json.load(f)
            
            # Extract hyperparameters
            hyperparams = extract_hyperparams(config)
            
            # Compute gradient metrics from training history (pass config for per-parameter analysis)
            metrics = compute_gradient_metrics(training_history, config=config)
            
            # Create row for this run
            row = {
                'run_name': os.path.basename(output_dir),
                **hyperparams,
                **metrics
            }
            
            results.append(row)
            print(f"Loaded {output_dir}")
            
        except Exception as e:
            print(f"Error loading {output_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("No valid results found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Get only the pattern columns (metrics that end with '_pattern')
    pattern_columns = [col for col in df.columns if col.endswith('_pattern')]
    
    # Sort by run name for consistent ordering
    df = df.sort_values('run_name')
    
    # Save to CSV
    output_file = os.path.join(analysis_folder, 'gradient_analysis.csv')
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total runs analyzed: {len(df)}")
    print(f"  Total patterns available: {len(pattern_columns)}")
    
    # Show pattern frequency
    print(f"\nPattern frequency (how many runs show each pattern):")
    pattern_counts = df[pattern_columns].sum().sort_values(ascending=False)
    for pattern, count in pattern_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {pattern}: {count}/{len(df)} runs ({percentage:.1f}%)")
        
        # Show alignment breakdown for patterns that have runs
        if count > 0:
            pattern_runs = df[df[pattern] == 1]
            alignment_col = f'{pattern.replace("_pattern", "")}_pattern_boost_pattern_aligned'
            if alignment_col in df.columns:
                aligned = pattern_runs[alignment_col].sum()
                not_aligned = len(pattern_runs) - aligned
                aligned_pct = (aligned / count * 100) if count > 0 else 0
                print(f"    └─ Aligned: {aligned}/{count} ({aligned_pct:.1f}%), Not aligned: {not_aligned}/{count} ({100-aligned_pct:.1f}%)")
    
    # Show runs with most patterns
    df['pattern_count'] = df[pattern_columns].sum(axis=1)
    df_sorted = df.sort_values('pattern_count', ascending=False)
    
    print(f"\nRuns with most patterns detected:")
    top_runs = df_sorted.head()[['run_name', 'pattern_count', 'final_test_accuracy']]
    for _, row in top_runs.iterrows():
        print(f"  {row['run_name']}: {row['pattern_count']} patterns, {row['final_test_accuracy']:.2f}% accuracy")
    
    print(f"\nTo create visualizations, run:")
    print(f"  python3 plot_results.py --pattern-analysis --input-folder {analysis_folder}")

if __name__ == '__main__':
    main()
