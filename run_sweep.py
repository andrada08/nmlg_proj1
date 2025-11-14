#!/usr/bin/env python3
import os
import subprocess
import glob
import json
from concurrent.futures import ProcessPoolExecutor
import argparse
from pathlib import Path

def run_single_config(config_path, output_subfolder):
    """Run a single config file and return results"""
    try:
        base_tag = Path(config_path).stem
        output_dir = Path('outputs') / output_subfolder / base_tag
        history_file = output_dir / 'training_history.json'
        if history_file.exists():
            print(f"Skipping {config_path} - training_history.json already exists")
            return {'config': config_path, 'status': 'skipped'}

        print(f"Starting {config_path}")
        env = os.environ.copy()
        
        # Set output subfolder environment variable
        env['OUTPUT_SUBFOLDER'] = output_subfolder
        
        result = subprocess.run(
            ['python3', 'main.py', config_path],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout per run
            env=env
        )
        
        if result.returncode == 0:
            print(f"Completed {config_path}")
            return {'config': config_path, 'status': 'success', 'output': result.stdout}
        else:
            print(f"Failed {config_path}: {result.stderr}")
            return {'config': config_path, 'status': 'failed', 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        print(f"Timeout {config_path}")
        return {'config': config_path, 'status': 'timeout', 'error': '30min timeout'}
    except Exception as e:
        print(f"Error {config_path}: {str(e)}")
        return {'config': config_path, 'status': 'error', 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Run parameter sweep in parallel')
    parser.add_argument('--configs', 
                       help='Glob pattern for config files (default: configs/{subfolder}/sweep_*.json)')
    parser.add_argument('--subfolder', default='sweeps',
                       help='Subfolder name in configs/ directory (default: sweeps)')
    parser.add_argument('--output-subfolder',
                       help='Subfolder name in outputs/ directory (defaults to --subfolder value)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of parallel workers (default: 4)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    # Set default configs pattern based on subfolder
    if args.configs is None:
        args.configs = f'configs/{args.subfolder}/sweep_*.json'
    
    # Set default output subfolder
    if args.output_subfolder is None:
        args.output_subfolder = args.subfolder
    
    # Find all config files
    config_files = sorted(glob.glob(args.configs))
    
    if not config_files:
        print(f"No config files found matching: {args.configs}")
        return
    
    print(f"Found {len(config_files)} config files")
    print(f"Will run with {args.max_workers} parallel workers")
    
    if args.dry_run:
        print("\nConfig files that would be run:")
        for config in config_files:
            print(f"  - {config}")
        return
    
    # Run configs in parallel
    print(f"\nStarting parallel execution...")
    print(f"Outputs will be saved to: outputs/{args.output_subfolder}/")
    results = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(run_single_config, config, args.output_subfolder): config 
            for config in config_files
        }
        
        # Collect results as they complete
        for future in future_to_config:
            result = future.result()
            results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    failed = len(results) - successful - skipped
    
    print(f"\nSummary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Results saved to: outputs/{args.output_subfolder}/")
    
    # Save results summary
    os.makedirs(f'outputs/{args.output_subfolder}', exist_ok=True)
    summary_file = f'outputs/{args.output_subfolder}/sweep_results.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Detailed results: {summary_file}")
    
    print(f"\nTo analyze results:")
    print(f"  python analyze_results.py --output-subfolder {args.output_subfolder}")

if __name__ == '__main__':
    main()
