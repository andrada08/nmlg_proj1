#!/usr/bin/env python3
import os
import subprocess
import glob
import json
from concurrent.futures import ProcessPoolExecutor
import argparse

def run_single_config(config_path, no_plots=False):
    """Run a single config file and return results"""
    try:
        print(f"Starting {config_path}")
        env = os.environ.copy()
        
        if no_plots:
            # Set environment variable to disable plotting
            env['MPLBACKEND'] = 'Agg'  # Use non-interactive backend
        
        result = subprocess.run(
            ['python3', 'main.py', config_path],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout per run
            env=env
        )
        
        if result.returncode == 0:
            print(f"✅ Completed {config_path}")
            return {'config': config_path, 'status': 'success', 'output': result.stdout}
        else:
            print(f"❌ Failed {config_path}: {result.stderr}")
            return {'config': config_path, 'status': 'failed', 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout {config_path}")
        return {'config': config_path, 'status': 'timeout', 'error': '30min timeout'}
    except Exception as e:
        print(f"💥 Error {config_path}: {str(e)}")
        return {'config': config_path, 'status': 'error', 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='Run parameter sweep in parallel')
    parser.add_argument('--configs', default='configs/sweep_*.json', 
                       help='Glob pattern for config files (default: configs/sweep_*.json)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of parallel workers (default: 4)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot popups (plots still saved to files)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    
    args = parser.parse_args()
    
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
    print(f"\n🚀 Starting parallel execution...")
    results = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(run_single_config, config, args.no_plots): config 
            for config in config_files
        }
        
        # Collect results as they complete
        for future in future_to_config:
            result = future.result()
            results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    print(f"\n📊 Summary:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📁 Results saved to: outputs/")
    
    # Save results summary
    summary_file = 'sweep_results.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  📄 Detailed results: {summary_file}")

if __name__ == '__main__':
    main()
