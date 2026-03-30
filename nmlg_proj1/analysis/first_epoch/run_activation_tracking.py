#!/usr/bin/env python3
"""
Run activation and gradient tracking for the first epoch in parallel across multiple folders.

Usage:
    python -m nmlg_proj1.analysis.first_epoch.run_activation_tracking --subfolder three_layer_skip_with_50_100_150 --max-workers 6
"""

import argparse
import glob
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def run_single_tracking(output_dir):
    """Run tracking for a single output directory and return results"""
    try:
        output_dir = Path(output_dir)

        # Check if config exists
        config_path = output_dir / "config.json"
        if not config_path.exists():
            return {
                "output_dir": str(output_dir),
                "status": "no_config",
                "error": "config.json not found",
            }

        # Determine expected output location
        output_dir_str = str(output_dir)
        if "gradients_across_training" in output_dir_str:
            parts = output_dir_str.split("gradients_across_training")
            if len(parts) == 2:
                save_base = Path("outputs") / "activation_gradient_analysis_first_epoch"
                relative_path = parts[1].lstrip("/")
                save_dir = save_base / relative_path
            else:
                # Fallback extraction
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
                        save_dir = None
                else:
                    save_dir = None
        else:
            # Try to infer structure
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
                    save_dir = None
            else:
                save_dir = None

        # Check if already completed
        if save_dir and save_dir.exists():
            output_file = save_dir / "activation_gradient_analysis_first_epoch.json"
            if output_file.exists():
                return {
                    "output_dir": str(output_dir),
                    "status": "skipped",
                    "reason": "already exists",
                }

        print(f"Starting {output_dir.name}")

        result = subprocess.run(
            [
                "python3",
                "-m",
                "nmlg_proj1.analysis.first_epoch.track_activations_gradients_first_epoch",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per run
            cwd=os.getcwd(),
        )

        if result.returncode == 0:
            print(f"Completed {output_dir.name}")
            return {"output_dir": str(output_dir), "status": "success"}
        else:
            # Capture both stdout and stderr for debugging
            error_msg = result.stderr if result.stderr else result.stdout
            if not error_msg:
                error_msg = f"Exit code: {result.returncode}, no error message"
            print(f"Failed {output_dir.name}: {error_msg[:200]}")
            return {
                "output_dir": str(output_dir),
                "status": "failed",
                "error": error_msg[:500],
                "exit_code": result.returncode,
            }

    except subprocess.TimeoutExpired:
        print(f"Timeout {output_dir.name}")
        return {"output_dir": str(output_dir), "status": "timeout", "error": "5min timeout"}
    except Exception as e:
        print(f"Error {output_dir.name}: {str(e)}")
        return {"output_dir": str(output_dir), "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run activation/gradient tracking in parallel")
    parser.add_argument(
        "--subfolder",
        required=True,
        help="Subfolder name in outputs/gradients_across_training/ (e.g., three_layer_skip_with_50_100_150)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: auto-detect based on device)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    args = parser.parse_args()

    # Auto-detect device and limit workers for MPS (Metal doesn't support concurrent access)
    if args.max_workers is None:
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                args.max_workers = 2  # MPS can handle 1-2 concurrent processes
                print(
                    f"Detected MPS device - limiting to {args.max_workers} workers for stability"
                )
            elif torch.cuda.is_available():
                args.max_workers = 6  # CUDA can handle more
            else:
                args.max_workers = 4  # CPU fallback
        except ImportError:
            args.max_workers = 2  # Conservative default

    # Find all output directories
    base_dir = Path("outputs") / "gradients_across_training" / args.subfolder
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return

    # Find all sweep directories (folders starting with 'sweep_')
    output_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("sweep_")])

    if not output_dirs:
        print(f"No sweep directories found in: {base_dir}")
        return

    print(f"Found {len(output_dirs)} directories to process")
    print(f"Will run with {args.max_workers} parallel workers")

    if args.dry_run:
        print("\nDirectories that would be processed:")
        for d in output_dirs[:10]:
            print(f"  - {d.name}")
        if len(output_dirs) > 10:
            print(f"  ... and {len(output_dirs) - 10} more")
        return

    # Run tracking in parallel
    print("\nStarting parallel execution...")
    print(
        f"Results will be saved to: outputs/activation_gradient_analysis_first_epoch/{args.subfolder}/"
    )
    results = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all jobs
        future_to_dir = {
            executor.submit(run_single_tracking, str(output_dir)): output_dir
            for output_dir in output_dirs
        }

        # Collect results as they complete
        completed = 0
        for future in future_to_dir:
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 10 == 0:
                print(f"Progress: {completed}/{len(output_dirs)} completed")

    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = len(results) - successful - skipped

    print("\nSummary:")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(
        f"  Results saved to: outputs/activation_gradient_analysis_first_epoch/{args.subfolder}/"
    )

    # Save results summary
    summary_dir = (
        Path("outputs") / "activation_gradient_analysis_first_epoch" / args.subfolder
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / "tracking_results_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Detailed results: {summary_file}")

    if failed > 0:
        print("\nFailed runs:")
        for r in results:
            if r["status"] != "success" and r["status"] != "skipped":
                print(f"  - {Path(r['output_dir']).name}: {r['status']}")


if __name__ == "__main__":
    main()

