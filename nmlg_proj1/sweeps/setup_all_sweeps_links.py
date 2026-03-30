#!/usr/bin/env python3
"""
Create symbolic links for all_sweeps folders to point to component folders.
This allows analyze_results.py to find all runs when analyzing combined sweeps.
"""

import os
from pathlib import Path


def setup_all_sweeps_links():
    """Create symbolic links for all_sweeps folders"""

    # Define the mappings: all_sweeps folder -> list of component folders
    mappings = {
        "three_layer_skip_all_sweeps": [
            "three_layer_skip_with_30",
            "three_layer_skip_with_50_100_150",
        ],
        "four_layer_integrating_all_sweeps": [
            "four_layer_integrating_with_30",
            "four_layer_integrating_with_50_100_150",
        ],
    }

    for all_sweeps_folder, component_folders in mappings.items():
        output_dir = Path("outputs") / all_sweeps_folder
        output_dir.mkdir(exist_ok=True)

        print(f"\nSetting up {all_sweeps_folder}...")
        print(f"  Output directory: {output_dir}")

        # Count existing links/files
        existing = list(output_dir.glob("*"))
        if existing:
            print(f"  Found {len(existing)} existing items in {output_dir}")
            # Remove broken symlinks or files (but keep valid symlinks)
            removed = 0
            for item in existing:
                if item.is_symlink():
                    # Check if symlink is broken
                    if not item.exists():
                        item.unlink()
                        removed += 1
                elif item.is_file():
                    # Remove any regular files (shouldn't be here)
                    item.unlink()
                    removed += 1
            if removed > 0:
                print(f"  Removed {removed} broken/invalid items")

        # Create symbolic links for all runs from component folders
        total_links = 0
        for component_folder in component_folders:
            component_path = Path("outputs") / component_folder
            if not component_path.exists():
                print(
                    f"  Warning: Component folder {component_path} does not exist, skipping"
                )
                continue

            # Find all sweep directories
            sweep_dirs = sorted(component_path.glob("sweep_*"))
            print(f"  Found {len(sweep_dirs)} runs in {component_folder}")

            for sweep_dir in sweep_dirs:
                link_path = output_dir / sweep_dir.name

                # Check if link already exists and is correct
                if link_path.exists():
                    if link_path.is_symlink():
                        # Check if it points to the right place
                        try:
                            target = os.readlink(link_path)
                            # Resolve to absolute paths for comparison
                            abs_target = os.path.abspath(os.path.join(output_dir, target))
                            abs_sweep = os.path.abspath(sweep_dir)
                            if abs_target == abs_sweep:
                                continue  # Already correctly linked
                            else:
                                # Wrong target, remove and recreate
                                link_path.unlink()
                        except (OSError, ValueError):
                            # Broken symlink, remove it
                            link_path.unlink()
                    else:
                        print(
                            f"    Warning: {link_path} exists but is not a symlink, skipping"
                        )
                        continue

                # Create relative symlink
                try:
                    relative_target = os.path.relpath(sweep_dir, output_dir)
                    link_path.symlink_to(relative_target)
                    total_links += 1
                except Exception as e:
                    print(f"    Error creating link for {sweep_dir.name}: {e}")

        print(f"  Created {total_links} symbolic links in {output_dir}")
        print(f"  Total items in {output_dir}: {len(list(output_dir.glob('*')))}")


if __name__ == "__main__":
    setup_all_sweeps_links()
    print("\n✓ Setup complete!")
    print("\nYou can now run:")
    print(
        "  python3 -m nmlg_proj1.analysis.analyze_results --output-subfolder three_layer_skip_all_sweeps"
    )
    print(
        "  python3 -m nmlg_proj1.analysis.analyze_results --output-subfolder four_layer_integrating_all_sweeps"
    )

