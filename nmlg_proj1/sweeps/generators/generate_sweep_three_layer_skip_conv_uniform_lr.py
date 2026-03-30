import os
import json
import sys
from itertools import product

# Parse command line arguments
subfolder_name = "three_layer_skip_conv_uniform_lr"  # Default subfolder name
if "--subfolder" in sys.argv:
    idx = sys.argv.index("--subfolder")
    if idx + 1 < len(sys.argv):
        subfolder_name = sys.argv[idx + 1]

BASE = os.path.join(os.getcwd(), "configs", subfolder_name)
os.makedirs(BASE, exist_ok=True)
print(f"Generating configs in: {BASE}")

# Layer size options - generate all possible combinations
# For conv layers, these represent number of output channels
layer_size_options = [32, 64, 128]
# Generate all combinations of (layer1, layer2, layer3)
layer_sizes = list(product(layer_size_options, layer_size_options, layer_size_options))

# Layer type combinations
# Options: all linear, all conv, layer1 conv + layer2 linear, layer1 linear + layer2 conv
layer_type_combinations = {
    "all_linear": {"layer1": "linear", "layer2": "linear"},
    "all_conv": {"layer1": "conv", "layer2": "conv"},
    "conv_linear": {"layer1": "conv", "layer2": "linear"},
    "linear_conv": {"layer1": "linear", "layer2": "conv"},
}

# Use a single uniform learning rate pattern to focus on layer type comparisons
# Using high_uniform (1e-3) to match old naming convention
lr_pattern = {"layer1": 1e-3, "layer2": 1e-3, "layer3": 1e-3}
lr_pattern_name = "high_uniform"  # Match old naming convention

template = {
    "optimizer": "Adam",
    "ln_rate": 1e-3,  # Default learning rate for layers not specified in layer_lns
    "input_size": 784,
    "output_size": 10,
    "n_epochs": 20,
    "batch_size": 128,
    "activation": "relu",
    "architecture": "three_layer_skip",
}


def write_config(name: str, cfg: dict):
    path = os.path.join(BASE, f"{name}.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Wrote", path)


# Generate configs for all combinations
for (l1, l2, l3), (type_name, layer_types) in product(
    layer_sizes, layer_type_combinations.items()
):
    # Create meaningful filename that encodes architecture
    type_abbrev = {
        "all_linear": "ll",
        "all_conv": "cc",
        "conv_linear": "cl",
        "linear_conv": "lc",
    }[type_name]

    filename = (
        f"sweep_three_layer_skip_{type_abbrev}_{lr_pattern_name}_"
        f"l{l1}-{l2}-{l3}_"
        f"lr{lr_pattern['layer1']:.0e}-{lr_pattern['layer2']:.0e}-{lr_pattern['layer3']:.0e}"
    )

    cfg = template | {
        "hidden_sizes": [l1, l2, l3],
        "layer_lns": lr_pattern,
        "layer_types": layer_types,
    }
    write_config(filename, cfg)

total_configs = len(layer_sizes) * len(layer_type_combinations)
print(f"\nGenerated {total_configs} config files in: {BASE}")
print("\nTo run these sweeps with the same subfolder:")
print(
    f"  python3 -m nmlg_proj1.sweeps.run_sweep --subfolder {subfolder_name} --output-subfolder {subfolder_name}"
)

