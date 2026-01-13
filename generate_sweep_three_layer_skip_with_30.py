import os
import json
import sys
from itertools import product

# Parse command line arguments
subfolder_name = "three_layer_skip_with_30"  # Default subfolder name
if '--subfolder' in sys.argv:
    idx = sys.argv.index('--subfolder')
    if idx + 1 < len(sys.argv):
        subfolder_name = sys.argv[idx + 1]

BASE = os.path.join(os.getcwd(), 'configs', subfolder_name)
os.makedirs(BASE, exist_ok=True)
print(f"Generating configs in: {BASE}")

# Layer size options - use 30 instead of 150 to match the pattern of existing sweeps
layer_size_options = [30, 50, 100]
# Generate all combinations of (layer1, layer2, layer3)
all_layer_sizes = list(product(layer_size_options, layer_size_options, layer_size_options))

# Filter to only include combinations where at least one layer is 30 (new combinations)
layer_sizes = [(l1, l2, l3) for (l1, l2, l3) in all_layer_sizes if 30 in (l1, l2, l3)]

print(f"Generated {len(layer_sizes)} layer size combinations that include 30")

# Learning rate patterns - only for main layers
# Two layer1 values, each with balanced, extreme, and uniform options
lr_patterns = {
    # Layer1 = 1e-4 (low): balanced, extreme, and uniform
    'low_balanced': {'layer1': 1e-4, 'layer2': 2e-4, 'layer3': 4e-4},
    'low_extreme': {'layer1': 1e-4, 'layer2': 5e-4, 'layer3': 1.5e-3},
    'low_uniform': {'layer1': 1e-4, 'layer2': 1e-4, 'layer3': 1e-4},
    # Layer1 = 1e-3 (high): balanced, extreme, and uniform
    'high_balanced': {'layer1': 1e-3, 'layer2': 2e-3, 'layer3': 4e-3},
    'high_extreme': {'layer1': 1e-3, 'layer2': 5e-3, 'layer3': 1.5e-2},
    'high_uniform': {'layer1': 1e-3, 'layer2': 1e-3, 'layer3': 1e-3},
}

template = {
    'optimizer': 'Adam',
    'ln_rate': 1e-3,  # Default learning rate for layers not specified in layer_lns
    'input_size': 784,
    'output_size': 10,
    'n_epochs': 20,
    'batch_size': 128,
    'activation': 'relu',
    'architecture': 'three_layer_skip',
}

def write_config(name: str, cfg: dict):
    path = os.path.join(BASE, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print('Wrote', path)

for (l1, l2, l3), (pattern_name, lr_dict) in product(layer_sizes, lr_patterns.items()):

    # Create meaningful filename that encodes architecture
    filename = (
        f'sweep_three_layer_skip_{pattern_name}_'
        f'l{l1}-{l2}-{l3}_'
        f"lr{lr_dict['layer1']:.0e}-{lr_dict['layer2']:.0e}-{lr_dict['layer3']:.0e}"
    )
    
    cfg = template | {
        'hidden_sizes': [l1, l2, l3],
        'layer_lns': lr_dict,
    }
    write_config(filename, cfg)

print(f"\nGenerated {len(layer_sizes) * len(lr_patterns)} config files in: {BASE}")
print(f"\nTo run these sweeps with the same subfolder:")
print(f"  python run_sweep.py --subfolder {subfolder_name} --output-subfolder {subfolder_name}")

