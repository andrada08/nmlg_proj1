import os
import json
import sys
from itertools import product

# Parse command line arguments
subfolder_name = "four_layer_integrating_all_sweeps"  # Default subfolder name
if '--subfolder' in sys.argv:
    idx = sys.argv.index('--subfolder')
    if idx + 1 < len(sys.argv):
        subfolder_name = sys.argv[idx + 1]

BASE = os.path.join(os.getcwd(), 'configs', subfolder_name)
os.makedirs(BASE, exist_ok=True)
print(f"Generating configs in: {BASE}")

# Layer size options - generate all possible combinations
layer_size_options = [50, 100, 150]
# Generate all combinations of (layer1, layer2, layer3, layer4)
layer_sizes = list(product(layer_size_options, repeat=4))

# Learning rate patterns - extend to four layers
lr_patterns = {
    'low_balanced': {'layer1': 1e-4, 'layer2': 2e-4, 'layer3': 4e-4, 'layer4': 8e-4},
    'low_extreme': {'layer1': 1e-4, 'layer2': 5e-4, 'layer3': 1.5e-3, 'layer4': 4.5e-3},
    'low_uniform': {'layer1': 1e-4, 'layer2': 1e-4, 'layer3': 1e-4, 'layer4': 1e-4},
    'high_balanced': {'layer1': 1e-3, 'layer2': 2e-3, 'layer3': 4e-3, 'layer4': 8e-3},
    'high_extreme': {'layer1': 1e-3, 'layer2': 5e-3, 'layer3': 1.5e-2, 'layer4': 4.5e-2},
    'high_uniform': {'layer1': 1e-3, 'layer2': 1e-3, 'layer3': 1e-3, 'layer4': 1e-3},
}

template = {
    'optimizer': 'Adam',
    'ln_rate': 1e-3,
    'input_size': 784,
    'output_size': 10,
    'n_epochs': 20,
    'batch_size': 128,
    'activation': 'relu',
    'architecture': 'four_layer_integrating',
}


def write_config(name: str, cfg: dict):
    path = os.path.join(BASE, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print('Wrote', path)


for (l1, l2, l3, l4), (pattern_name, lr_dict) in product(layer_sizes, lr_patterns.items()):
    filename = (
        f'sweep_four_layer_integrating_{pattern_name}_'
        f'l{l1}-{l2}-{l3}-{l4}_'
        f"lr{lr_dict['layer1']:.0e}-{lr_dict['layer2']:.0e}-{lr_dict['layer3']:.0e}-{lr_dict['layer4']:.0e}"
    )

    cfg = template | {
        'hidden_sizes': [l1, l2, l3, l4],
        'layer_lns': lr_dict,
    }
    write_config(filename, cfg)

print(f"\nGenerated {len(layer_sizes) * len(lr_patterns)} config files in: {BASE}")
print(f"\nTo run these sweeps with the same subfolder:")
print(f"  python run_sweep.py --subfolder {subfolder_name} --output-subfolder {subfolder_name}")

