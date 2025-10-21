import os
import json
from itertools import product

BASE = os.path.join(os.getcwd(), 'configs')
os.makedirs(BASE, exist_ok=True)

# Layer size combinations (layer1, layer2) - layer3 will be layer1 + layer2
layer_sizes = [
    (250, 150),  # layer3 = 400
    (250, 100),  # layer3 = 350
    (150, 100),  # layer3 = 250
    (150, 250),  # layer3 = 400
    (100, 150),  # layer3 = 250
    (100, 250),  # layer3 = 350
]

# Learning rate patterns
lr_patterns = {
    'A1': {'layer1': 1e-4, 'layer2': 2e-4, 'layer3': 4e-4},
    'A2': {'layer1': 1e-4, 'layer2': 3e-4, 'layer3': 8e-4},
    'A3': {'layer1': 1e-4, 'layer2': 5e-4, 'layer3': 1.5e-3},
    'B1': {'layer1': 3e-4, 'layer2': 6e-4, 'layer3': 1.2e-3},
    'B2': {'layer1': 3e-4, 'layer2': 9e-4, 'layer3': 2.4e-3},
    'B3': {'layer1': 3e-4, 'layer2': 1.5e-3, 'layer3': 4.5e-3},
    'C1': {'layer1': 1e-3, 'layer2': 2e-3, 'layer3': 4e-3},
    'C2': {'layer1': 1e-3, 'layer2': 3e-3, 'layer3': 8e-3},
    'C3': {'layer1': 1e-3, 'layer2': 5e-3, 'layer3': 1.5e-2},
}

template = {
    'optimizer': 'Adam',
    'input_size': 784,
    'output_size': 10,
    'n_epochs': 20,
    'batch_size': 128,
    'activation': 'relu',
}

def write_config(name: str, cfg: dict):
    path = os.path.join(BASE, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print('Wrote', path)

for (l1, l2), (pattern_name, lr_dict) in product(layer_sizes, lr_patterns.items()):
    l3 = l1 + l2  # layer3 size
    
    # Create meaningful filename
    filename = f'sweep_{pattern_name}_l{l1}-{l2}-{l3}_lr{lr_dict["layer1"]:.0e}-{lr_dict["layer2"]:.0e}-{lr_dict["layer3"]:.0e}'
    
    cfg = template | {
        'hidden_sizes': [l1, l2],
        'layer_lns': lr_dict,
    }
    write_config(filename, cfg)

print(f"\nGenerated {len(layer_sizes) * len(lr_patterns)} config files")
