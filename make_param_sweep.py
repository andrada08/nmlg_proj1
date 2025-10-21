import os
import json
from itertools import product

BASE = os.path.join(os.getcwd(), 'configs')
os.makedirs(BASE, exist_ok=True)

# Define sweep
n_epochs_list = [20]
batch_sizes = [64, 128]
ln_rates = [1e-5, 3e-5, 1e-4]
hidden_sizes_list = [[256, 256], [128, 128], [64, 64],  [256, 128], [128, 64], [256, 64]]
# layer_lns_options = [None, {'layer1': ln_rate,
#                             'layer2': ln_rate*5, 
#                             'layer3': ln_rate*20}]
activations = ["relu"]

template = {
    'optimizer': 'Adam',
    'input_size': 784,
    'output_size': 10,
}

def write_config(name: str, cfg: dict):
    path = os.path.join(BASE, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print('Wrote', path)

for n_epochs, batch_size, ln_rate, hidden_sizes, layer_lns, activation in product(
    n_epochs_list, batch_sizes, ln_rates, hidden_sizes_list, layer_lns_options, activations
):
    tag = 'groups' if layer_lns else 'single'
    name = f'ep{n_epochs}_bs{batch_size}_lr{ln_rate}_h{hidden_sizes[0]}-{hidden_sizes[1]}_{activation}_{tag}'
    cfg = template | {
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'ln_rate': ln_rate,
        'hidden_sizes': hidden_sizes,
        'layer_lns': layer_lns,
        'activation': activation,
    }
    write_config(name, cfg)


