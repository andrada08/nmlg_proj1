import numpy as np
import torch
import torch.optim as optim 
import os
import sys
import json
import shutil

from nets import Net
from train import train_with_gradient_tracking
from load_data import load_data
from visualize import visualize_gradients, visualize_loss_and_accuracy

np.random.seed(0)
torch.manual_seed(0)

# CUDA (cluster), Apple Silicon MPS (macOS), else CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load config from CLI (python3 main.py path/to/config.json)
if len(sys.argv) < 2:
    raise ValueError("Please provide a config file path: python3 main.py configs/your_config.json")
config_path = sys.argv[1]
with open(config_path, 'r') as f:
    cfg = json.load(f)

# Unpack
n_epochs = cfg['n_epochs']
batch_size = cfg['batch_size']
ln_rate = cfg.get('ln_rate', None)
opt = getattr(optim, cfg["optimizer"])
input_size = cfg['input_size']
hidden_sizes = cfg['hidden_sizes']
output_size = cfg['output_size']
layer_lns = cfg.get('layer_lns')
activation = cfg['activation']

# define output folder: use config filename as folder name
base_tag = os.path.splitext(os.path.basename(config_path))[0]
save_dir = os.path.join(os.getcwd(), 'outputs', base_tag)
if os.path.exists(save_dir):
    i = 2
    while os.path.exists(f"{save_dir}_{i}"):
        i += 1
    save_dir = f"{save_dir}_{i}"

os.makedirs(save_dir, exist_ok=True)
shutil.copy2(config_path, os.path.join(save_dir, 'config.json'))

# load and define
print("\nLoading MNIST data...")
train_loader, test_loader = load_data(batch_size)
    
model = Net(input_size, hidden_sizes, output_size, activation=activation)

# train
print(f"\nTraining model with gradient tracking...")
history = train_with_gradient_tracking(
    model, 
    train_loader,
    test_loader,
    epochs=n_epochs,
    optimizer=opt,
    ln_rate=ln_rate,
    layer_lns=layer_lns,
    device = device
)

# plot
print(f"\nPlotting results...")
visualize_loss_and_accuracy(history, save_dir=save_dir)

# Create layer info for gradient plot labels
layer_info = {}
hidden_sizes = cfg['hidden_sizes']
layer_sizes = [input_size] + hidden_sizes + [hidden_sizes[0] + hidden_sizes[1]]  # [784, h1, h2, h1+h2]
layer_names = model.get_layer_names()

for i, layer_name in enumerate(layer_names):
    layer_info[layer_name] = {
        'size': layer_sizes[i],
        'lr': layer_lns[layer_name] if layer_lns else ln_rate
    }

visualize_gradients(history, model.get_layer_names(), layer_info=layer_info, save_dir=save_dir)

# Save gradient metrics to file (include final accuracy)
metrics_data = history['gradient_metrics'].copy()
metrics_data['final_test_accuracy'] = history['test_accuracy'][-1] if history['test_accuracy'] else 0.0
metrics_data['final_train_accuracy'] = history['accuracy'][-1] if history['accuracy'] else 0.0

metrics_file = os.path.join(save_dir, 'gradient_metrics.json')
with open(metrics_file, 'w') as f:
    json.dump(metrics_data, f, indent=2)
print(f"\nGradient metrics saved to: {metrics_file}")

# Also print a summary
print(f"\nGradient Metrics Summary:")
print(f"Patterns detected: {sum(history['gradient_metrics'].values())} out of {len(history['gradient_metrics'])}")
print(f"Large drops: L1={history['gradient_metrics']['layer1_large_drop']}, L2={history['gradient_metrics']['layer2_large_drop']}, L3={history['gradient_metrics']['layer3_large_drop']}")
print(f"Switches: 1-2={history['gradient_metrics']['switches_12']}, 1-3={history['gradient_metrics']['switches_13']}, 2-3={history['gradient_metrics']['switches_23']}")