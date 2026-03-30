import numpy as np
import torch
import torch.optim as optim 
import os
import sys
import json
import shutil

from nets import build_model
from train import train_with_gradient_tracking
from load_data import load_data

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
architecture = cfg.get('architecture', 'three_layer_skip')
layer_types = cfg.get('layer_types', None)

# define output folder: use config filename as folder name
base_tag = os.path.splitext(os.path.basename(config_path))[0]

# Get output subfolder from environment variable (set by run_sweep.py)
output_subfolder = os.environ.get('OUTPUT_SUBFOLDER', '')
if output_subfolder:
    save_dir = os.path.join(os.getcwd(), 'outputs', output_subfolder, base_tag)
else:
    save_dir = os.path.join(os.getcwd(), 'outputs', base_tag)

os.makedirs(save_dir, exist_ok=True)
shutil.copy2(config_path, os.path.join(save_dir, 'config.json'))

# load and define
print("\nLoading MNIST data...")
train_loader, test_loader = load_data(batch_size)

model = build_model(
    architecture=architecture,
    input_size=input_size,
    hidden_sizes=tuple(hidden_sizes),
    output_size=output_size,
    activation=activation,
    layer_types=layer_types,
)

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

# Note: gradient_metrics removed - now computed post-hoc in analyze_results.py
# Only saving training history which contains all gradient data

# Save full training history for analysis and plot regeneration
history_file = os.path.join(save_dir, 'training_history.json')
with open(history_file, 'w') as f:
    json.dump(history, f, indent=2)
print(f"Training history saved to: {history_file}")

# Print final accuracy
final_test_acc = history['test_accuracy'][-1] if history['test_accuracy'] else 0.0
final_train_acc = history['accuracy'][-1] if history['accuracy'] else 0.0
print(f"\nTraining complete!")
print(f"Final test accuracy: {final_test_acc:.2f}%")
print(f"Final train accuracy: {final_train_acc:.2f}%")