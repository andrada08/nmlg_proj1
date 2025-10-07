import numpy as np
import torch
from torch.optim import Adam


from nets import Net
from train import train_with_gradient_tracking
from load_data import load_data
from visualize import visualize_gradients, visualize_loss_and_accuracy
import os
from datetime import datetime

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

# set things
n_epochs = 20
batch_size = 128
ln_rate = 0.01
opt = Adam
input_size = 784
hidden_sizes = [128, 64]
output_size = 10
layer_lns = {'layer1': 0.01, 'layer2': 0.001, 'layer3': 0.001}

run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join(os.getcwd(), 'outputs', run_id)

# load and define
print("\nLoading MNIST data...")
train_loader, test_loader = load_data(batch_size)
    
model = Net(input_size, hidden_sizes, output_size)

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
    
print(history['gradients']['epoch'])
print([len(history['gradients'][n]) for n in model.get_layer_names()])

# plot
print(f"\nPlotting results...")
visualize_loss_and_accuracy(history, save_dir=save_dir)
visualize_gradients(history, model.get_layer_names(), save_dir=save_dir)