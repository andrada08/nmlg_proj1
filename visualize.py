import matplotlib.pyplot as plt
import os

def should_show_plots():
    """Check if plots should be displayed based on matplotlib backend"""
    import matplotlib
    return matplotlib.get_backend() != 'Agg'

def visualize_gradients(history, layer_names, layer_info=None, save_dir=None):
    epochs = history['gradients']['epoch']
    plt.figure(figsize=(10, 6))
    for layer_name in layer_names:
        if layer_info and layer_name in layer_info:
            # Create descriptive label with size and learning rate
            info = layer_info[layer_name]
            label = f"{layer_name} (size={info['size']}, lr={info['lr']:.2e})"
        else:
            label = layer_name
        plt.plot(epochs, history['gradients'][layer_name], label=label)
    plt.title('Gradient Norms by Layer')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'gradients.png'), dpi=150)
    
    if should_show_plots():
        plt.show()
    else:
        plt.close()  # Clean up when not showing

def visualize_loss_and_accuracy(history, save_dir=None):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(history['loss'])
    axs[0].set_title('Training Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[1].plot(history['accuracy'])
    axs[1].set_title('Training Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, 'loss_accuracy.png'), dpi=150)
    
    if should_show_plots():
        plt.show()
    else:
        plt.close()  # Clean up when not showing
