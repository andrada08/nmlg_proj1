import matplotlib.pyplot as plt
import os

def visualize_gradients(history, layer_names, save_dir=None):
    epochs = history['gradients']['epoch']
    plt.figure(figsize=(10, 6))
    for layer_name in layer_names:
        plt.plot(epochs, history['gradients'][layer_name], label=layer_name)
    plt.title('Gradient Norms by Layer')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'gradients.png'), dpi=150)
    plt.show()

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
    plt.show()
