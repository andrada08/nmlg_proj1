#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_gradients(history, layer_names, layer_info=None, save_dir=None, smoothing_window=3, log_scale=False):
    """Simplified gradient visualization that should work correctly"""
    epochs = history['gradients']['epoch']
    
    # Create two subplots: one for raw gradients, one for smoothed
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Raw gradients
    for layer_name in layer_names:
        if layer_info and layer_name in layer_info:
            info = layer_info[layer_name]
            label = f"{layer_name} (size={info['size']}, lr={info['lr']:.2e})"
        else:
            label = layer_name
        
        grads = history['gradients'][layer_name]
        
        # Apply log scale to raw data if requested
        if log_scale:
            grads = [np.log10(max(g, 1e-10)) for g in grads]
            ylabel = 'Log10(Gradient Norm)'
        else:
            ylabel = 'Gradient Norm'
            
        ax1.plot(epochs, grads, label=label, linewidth=2)
    
    ax1.set_title('Raw Gradient Norms by Layer')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(ylabel)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Smoothed gradients
    for layer_name in layer_names:
        if layer_info and layer_name in layer_info:
            info = layer_info[layer_name]
            label = f"{layer_name} (smoothed)"
        else:
            label = f"{layer_name} (smoothed)"
        
        grads = history['gradients'][layer_name]
        
        # Apply smoothing to raw data
        if len(grads) >= smoothing_window:
            smoothed_grads = []
            for i in range(len(grads)):
                start_idx = max(0, i - smoothing_window // 2)
                end_idx = min(len(grads), i + smoothing_window // 2 + 1)
                smoothed_grads.append(np.mean(grads[start_idx:end_idx]))
        else:
            smoothed_grads = grads
        
        # Apply log scale to smoothed data if requested
        if log_scale:
            smoothed_grads = [np.log10(max(g, 1e-10)) for g in smoothed_grads]
            
        ax2.plot(epochs, smoothed_grads, label=label, linewidth=2)
    
    ax2.set_title(f'Smoothed Gradient Norms (window={smoothing_window})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(ylabel)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'gradients.png'), dpi=150)

    # Check if we should show plots
    import matplotlib
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    else:
        plt.close()

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

def should_show_plots():
    """Check if plots should be displayed based on matplotlib backend"""
    import matplotlib
    return matplotlib.get_backend() != 'Agg'
