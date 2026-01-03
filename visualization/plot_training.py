"""Plot training metrics and model performance."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Training Curves",
    save_path: Optional[str] = None
):
    """
    Plot training and validation loss over epochs.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=4, 
           label='Training Loss', alpha=0.7)
    ax.plot(epochs, val_losses, 'r-s', linewidth=2, markersize=4, 
           label='Validation Loss', alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_uncertainty_bounds(
    timestamps: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    observed: Optional[np.ndarray] = None,
    title: str = "Predictions with Uncertainty Bounds",
    save_path: Optional[str] = None
):
    """
    Plot predictions with confidence intervals.
    
    Args:
        timestamps: Time array
        mean: Mean predictions (n_samples, n_features)
        lower: Lower confidence bounds
        upper: Upper confidence bounds
        observed: Optional observed values
        title: Plot title
        save_path: Path to save figure
    """
    n_features = mean.shape[1] if len(mean.shape) > 1 else 1
    
    if n_features == 1:
        mean = mean.reshape(-1, 1)
        lower = lower.reshape(-1, 1)
        upper = upper.reshape(-1, 1)
        if observed is not None:
            observed = observed.reshape(-1, 1)
    
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features), sharex=True)
    
    if n_features == 1:
        axes = [axes]
    
    feature_names = ['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz']
    
    for i in range(n_features):
        axes[i].fill_between(timestamps, lower[:, i], upper[:, i], 
                            alpha=0.3, color='blue', label='95% Confidence Interval')
        axes[i].plot(timestamps, mean[:, i], 'b-', linewidth=2, 
                    label='Mean Prediction', alpha=0.8)
        
        if observed is not None:
            axes[i].plot(timestamps, observed[:, i], 'r--', linewidth=1.5, 
                        label='Observed', alpha=0.7)
        
        axes[i].set_ylabel(f'{feature_names[i]}', fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right')
    
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[-1].set_xlabel('Time', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

