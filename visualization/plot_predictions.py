"""Plot predictions, residuals, and errors."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple


def plot_predictions_vs_observed(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    title: str = "Predictions vs Observed",
    save_path: Optional[str] = None,
    show_residuals: bool = True
):
    """
    Plot predicted vs observed positions/velocities.
    
    Args:
        observed: DataFrame with observed telemetry
        predicted: DataFrame with predicted telemetry
        title: Plot title
        save_path: Path to save figure
        show_residuals: Whether to show residual subplot
    """
    n_plots = 6 if not show_residuals else 7
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3*n_plots), sharex=True)
    
    if n_plots == 6:
        axes_list = axes
    else:
        axes_list = axes[:-1]
        residual_ax = axes[-1]
    
    features = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    labels = ['X (km)', 'Y (km)', 'Z (km)', 'Vx (km/s)', 'Vy (km/s)', 'Vz (km/s)']
    
    for i, (feat, label) in enumerate(zip(features, labels)):
        axes_list[i].plot(observed['timestamp'], observed[feat], 
                          'b-', linewidth=1.5, label='Observed', alpha=0.7)
        axes_list[i].plot(predicted['timestamp'], predicted[feat], 
                         'r--', linewidth=1.5, label='Predicted', alpha=0.7)
        axes_list[i].set_ylabel(label, fontsize=11)
        axes_list[i].grid(True, alpha=0.3)
        axes_list[i].legend(loc='upper right')
    
    axes_list[0].set_title(title, fontsize=14, fontweight='bold')
    axes_list[-1].set_xlabel('Time', fontsize=11)
    
    # Residual plot
    if show_residuals:
        residuals = observed[features].values - predicted[features].values
        position_residual = np.linalg.norm(residuals[:, :3], axis=1)
        residual_ax.plot(observed['timestamp'], position_residual, 
                        'purple', linewidth=1.5, label='Position Residual')
        residual_ax.set_ylabel('Position Error (km)', fontsize=11)
        residual_ax.set_xlabel('Time', fontsize=11)
        residual_ax.grid(True, alpha=0.3)
        residual_ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_residuals(
    residuals: pd.DataFrame,
    title: str = "Prediction Residuals",
    save_path: Optional[str] = None
):
    """
    Plot residual components over time.
    
    Args:
        residuals: DataFrame with residual columns
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Position residuals
    axes[0].plot(residuals['timestamp'], residuals['x_residual'], 
                 'b-', linewidth=1.5, label='X residual', alpha=0.7)
    axes[0].plot(residuals['timestamp'], residuals['y_residual'], 
                 'r-', linewidth=1.5, label='Y residual', alpha=0.7)
    axes[0].plot(residuals['timestamp'], residuals['z_residual'], 
                 'g-', linewidth=1.5, label='Z residual', alpha=0.7)
    axes[0].set_ylabel('Position Residuals (km)', fontsize=11)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Total position residual
    axes[1].plot(residuals['timestamp'], residuals['position_residual'], 
                 'purple', linewidth=2, label='Position Error')
    axes[1].set_ylabel('Total Position Error (km)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Statistics
    mean_error = residuals['position_residual'].mean()
    std_error = residuals['position_residual'].std()
    axes[2].plot(residuals['timestamp'], residuals['position_residual'], 
                 'purple', linewidth=1.5, alpha=0.7, label='Error')
    axes[2].axhline(y=mean_error, color='r', linestyle='--', 
                    linewidth=2, label=f'Mean: {mean_error:.6f} km')
    axes[2].axhline(y=mean_error + std_error, color='orange', 
                    linestyle=':', linewidth=1.5, label=f'±1σ: {std_error:.6f} km')
    axes[2].axhline(y=mean_error - std_error, color='orange', linestyle=':', linewidth=1.5)
    axes[2].set_ylabel('Position Error (km)', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Histogram
    axes[3].hist(residuals['position_residual'], bins=50, color='purple', 
                alpha=0.7, edgecolor='black')
    axes[3].axvline(x=mean_error, color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {mean_error:.6f} km')
    axes[3].set_xlabel('Position Error (km)', fontsize=11)
    axes[3].set_ylabel('Frequency', fontsize=11)
    axes[3].grid(True, alpha=0.3, axis='y')
    axes[3].legend()
    
    axes[3].set_xlabel('Time', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_prediction_error_over_horizon(
    errors: dict,
    title: str = "Prediction Error vs Horizon",
    save_path: Optional[str] = None
):
    """
    Plot prediction error as a function of prediction horizon.
    
    Args:
        errors: Dictionary mapping horizon (int) to mean error (float)
        title: Plot title
        save_path: Path to save figure
    """
    horizons = sorted(errors.keys())
    mean_errors = [errors[h] for h in horizons]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(horizons, mean_errors, 'b-o', linewidth=2, markersize=8, label='Mean Error')
    ax.set_xlabel('Prediction Horizon (time steps)', fontsize=12)
    ax.set_ylabel('Mean Position Error (km)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

