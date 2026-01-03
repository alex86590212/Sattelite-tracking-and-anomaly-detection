"""Plot anomaly detection results."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple


def plot_anomalies_over_time(
    df: pd.DataFrame,
    anomaly_flags: np.ndarray,
    anomaly_scores: Optional[np.ndarray] = None,
    title: str = "Anomaly Detection Results",
    save_path: Optional[str] = None
):
    """
    Plot anomalies detected over time with telemetry.
    
    Args:
        df: DataFrame with telemetry data
        anomaly_flags: Boolean array indicating anomalies
        anomaly_scores: Optional anomaly scores
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Position with anomalies highlighted
    for i, coord in enumerate(['x', 'y', 'z']):
        axes[i].plot(df['timestamp'], df[coord], 'b-', linewidth=1.5, 
                    label=f'{coord.upper()}', alpha=0.7)
        
        # Highlight anomalies
        anomaly_indices = np.where(anomaly_flags)[0]
        if len(anomaly_indices) > 0:
            axes[i].scatter(df['timestamp'].iloc[anomaly_indices], 
                          df[coord].iloc[anomaly_indices],
                          color='red', s=50, marker='x', 
                          label='Anomaly', zorder=5, linewidths=2)
        
        axes[i].set_ylabel(f'{coord.upper()} Position (km)', fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right')
    
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    
    # Anomaly scores or flags
    if anomaly_scores is not None:
        axes[3].plot(df['timestamp'], anomaly_scores, 'purple', 
                    linewidth=1.5, label='Anomaly Score', alpha=0.7)
        axes[3].axhline(y=np.percentile(anomaly_scores, 95), 
                       color='r', linestyle='--', linewidth=2, 
                       label='95th Percentile Threshold')
        axes[3].fill_between(df['timestamp'], 0, anomaly_scores, 
                            where=anomaly_flags, color='red', alpha=0.3, 
                            label='Anomaly Region')
        axes[3].set_ylabel('Anomaly Score', fontsize=11)
    else:
        axes[3].plot(df['timestamp'], anomaly_flags.astype(int), 
                    'r-', linewidth=1.5, label='Anomaly Flag')
        axes[3].set_ylabel('Anomaly (0/1)', fontsize=11)
        axes[3].set_ylim(-0.1, 1.1)
    
    axes[3].set_xlabel('Time', fontsize=11)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_anomaly_types(
    df: pd.DataFrame,
    anomaly_dict: dict,
    title: str = "Anomaly Types Detected",
    save_path: Optional[str] = None
):
    """
    Plot different types of anomalies (thruster, drag, sensor).
    
    Args:
        df: DataFrame with telemetry
        anomaly_dict: Dictionary with keys like 'thruster', 'drag', 'sensor'
                      and values as boolean arrays
        title: Plot title
        save_path: Path to save figure
    """
    n_types = len(anomaly_dict)
    fig, axes = plt.subplots(n_types + 1, 1, figsize=(14, 4*(n_types+1)), sharex=True)
    
    # Plot position
    axes[0].plot(df['timestamp'], df['x'], 'b-', linewidth=1.5, 
                label='X', alpha=0.7)
    axes[0].plot(df['timestamp'], df['y'], 'r-', linewidth=1.5, 
                label='Y', alpha=0.7)
    axes[0].plot(df['timestamp'], df['z'], 'g-', linewidth=1.5, 
                label='Z', alpha=0.7)
    axes[0].set_ylabel('Position (km)', fontsize=11)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot each anomaly type
    colors = ['red', 'orange', 'purple', 'brown', 'pink']
    for idx, (anomaly_type, flags) in enumerate(anomaly_dict.items(), 1):
        axes[idx].plot(df['timestamp'], flags.astype(int), 
                      color=colors[idx-1], linewidth=2, label=anomaly_type.title())
        axes[idx].fill_between(df['timestamp'], 0, flags.astype(int), 
                               color=colors[idx-1], alpha=0.3)
        axes[idx].set_ylabel(f'{anomaly_type.title()} Anomaly', fontsize=11)
        axes[idx].set_ylim(-0.1, 1.1)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    
    axes[-1].set_xlabel('Time', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

