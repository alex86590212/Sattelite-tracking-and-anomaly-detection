"""Plot satellite trajectories and orbits."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from typing import Optional, List, Tuple


def plot_3d_trajectory(
    df: pd.DataFrame,
    title: str = "Satellite Trajectory",
    save_path: Optional[str] = None,
    show_earth: bool = True,
    earth_radius: float = 6378.137  # km
):
    """
    Plot 3D satellite trajectory.
    
    Args:
        df: DataFrame with columns ['x', 'y', 'z']
        title: Plot title
        save_path: Path to save figure (if None, displays)
        show_earth: Whether to show Earth sphere
        earth_radius: Earth radius in km
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(df['x'], df['y'], df['z'], 'b-', linewidth=1.5, label='Trajectory', alpha=0.7)
    ax.scatter(df['x'].iloc[0], df['y'].iloc[0], df['z'].iloc[0], 
               color='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(df['x'].iloc[-1], df['y'].iloc[-1], df['z'].iloc[-1], 
               color='red', s=100, marker='s', label='End', zorder=5)
    
    # Plot Earth sphere
    if show_earth:
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_earth = earth_radius * np.outer(np.cos(u), np.sin(v))
        y_earth = earth_radius * np.outer(np.sin(u), np.sin(v))
        z_earth = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_earth, y_earth, z_earth, alpha=0.2, color='blue')
    
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_zlabel('Z (km)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True)
    
    # Set equal aspect ratio
    max_range = np.array([
        df['x'].max() - df['x'].min(),
        df['y'].max() - df['y'].min(),
        df['z'].max() - df['z'].min()
    ]).max() / 2.0
    mid_x = (df['x'].max() + df['x'].min()) * 0.5
    mid_y = (df['y'].max() + df['y'].min()) * 0.5
    mid_z = (df['z'].max() + df['z'].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_2d_projections(
    df: pd.DataFrame,
    title: str = "Trajectory Projections",
    save_path: Optional[str] = None
):
    """
    Plot 2D projections of trajectory (XY, XZ, YZ planes).
    
    Args:
        df: DataFrame with columns ['x', 'y', 'z', 'timestamp']
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # XY projection
    axes[0].plot(df['x'], df['y'], 'b-', linewidth=1.5, alpha=0.7)
    axes[0].scatter(df['x'].iloc[0], df['y'].iloc[0], color='green', s=50, marker='o', zorder=5)
    axes[0].scatter(df['x'].iloc[-1], df['y'].iloc[-1], color='red', s=50, marker='s', zorder=5)
    axes[0].set_xlabel('X (km)', fontsize=11)
    axes[0].set_ylabel('Y (km)', fontsize=11)
    axes[0].set_title('XY Projection', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', adjustable='box')
    
    # XZ projection
    axes[1].plot(df['x'], df['z'], 'b-', linewidth=1.5, alpha=0.7)
    axes[1].scatter(df['x'].iloc[0], df['z'].iloc[0], color='green', s=50, marker='o', zorder=5)
    axes[1].scatter(df['x'].iloc[-1], df['z'].iloc[-1], color='red', s=50, marker='s', zorder=5)
    axes[1].set_xlabel('X (km)', fontsize=11)
    axes[1].set_ylabel('Z (km)', fontsize=11)
    axes[1].set_title('XZ Projection', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='box')
    
    # YZ projection
    axes[2].plot(df['y'], df['z'], 'b-', linewidth=1.5, alpha=0.7)
    axes[2].scatter(df['y'].iloc[0], df['z'].iloc[0], color='green', s=50, marker='o', zorder=5)
    axes[2].scatter(df['y'].iloc[-1], df['z'].iloc[-1], color='red', s=50, marker='s', zorder=5)
    axes[2].set_xlabel('Y (km)', fontsize=11)
    axes[2].set_ylabel('Z (km)', fontsize=11)
    axes[2].set_title('YZ Projection', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal', adjustable='box')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_position_over_time(
    df: pd.DataFrame,
    title: str = "Position Over Time",
    save_path: Optional[str] = None
):
    """
    Plot position components (x, y, z) over time.
    
    Args:
        df: DataFrame with columns ['timestamp', 'x', 'y', 'z']
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    axes[0].plot(df['timestamp'], df['x'], 'b-', linewidth=1.5, label='X')
    axes[0].set_ylabel('X Position (km)', fontsize=11)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(df['timestamp'], df['y'], 'r-', linewidth=1.5, label='Y')
    axes[1].set_ylabel('Y Position (km)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(df['timestamp'], df['z'], 'g-', linewidth=1.5, label='Z')
    axes[2].set_ylabel('Z Position (km)', fontsize=11)
    axes[2].set_xlabel('Time', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_velocity_over_time(
    df: pd.DataFrame,
    title: str = "Velocity Over Time",
    save_path: Optional[str] = None
):
    """
    Plot velocity components (vx, vy, vz) and magnitude over time.
    
    Args:
        df: DataFrame with columns ['timestamp', 'vx', 'vy', 'vz']
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Velocity components
    axes[0].plot(df['timestamp'], df['vx'], 'b-', linewidth=1.5, label='Vx', alpha=0.7)
    axes[0].plot(df['timestamp'], df['vy'], 'r-', linewidth=1.5, label='Vy', alpha=0.7)
    axes[0].plot(df['timestamp'], df['vz'], 'g-', linewidth=1.5, label='Vz', alpha=0.7)
    axes[0].set_ylabel('Velocity Components (km/s)', fontsize=11)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Velocity magnitude
    velocity_magnitude = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
    axes[1].plot(df['timestamp'], velocity_magnitude, 'purple', linewidth=2, label='|V|')
    axes[1].set_ylabel('Velocity Magnitude (km/s)', fontsize=11)
    axes[1].set_xlabel('Time', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

