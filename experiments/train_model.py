"""Training script for orbit prediction models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.tle_loader import TLELoader
from data.telemetry_processor import TelemetryProcessor
from physics_model.sgp4_propagator import SGP4Propagator
from ml_models.lstm_model import OrbitLSTM, ResidualPredictor
from ml_models.transformer_model import OrbitTransformer
from ml_models.bayesian_model import BayesianOrbitPredictor
from visualization.plot_training import plot_training_curves


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    early_stopping_patience: int = 10,
    is_bayesian: bool = False,
    kl_weight: float = 0.01
):
    """Train a model with early stopping."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Track separate losses for Bayesian models
    train_mse_losses = [] if is_bayesian else None
    train_kl_losses = [] if is_bayesian else None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_kl = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            
            # Handle Bayesian models that return (mean, std, kl_divergence)
            if is_bayesian:
                mean_pred, std_pred, kl_div = output
                mse_loss = criterion(mean_pred, batch_y)
                # Combine MSE loss with KL divergence (ELBO)
                # Ensure KL is non-negative and scale it properly
                kl_div = torch.clamp(kl_div, min=0.0)
                loss = mse_loss + kl_weight * kl_div
                
                train_mse += mse_loss.item()
                train_kl += kl_div.item()
            else:
                loss = criterion(output, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        if is_bayesian:
            train_mse /= len(train_loader)
            train_kl /= len(train_loader)
            train_mse_losses.append(train_mse)
            train_kl_losses.append(train_kl)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                output = model(batch_x)
                
                # Handle Bayesian models
                if is_bayesian:
                    mean_pred, std_pred, kl_div = output
                    loss = criterion(mean_pred, batch_y)
                else:
                    loss = criterion(output, batch_y)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            if is_bayesian:
                print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}, KL: {train_kl:.2f}), Val Loss: {val_loss:.6f}, Best Val: {best_val_loss:.6f}')
            else:
                print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Best Val: {best_val_loss:.6f}')
        
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1} (patience: {early_stopping_patience})')
            model.load_state_dict(best_model_state)
            break
    
    # Load best model if early stopping occurred
    if best_model_state is not None and patience_counter >= early_stopping_patience:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='Train orbit prediction model')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'transformer', 'bayesian'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sequence_length', type=int, default=60)
    parser.add_argument('--prediction_horizon', type=int, default=1)
    parser.add_argument('--satellite_id', type=int, default=25544)  # ISS
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    print("Loading TLE data...")
    loader = TLELoader()
    sat, tle_data = loader.load_satellite(catalog_number=args.satellite_id)
    print(f"Loaded TLE for: {tle_data['name']}")
    print(f"  Epoch: {tle_data['epoch']}")
    
    print("Generating telemetry...")
    processor = TelemetryProcessor(time_step_minutes=1.0)
    start_time = tle_data['epoch']  # Use TLE epoch for accurate propagation
    telemetry = processor.generate_telemetry(
        sat, start_time, duration_hours=24.0, add_noise=True
    )
    
    if len(telemetry) == 0:
        raise ValueError("No telemetry generated! Check TLE data and start time.")
    
    print(f"Generated {len(telemetry)} telemetry points")
    print("Preparing sequences...")
    X, y = processor.prepare_sequences(
        telemetry,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon
    )
    
    if len(X) == 0:
        raise ValueError(f"No sequences generated! Need at least {args.sequence_length + args.prediction_horizon} telemetry points.")
    
    # Train/val split - use middle section for validation to avoid easier end-of-orbit bias
    # This creates a more balanced split: first 70% train, middle 10% val, last 20% train
    total_len = len(X)
    train_end = int(0.7 * total_len)
    val_end = int(0.8 * total_len)
    
    # Shuffle indices for more balanced distribution (but keep temporal structure within splits)
    indices = np.arange(total_len)
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)
    
    train_indices = np.concatenate([indices[:train_end], indices[val_end:]])
    val_indices = indices[train_end:val_end]
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    print(f"Train set: {len(X_train)} samples, Val set: {len(X_val)} samples")
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Creating {args.model} model...")
    if args.model == 'lstm':
        # LSTM: 2 layers, dropout 0.2 (more regularization)
        model = OrbitLSTM(input_size=6, hidden_size=128, num_layers=2, dropout=0.01, output_size=6)
    elif args.model == 'transformer':
        # Transformer: Reduced capacity + higher dropout to prevent overfitting
        # 2 layers instead of 4, dropout 0.2 instead of 0.1
        model = OrbitTransformer(
            input_size=6, 
            d_model=128, 
            nhead=8, 
            num_layers=2,  # Reduced from 4 to 2
            dim_feedforward=256,  # Reduced from 512 to 256
            dropout=0.01,  # Increased from 0.1 to 0.2
            output_size=6
        )
    elif args.model == 'bayesian':
        model = BayesianOrbitPredictor(input_size=6, hidden_size=128, num_layers=2, output_size=6)
    
    print("Training...")
    device = args.device if torch.cuda.is_available() else 'cpu'
    is_bayesian = (args.model == 'bayesian')
    # Bayesian models need more patience, lower KL weight, and higher learning rate
    early_stopping_patience = 20 if is_bayesian else 10
    kl_weight = 0.0001 if is_bayesian else 0.01  # Much lower KL weight - KL is very large
    learning_rate = args.lr * 2.0 if is_bayesian else args.lr  # Higher LR for faster convergence
    
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        learning_rate=learning_rate,
        device=device,
        is_bayesian=is_bayesian,
        early_stopping_patience=early_stopping_patience,
        kl_weight=kl_weight
    )
    
    # Save model
    model_path = Path(__file__).parent.parent / 'ml_models' / f'{args.model}_trained.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training curves
    try:
        output_dir = Path(__file__).parent.parent / 'output'
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / f'training_curves_{args.model}.png'
        plot_training_curves(
            train_losses, val_losses,
            title=f"{args.model.upper()} Training Curves",
            save_path=str(plot_path)
        )
        print(f"Training curves saved to {plot_path}")
    except Exception as e:
        print(f"Warning: Could not create training curves plot: {e}")
    
    print("Training complete!")


if __name__ == '__main__':
    main()

