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
    device: str = 'cpu'
):
    """Train a model."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                output = model(batch_x)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
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
    tle_lines = loader.fetch_tle_from_celestrak(catalog_number=args.satellite_id)
    tle_data = loader.parse_tle(tle_lines[:3])
    sat = loader.create_satrec(tle_data)
    
    print("Generating telemetry...")
    processor = TelemetryProcessor(time_step_minutes=1.0)
    start_time = datetime.now()
    telemetry = processor.generate_telemetry(
        sat, start_time, duration_hours=24.0, add_noise=True
    )
    
    print("Preparing sequences...")
    X, y = processor.prepare_sequences(
        telemetry,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon
    )
    
    # Train/val split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
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
        model = OrbitLSTM(input_size=6, hidden_size=128, num_layers=2, output_size=6)
    elif args.model == 'transformer':
        model = OrbitTransformer(input_size=6, d_model=128, nhead=8, num_layers=4, output_size=6)
    elif args.model == 'bayesian':
        model = BayesianOrbitPredictor(input_size=6, hidden_size=128, num_layers=2, output_size=6)
    
    print("Training...")
    device = args.device if torch.cuda.is_available() else 'cpu'
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device
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

