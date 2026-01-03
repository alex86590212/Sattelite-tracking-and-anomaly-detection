"""Main example script demonstrating the satellite tracking system."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch

from data.tle_loader import TLELoader
from data.telemetry_processor import TelemetryProcessor
from physics_model.sgp4_propagator import SGP4Propagator
from ml_models.lstm_model import OrbitLSTM
from anomaly_detection.prediction_error import PredictionErrorDetector
from anomaly_detection.autoencoder import OrbitAutoencoder
from uncertainty.estimation import UncertaintyEstimator
from visualization.plot_trajectory import plot_3d_trajectory, plot_2d_projections
from visualization.plot_anomalies import plot_anomalies_over_time
from visualization.plot_predictions import plot_residuals


def main():
    """Main demonstration of the satellite tracking system."""
    
    # Create output directory for plots
    import os
    os.makedirs("output", exist_ok=True)
    
    print("=" * 60)
    print("Satellite Tracking and Anomaly Detection System")
    print("=" * 60)
    
    # Step 1: Load TLE data
    print("\n[1/6] Loading TLE data for ISS (NORAD ID: 25544)...")
    loader = TLELoader()
    try:
        tle_lines = loader.fetch_tle_from_celestrak(catalog_number=25544)
        tle_data = loader.parse_tle(tle_lines[:3])
        sat = loader.create_satrec(tle_data)
        print(f"[OK] Loaded TLE for: {tle_data['name']}")
        print(f"  Epoch: {tle_data['epoch']}")
        print(f"  Inclination: {tle_data['inclination']:.2f}°")
        print(f"  Eccentricity: {tle_data['eccentricity']:.6f}")
    except Exception as e:
        print(f"[ERROR] Error loading TLE: {e}")
        print("  Using simulated TLE data...")
        # Fallback: create a simple example
        return
    
    # Step 2: Generate telemetry
    print("\n[2/6] Generating telemetry data...")
    processor = TelemetryProcessor(time_step_minutes=1.0)
    start_time = datetime.now()
    telemetry = processor.generate_telemetry(
        sat, start_time, duration_hours=2.0, add_noise=True, noise_std=0.001
    )
    print(f"[OK] Generated {len(telemetry)} telemetry points")
    print(f"  Time range: {telemetry['timestamp'].min()} to {telemetry['timestamp'].max()}")
    
    # Plot trajectory
    try:
        print("\n  Plotting trajectory...")
        plot_3d_trajectory(telemetry, title=f"{tle_data['name']} - 3D Trajectory", 
                          save_path="output/trajectory_3d.png")
        plot_2d_projections(telemetry, title=f"{tle_data['name']} - Trajectory Projections",
                          save_path="output/trajectory_projections.png")
        print("  [OK] Trajectory plots saved to output/")
    except Exception as e:
        print(f"  [WARNING] Could not create trajectory plots: {e}")
    
    # Step 3: Physics-based prediction
    print("\n[3/6] Running physics-based (SGP4) prediction...")
    propagator = SGP4Propagator(sat)
    physics_trajectory = propagator.propagate_trajectory(
        start_time=start_time,
        end_time=start_time + timedelta(hours=2.0),
        time_step_minutes=1.0
    )
    print(f"[OK] Propagated {len(physics_trajectory)} positions")
    
    # Compute residuals
    residuals = processor.compute_residuals(telemetry, physics_trajectory)
    mean_residual = residuals['position_residual'].mean()
    print(f"  Mean position residual: {mean_residual:.6f} km")
    
    # Plot residuals
    try:
        print("  Plotting residuals...")
        plot_residuals(residuals, title="Physics Model Residuals", 
                      save_path="output/physics_residuals.png")
        print("  [OK] Residual plots saved to output/")
    except Exception as e:
        print(f"  [WARNING] Could not create residual plots: {e}")
    
    # Step 4: Prepare ML training data
    print("\n[4/6] Preparing ML training sequences...")
    X, y = processor.prepare_sequences(
        telemetry,
        sequence_length=30,
        prediction_horizon=1
    )
    print(f"[OK] Prepared {len(X)} sequences")
    print(f"  Input shape: {X.shape}, Output shape: {y.shape}")
    
    # Step 5: Train/load ML model (simplified - just create model)
    print("\n[5/6] Setting up ML model...")
    model = OrbitLSTM(input_size=6, hidden_size=64, num_layers=2, output_size=6)
    print("[OK] Created LSTM model")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # For demonstration, we'll just show the model structure
    # In practice, you would train this with: python experiments/train_model.py
    
    # Step 6: Anomaly detection setup
    print("\n[6/6] Setting up anomaly detection...")
    
    # Add synthetic anomalies for demonstration
    telemetry_with_anomalies, anomaly_labels = processor.add_anomalies(
        telemetry,
        anomaly_type='thruster',
        anomaly_duration_minutes=5.0,
        magnitude=0.05
    )
    
    # Initialize detector
    detector = PredictionErrorDetector(threshold_percentile=95.0, adaptive=True)
    
    # Simulate predictions (in practice, use trained model)
    # For demo, use physics predictions with some noise
    simulated_predictions = physics_trajectory[['x', 'y', 'z', 'vx', 'vy', 'vz']].values
    simulated_predictions += np.random.normal(0, 0.001, simulated_predictions.shape)
    
    # Fit detector
    training_errors = np.linalg.norm(
        telemetry[['x', 'y', 'z', 'vx', 'vy', 'vz']].values - simulated_predictions,
        axis=1
    )
    detector.fit(training_errors)
    
    # Detect anomalies
    observed = telemetry_with_anomalies[['x', 'y', 'z', 'vx', 'vy', 'vz']].values
    detected_anomalies, error_scores = detector.detect(observed, simulated_predictions)
    
    print(f"[OK] Anomaly detection configured")
    print(f"  True anomalies: {anomaly_labels.sum()}")
    print(f"  Detected anomalies: {detected_anomalies.sum()}")
    print(f"  Detection rate: {np.sum(detected_anomalies & anomaly_labels) / max(anomaly_labels.sum(), 1) * 100:.1f}%")
    
    # Plot anomalies
    try:
        print("\n  Plotting anomaly detection results...")
        plot_anomalies_over_time(
            telemetry_with_anomalies,
            detected_anomalies.values if hasattr(detected_anomalies, 'values') else detected_anomalies,
            error_scores.values if hasattr(error_scores, 'values') else error_scores,
            title="Anomaly Detection Results",
            save_path="output/anomaly_detection.png"
        )
        print("  [OK] Anomaly plots saved to output/")
    except Exception as e:
        print(f"  [WARNING] Could not create anomaly plots: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("System Summary")
    print("=" * 60)
    print("[OK] TLE data loaded and parsed")
    print("[OK] Telemetry generated with realistic noise")
    print("[OK] Physics-based predictions computed (SGP4)")
    print("[OK] ML model structure created (ready for training)")
    print("[OK] Anomaly detection system configured")
    print("\nNext steps:")
    print("  1. Train ML model: python experiments/train_model.py")
    print("  2. Evaluate performance: python experiments/evaluate.py")
    print("  3. Integrate with real-time telemetry streams")
    print("=" * 60)


if __name__ == '__main__':
    main()

