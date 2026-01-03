"""Evaluation script for orbit prediction and anomaly detection."""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional
from datetime import timedelta, timezone

sys.path.append(str(Path(__file__).parent.parent))

from data.telemetry_processor import TelemetryProcessor
from ml_models.lstm_model import OrbitLSTM
from anomaly_detection.prediction_error import PredictionErrorDetector
from anomaly_detection.autoencoder import OrbitAutoencoder
from visualization.plot_predictions import plot_predictions_vs_observed, plot_residuals, plot_prediction_error_over_horizon


def compute_position_error(
    predicted: np.ndarray,
    observed: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute position error in km.
    
    Args:
        predicted: Predicted positions (n_samples, 3)
        observed: Observed positions (n_samples, 3)
    
    Returns:
        (mean_error_km, errors_km)
    """
    errors = np.linalg.norm(predicted[:, :3] - observed[:, :3], axis=1)
    return np.mean(errors), errors


def compute_prediction_horizon_stability(
    errors: np.ndarray,
    max_horizon: int = 100
) -> Dict[int, float]:
    """
    Compute prediction error stability over different horizons.
    
    Args:
        errors: Prediction errors
        max_horizon: Maximum prediction horizon
    
    Returns:
        Dictionary mapping horizon to mean error
    """
    stability = {}
    for horizon in range(1, min(max_horizon, len(errors)) + 1):
        horizon_errors = errors[::horizon]
        stability[horizon] = np.mean(horizon_errors)
    return stability


def compute_anomaly_metrics(
    true_anomalies: np.ndarray,
    detected_anomalies: np.ndarray
) -> Dict[str, float]:
    """
    Compute anomaly detection metrics.
    
    Args:
        true_anomalies: True anomaly labels
        detected_anomalies: Detected anomaly labels
    
    Returns:
        Dictionary of metrics
    """
    tp = np.sum(true_anomalies & detected_anomalies)
    fp = np.sum(~true_anomalies & detected_anomalies)
    fn = np.sum(true_anomalies & ~detected_anomalies)
    tn = np.sum(~true_anomalies & ~detected_anomalies)
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    false_alarm_rate = fp / (fp + tn + 1e-10)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_alarm_rate': false_alarm_rate,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn)
    }


def compute_detection_latency(
    true_anomalies: np.ndarray,
    detected_anomalies: np.ndarray,
    time_step_minutes: float = 1.0
) -> float:
    """
    Compute average detection latency.
    
    Args:
        true_anomalies: True anomaly labels
        detected_anomalies: Detected anomaly labels
        time_step_minutes: Time step in minutes
    
    Returns:
        Average latency in minutes
    """
    latencies = []
    
    # Find start of each true anomaly period
    anomaly_starts = np.where(np.diff(np.concatenate([[False], true_anomalies])) == True)[0]
    
    for start in anomaly_starts:
        # Find when it was detected
        detection_idx = np.where(detected_anomalies[start:])[0]
        if len(detection_idx) > 0:
            latency = detection_idx[0] * time_step_minutes
            latencies.append(latency)
    
    return np.mean(latencies) if latencies else float('inf')


def evaluate_model(
    model,
    test_data: Tuple[np.ndarray, np.ndarray],
    test_timestamps: Optional[pd.Series] = None,
    device: str = 'cpu',
    plot_results: bool = True,
    model_name: str = 'model',
    is_bayesian: bool = False
) -> Dict:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        test_data: Tuple of (X_test, y_test)
        test_timestamps: Optional timestamps for plotting
        device: Device to run on
        plot_results: Whether to generate plots
        model_name: Name of model for plot titles
        is_bayesian: Whether this is a Bayesian model (returns tuple)
    
    Returns:
        Dictionary of evaluation metrics
    """
    X_test, y_test = test_data
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        output = model(X_tensor)
        
        # Handle Bayesian models that return (mean, std, kl)
        if is_bayesian:
            predictions = output[0].cpu().numpy()  # Use mean prediction
        else:
            predictions = output.cpu().numpy()
    
    # Position error
    mean_error, errors = compute_position_error(predictions, y_test)
    
    # Prediction horizon stability
    stability = compute_prediction_horizon_stability(errors)
    
    metrics = {
        'mean_position_error_km': mean_error,
        'std_position_error_km': np.std(errors),
        'max_position_error_km': np.max(errors),
        'prediction_horizon_stability': stability
    }
    
    # Generate plots if requested
    if plot_results:
        try:
            output_dir = Path(__file__).parent.parent / 'output'
            output_dir.mkdir(exist_ok=True)
            
            # Create DataFrames for plotting
            if test_timestamps is not None:
                observed_df = pd.DataFrame({
                    'timestamp': test_timestamps.iloc[:len(y_test)],
                    'x': y_test[:, 0], 'y': y_test[:, 1], 'z': y_test[:, 2],
                    'vx': y_test[:, 3], 'vy': y_test[:, 4], 'vz': y_test[:, 5]
                })
                predicted_df = pd.DataFrame({
                    'timestamp': test_timestamps.iloc[:len(predictions)],
                    'x': predictions[:, 0], 'y': predictions[:, 1], 'z': predictions[:, 2],
                    'vx': predictions[:, 3], 'vy': predictions[:, 4], 'vz': predictions[:, 5]
                })
                
                # Plot predictions vs observed
                plot_predictions_vs_observed(
                    observed_df, predicted_df,
                    title=f"{model_name} - Predictions vs Observed",
                    save_path=str(output_dir / f'evaluation_predictions_{model_name}.png')
                )
                
                # Compute and plot residuals
                from data.telemetry_processor import TelemetryProcessor
                processor = TelemetryProcessor()
                residuals = processor.compute_residuals(observed_df, predicted_df)
                plot_residuals(
                    residuals,
                    title=f"{model_name} - Prediction Residuals",
                    save_path=str(output_dir / f'evaluation_residuals_{model_name}.png')
                )
            
            # Plot error over horizon
            plot_prediction_error_over_horizon(
                stability,
                title=f"{model_name} - Prediction Error vs Horizon",
                save_path=str(output_dir / f'evaluation_horizon_{model_name}.png')
            )
            
            print(f"Evaluation plots saved to {output_dir}/")
        except Exception as e:
            print(f"Warning: Could not create evaluation plots: {e}")
    
    return metrics


def load_trained_model(model_type: str, model_path: Path, device: str = 'cpu'):
    """Load a trained model from disk."""
    from ml_models.lstm_model import OrbitLSTM
    from ml_models.transformer_model import OrbitTransformer
    from ml_models.bayesian_model import BayesianOrbitPredictor
    
    if model_type == 'lstm':
        model = OrbitLSTM(input_size=6, hidden_size=128, num_layers=2, dropout=0.01, output_size=6)
    elif model_type == 'transformer':
        model = OrbitTransformer(
            input_size=6, d_model=128, nhead=8, num_layers=2,
            dim_feedforward=256, dropout=0.01, output_size=6
        )
    elif model_type == 'bayesian':
        model = BayesianOrbitPredictor(input_size=6, hidden_size=128, num_layers=2, output_size=6)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def main():
    """Main evaluation function."""
    import argparse
    from data.tle_loader import TLELoader
    from data.telemetry_processor import TelemetryProcessor
    
    parser = argparse.ArgumentParser(description='Evaluate trained orbit prediction models')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'transformer', 'bayesian', 'all'])
    parser.add_argument('--satellite_id', type=int, default=25544)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--sequence_length', type=int, default=60)
    parser.add_argument('--test_duration_hours', type=float, default=6.0)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Load TLE and generate test data
    print("\n[1/4] Loading TLE data and generating test telemetry...")
    loader = TLELoader()
    sat, tle_data = loader.load_satellite(catalog_number=args.satellite_id)
    print(f"Loaded TLE for: {tle_data['name']}")
    
    processor = TelemetryProcessor(time_step_minutes=1.0)
    # Generate test data from AFTER training period to avoid overlap
    # Training uses 24 hours from epoch, so start test 25 hours after epoch
    start_time = tle_data['epoch'] + timedelta(hours=25)
    test_telemetry = processor.generate_telemetry(
        sat, start_time, duration_hours=args.test_duration_hours, add_noise=True
    )
    
    if len(test_telemetry) == 0:
        raise ValueError("No test telemetry generated!")
    
    print(f"Generated {len(test_telemetry)} test telemetry points")
    print(f"Test period: {test_telemetry['timestamp'].min()} to {test_telemetry['timestamp'].max()}")
    
    # Load normalization statistics from training
    # Note: In production, these should be saved during training
    print("\n[2/4] Loading normalization statistics...")
    # For now, we'll compute from a sample, but ideally load from training
    # This is a workaround - in production, save train_mean and train_std during training
    sample_telemetry = processor.generate_telemetry(
        sat, tle_data['epoch'], duration_hours=24.0, add_noise=True
    )
    _, _, train_mean, train_std = processor.prepare_sequences(
        sample_telemetry,
        sequence_length=args.sequence_length,
        prediction_horizon=1,
        normalize=True
    )
    
    # Prepare test sequences using TRAINING normalization statistics
    print("Preparing test sequences...")
    X_test, y_test, _, _ = processor.prepare_sequences(
        test_telemetry,
        sequence_length=args.sequence_length,
        prediction_horizon=1,
        normalize=True,
        mean=train_mean,
        std=train_std
    )
    
    if len(X_test) == 0:
        raise ValueError("No test sequences generated!")
    
    print(f"Prepared {len(X_test)} test sequences")
    
    # Evaluate physics model (SGP4) as baseline
    print("\n[3/5] Evaluating Physics Model (SGP4) baseline...")
    from physics_model.sgp4_propagator import SGP4Propagator
    
    propagator = SGP4Propagator(sat)
    physics_predictions = []
    
    # Generate physics predictions for each test target
    # y_test corresponds to targets at indices: sequence_length, sequence_length+1, ...
    for i in range(len(y_test)):
        # The target timestamp is at index: sequence_length + i in the original telemetry
        target_idx = args.sequence_length + i
        if target_idx < len(test_telemetry):
            target_time = test_telemetry.iloc[target_idx]['timestamp']
            # Convert pandas Timestamp to Python datetime and ensure timezone-aware
            if isinstance(target_time, pd.Timestamp):
                target_time = target_time.to_pydatetime()
                # If timezone-naive, assume UTC (SGP4 works with UTC)
                if target_time.tzinfo is None:
                    target_time = target_time.replace(tzinfo=timezone.utc)
            r, v, error_code = propagator.propagate(target_time)
            if error_code == 0:
                physics_predictions.append(np.concatenate([r, v]))
            else:
                # Use previous prediction if SGP4 fails
                if len(physics_predictions) > 0:
                    physics_predictions.append(physics_predictions[-1])
                else:
                    physics_predictions.append(np.zeros(6))
        else:
            # Use last available prediction
            if len(physics_predictions) > 0:
                physics_predictions.append(physics_predictions[-1])
            else:
                physics_predictions.append(np.zeros(6))
    
    physics_predictions = np.array(physics_predictions)
    
    # Normalize physics predictions using training statistics
    if train_mean and train_std:
        physics_normalized = physics_predictions.copy()
        features = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        for idx, col in enumerate(features):
            physics_normalized[:, idx] = (physics_predictions[:, idx] - train_mean[col]) / (train_std[col] + 1e-8)
        physics_predictions = physics_normalized
    
    # Compute physics model metrics
    physics_mean_error, physics_errors = compute_position_error(physics_predictions, y_test)
    physics_stability = compute_prediction_horizon_stability(physics_errors)
    
    physics_metrics = {
        'mean_position_error_km': physics_mean_error,
        'std_position_error_km': np.std(physics_errors),
        'max_position_error_km': np.max(physics_errors),
        'median_position_error_km': np.median(physics_errors),
        'prediction_horizon_stability': physics_stability
    }
    
    print(f"    Mean position error: {physics_mean_error:.6f} km")
    print(f"    Std position error: {np.std(physics_errors):.6f} km")
    print(f"    Max position error: {np.max(physics_errors):.6f} km")
    print(f"    Median position error: {np.median(physics_errors):.6f} km")
    
    # Models to evaluate
    models_to_eval = ['lstm', 'transformer', 'bayesian'] if args.model == 'all' else [args.model]
    
    print(f"\n[4/5] Evaluating {len(models_to_eval)} ML model(s)...")
    
    results = {'physics': physics_metrics}
    model_dir = Path(__file__).parent.parent / 'ml_models'
    
    for model_type in models_to_eval:
        model_path = model_dir / f'{model_type}_trained.pt'
        
        if not model_path.exists():
            print(f"  WARNING: Model {model_type} not found at {model_path}, skipping...")
            continue
        
        print(f"\n  Evaluating {model_type.upper()} model...")
        
        # Load model
        model = load_trained_model(model_type, model_path, device=args.device)
        
        # Handle Bayesian models (return tuple)
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(args.device)
            output = model(X_tensor)
            
            if model_type == 'bayesian':
                predictions, std_pred, kl_div = output
                predictions = predictions.cpu().numpy()
                print(f"    Uncertainty (std): {std_pred.mean().item():.6f}")
            else:
                predictions = output.cpu().numpy()
        
        # Compute metrics
        mean_error, errors = compute_position_error(predictions, y_test)
        stability = compute_prediction_horizon_stability(errors)
        
        metrics = {
            'mean_position_error_km': mean_error,
            'std_position_error_km': np.std(errors),
            'max_position_error_km': np.max(errors),
            'median_position_error_km': np.median(errors),
            'prediction_horizon_stability': stability
        }
        
        results[model_type] = metrics
        
        print(f"    Mean position error: {mean_error:.6f} km")
        print(f"    Std position error: {np.std(errors):.6f} km")
        print(f"    Max position error: {np.max(errors):.6f} km")
        print(f"    Median position error: {np.median(errors):.6f} km")
        
        # Generate plots
        print(f"\n[5/5] Generating plots for {model_type}...")
        test_timestamps = test_telemetry['timestamp'].iloc[:len(y_test)]
        evaluate_model(
            model,
            (X_test, y_test),
            test_timestamps=test_timestamps,
            device=args.device,
            plot_results=True,
            model_name=model_type,
            is_bayesian=(model_type == 'bayesian')
        )
    
    # Print comparison including physics model
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("Model Comparison (vs Physics Baseline)")
        print("=" * 60)
        print(f"{'Model':<15} {'Mean Error (km)':<20} {'Std Error (km)':<20} {'Max Error (km)':<20} {'vs Physics':<15}")
        print("-" * 90)
        
        physics_error = results['physics']['mean_position_error_km']
        
        for model_type, metrics in results.items():
            mean_err = metrics['mean_position_error_km']
            std_err = metrics['std_position_error_km']
            max_err = metrics['max_position_error_km']
            
            if model_type == 'physics':
                vs_physics = "baseline"
            else:
                improvement = ((physics_error - mean_err) / physics_error) * 100
                vs_physics = f"{improvement:+.1f}%"
            
            print(f"{model_type.upper():<15} {mean_err:<20.6f} {std_err:<20.6f} {max_err:<20.6f} {vs_physics:<15}")
        
        # Find best ML model (excluding physics)
        ml_results = {k: v for k, v in results.items() if k != 'physics'}
        if ml_results:
            best_model = min(ml_results.items(), key=lambda x: x[1]['mean_position_error_km'])
            improvement = ((physics_error - best_model[1]['mean_position_error_km']) / physics_error) * 100
            print(f"\nBest ML model: {best_model[0].upper()} "
                  f"(Mean error: {best_model[1]['mean_position_error_km']:.6f} km, "
                  f"{improvement:+.1f}% vs physics)")
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

