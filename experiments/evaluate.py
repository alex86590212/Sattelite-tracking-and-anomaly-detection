"""Evaluation script for orbit prediction and anomaly detection."""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional

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
    model_name: str = 'model'
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
    
    Returns:
        Dictionary of evaluation metrics
    """
    X_test, y_test = test_data
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        predictions = model(X_tensor).cpu().numpy()
    
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


def main():
    """Main evaluation function."""
    print("Evaluation metrics:")
    print("- Position error (km)")
    print("- Prediction horizon stability")
    print("- Anomaly detection metrics (precision, recall, F1, false alarm rate)")
    print("- Detection latency")
    
    # This is a template - implement with actual test data
    print("\nRun with actual test data to get metrics.")


if __name__ == '__main__':
    main()

