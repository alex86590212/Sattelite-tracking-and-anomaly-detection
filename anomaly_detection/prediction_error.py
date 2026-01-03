"""Prediction error-based anomaly detection."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from scipy import stats


class PredictionErrorDetector:
    """
    Detect anomalies using prediction error thresholds.
    
    Flags observations where ML model predictions deviate significantly
    from actual telemetry.
    """
    
    def __init__(
        self,
        threshold_percentile: float = 95.0,
        window_size: int = 100,
        adaptive: bool = True
    ):
        """
        Initialize detector.
        
        Args:
            threshold_percentile: Percentile for threshold (e.g., 95 = 95th percentile)
            window_size: Window size for adaptive threshold
            adaptive: Whether to use adaptive thresholding
        """
        self.threshold_percentile = threshold_percentile
        self.window_size = window_size
        self.adaptive = adaptive
        self.threshold = None
        self.error_history = []
    
    def compute_prediction_errors(
        self,
        observed: np.ndarray,
        predicted: np.ndarray
    ) -> np.ndarray:
        """
        Compute prediction errors.
        
        Args:
            observed: Observed values (n_samples, n_features)
            predicted: Predicted values (n_samples, n_features)
        
        Returns:
            Error magnitudes (n_samples,)
        """
        errors = np.linalg.norm(observed - predicted, axis=-1)
        return errors
    
    def fit(self, errors: np.ndarray):
        """
        Fit threshold on training errors.
        
        Args:
            errors: Training error magnitudes
        """
        self.error_history = list(errors)
        self.threshold = np.percentile(errors, self.threshold_percentile)
    
    def detect(
        self,
        observed: np.ndarray,
        predicted: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies.
        
        Args:
            observed: Observed values
            predicted: Predicted values
        
        Returns:
            Tuple of (anomaly_flags, error_scores)
        """
        errors = self.compute_prediction_errors(observed, predicted)
        
        if self.adaptive:
            # Update threshold based on recent errors
            if len(self.error_history) >= self.window_size:
                self.error_history = self.error_history[-self.window_size:]
            
            self.error_history.extend(errors)
            threshold = np.percentile(self.error_history, self.threshold_percentile)
        else:
            threshold = self.threshold
        
        if threshold is None:
            threshold = np.percentile(errors, self.threshold_percentile)
        
        anomaly_flags = errors > threshold
        
        return anomaly_flags, errors
    
    def detect_statistical(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        z_threshold: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using statistical z-score method.
        
        Args:
            observed: Observed values
            predicted: Predicted values
            z_threshold: Z-score threshold (e.g., 3.0 for 3-sigma)
        
        Returns:
            Tuple of (anomaly_flags, z_scores)
        """
        errors = self.compute_prediction_errors(observed, predicted)
        
        # Compute z-scores
        if len(self.error_history) > 0:
            mean_error = np.mean(self.error_history)
            std_error = np.std(self.error_history) + 1e-8
        else:
            mean_error = np.mean(errors)
            std_error = np.std(errors) + 1e-8
        
        z_scores = (errors - mean_error) / std_error
        anomaly_flags = np.abs(z_scores) > z_threshold
        
        # Update history
        self.error_history.extend(errors)
        if len(self.error_history) > self.window_size * 10:
            self.error_history = self.error_history[-self.window_size * 10:]
        
        return anomaly_flags, z_scores


class ResidualAnomalyDetector:
    """
    Detect anomalies in residuals (physics model errors).
    
    Useful for detecting thruster faults, drag spikes, etc.
    """
    
    def __init__(
        self,
        residual_threshold: float = 0.01,  # km
        velocity_residual_threshold: float = 0.001  # km/s
    ):
        """
        Initialize detector.
        
        Args:
            residual_threshold: Position residual threshold (km)
            velocity_residual_threshold: Velocity residual threshold (km/s)
        """
        self.residual_threshold = residual_threshold
        self.velocity_residual_threshold = velocity_residual_threshold
    
    def detect(
        self,
        physics_predicted: np.ndarray,
        observed: np.ndarray,
        anomaly_types: Optional[list] = None
    ) -> Dict[str, np.ndarray]:
        """
        Detect anomalies in residuals.
        
        Args:
            physics_predicted: Physics model predictions (n_samples, 6) [x,y,z,vx,vy,vz]
            observed: Observed telemetry (n_samples, 6)
            anomaly_types: Types to detect ['thruster', 'drag', 'sensor']
        
        Returns:
            Dictionary of anomaly flags for each type
        """
        if anomaly_types is None:
            anomaly_types = ['thruster', 'drag', 'sensor']
        
        residuals = observed - physics_predicted
        position_residuals = residuals[:, :3]
        velocity_residuals = residuals[:, 3:6]
        
        position_error = np.linalg.norm(position_residuals, axis=1)
        velocity_error = np.linalg.norm(velocity_residuals, axis=1)
        
        anomalies = {}
        
        if 'thruster' in anomaly_types:
            # Thruster fault: sudden velocity change
            velocity_change = np.abs(np.diff(velocity_error, prepend=velocity_error[0]))
            anomalies['thruster'] = velocity_change > self.velocity_residual_threshold * 5
        
        if 'drag' in anomaly_types:
            # Drag spike: gradual position error increase
            position_trend = np.convolve(
                position_error,
                np.array([-1, 0, 1]) / 2,
                mode='same'
            )
            anomalies['drag'] = (position_error > self.residual_threshold) & (position_trend > 0)
        
        if 'sensor' in anomaly_types:
            # Sensor glitch: high position error but low velocity error
            anomalies['sensor'] = (
                (position_error > self.residual_threshold) &
                (velocity_error < self.velocity_residual_threshold)
            )
        
        return anomalies

