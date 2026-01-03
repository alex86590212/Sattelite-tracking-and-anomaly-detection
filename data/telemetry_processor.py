"""Process and prepare telemetry data for ML models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sgp4.api import Satrec
from sgp4.conveniences import jday_datetime


class TelemetryProcessor:
    """Process satellite telemetry data for machine learning."""
    
    def __init__(self, time_step_minutes: float = 1.0):
        """
        Initialize telemetry processor.
        
        Args:
            time_step_minutes: Time step for generating telemetry (minutes)
        """
        self.time_step_minutes = time_step_minutes
    
    def generate_telemetry(
        self,
        sat: Satrec,
        start_time: datetime,
        duration_hours: float,
        add_noise: bool = True,
        noise_std: float = 0.001
    ) -> pd.DataFrame:
        """
        Generate time-series telemetry from SGP4 propagation.
        
        Args:
            sat: Satrec object for propagation
            start_time: Start time for telemetry
            duration_hours: Duration of telemetry in hours
            add_noise: Whether to add realistic noise
            noise_std: Standard deviation of noise (km)
        
        Returns:
            DataFrame with columns: timestamp, x, y, z, vx, vy, vz, position_error
        """
        timestamps = []
        positions = []
        velocities = []
        position_errors = []
        
        current_time = start_time
        end_time = start_time + timedelta(hours=duration_hours)
        
        while current_time <= end_time:
            jd, fr = jday_datetime(current_time)
            error_code, r, v = sat.sgp4(jd, fr)
            
            if error_code != 0:
                current_time += timedelta(minutes=self.time_step_minutes)
                continue
            
            # Convert from km to meters (SGP4 returns km)
            r_km = np.array(r)
            v_km_s = np.array(v)
            
            # Add noise if requested
            if add_noise:
                noise = np.random.normal(0, noise_std, 3)
                r_km += noise
                # Position error estimate (simplified)
                pos_error = np.linalg.norm(noise)
            else:
                pos_error = 0.0
            
            timestamps.append(current_time)
            positions.append(r_km)
            velocities.append(v_km_s)
            position_errors.append(pos_error)
            
            current_time += timedelta(minutes=self.time_step_minutes)
        
        # Create DataFrame
        if len(timestamps) == 0:
            return pd.DataFrame(columns=['timestamp', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'position_error'])
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'x': [p[0] for p in positions],
            'y': [p[1] for p in positions],
            'z': [p[2] for p in positions],
            'vx': [v[0] for v in velocities],
            'vy': [v[1] for v in velocities],
            'vz': [v[2] for v in velocities],
            'position_error': position_errors
        })
        
        return df
    
    def compute_residuals(
        self,
        observed: pd.DataFrame,
        predicted: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute residuals between observed and predicted positions.
        
        Args:
            observed: DataFrame with observed telemetry
            predicted: DataFrame with predicted telemetry
        
        Returns:
            DataFrame with residual components
        """
        residuals = pd.DataFrame()
        residuals['timestamp'] = observed['timestamp']
        
        for coord in ['x', 'y', 'z']:
            residuals[f'{coord}_residual'] = (
                observed[coord] - predicted[coord]
            )
        
        residuals['position_residual'] = np.sqrt(
            residuals['x_residual']**2 +
            residuals['y_residual']**2 +
            residuals['z_residual']**2
        )
        
        return residuals
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        prediction_horizon: int,
        features: Optional[List[str]] = None,
        normalize: bool = True,
        mean: Optional[Dict[str, float]] = None,
        std: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """
        Prepare sequences for time-series prediction.
        
        Args:
            df: Telemetry DataFrame
            sequence_length: Length of input sequences
            prediction_horizon: Steps ahead to predict
            features: List of feature columns (default: x, y, z, vx, vy, vz)
            normalize: Whether to normalize features
            mean: Pre-computed mean values (for validation/test sets)
            std: Pre-computed std values (for validation/test sets)
        
        Returns:
            Tuple of (X, y, mean_dict, std_dict) arrays for training
            For validation/test, use the mean and std from training
        """
        if features is None:
            features = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        
        df_normalized = df[features].copy()
        
        # Normalize features - compute stats if not provided (training), use provided (val/test)
        if normalize:
            if mean is None or std is None:
                # Training: compute statistics
                mean = {}
                std = {}
                for col in features:
                    mean[col] = df_normalized[col].mean()
                    std[col] = df_normalized[col].std()
            # Apply normalization
            for col in features:
                df_normalized[col] = (df_normalized[col] - mean[col]) / (std[col] + 1e-8)
        else:
            mean = None
            std = None
        
        X = []
        y = []
        
        for i in range(len(df_normalized) - sequence_length - prediction_horizon + 1):
            X.append(df_normalized.iloc[i:i+sequence_length].values)
            y.append(df_normalized.iloc[i+sequence_length+prediction_horizon-1].values)
        
        return np.array(X), np.array(y), mean, std
    
    def add_anomalies(
        self,
        df: pd.DataFrame,
        anomaly_type: str = 'thruster',
        anomaly_start: Optional[datetime] = None,
        anomaly_duration_minutes: float = 10.0,
        magnitude: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Inject synthetic anomalies into telemetry data.
        
        Args:
            df: Telemetry DataFrame
            anomaly_type: Type of anomaly ('thruster', 'drag', 'sensor')
            anomaly_start: Start time of anomaly (default: random)
            anomaly_duration_minutes: Duration of anomaly
            magnitude: Magnitude of anomaly
        
        Returns:
            Tuple of (modified_df, anomaly_labels)
        """
        df_modified = df.copy()
        anomaly_labels = pd.Series([False] * len(df), index=df.index)
        
        # Calculate valid range for anomaly
        min_start_idx = len(df) // 4
        min_end_idx = int(anomaly_duration_minutes / self.time_step_minutes)
        max_start_idx = len(df) - min_end_idx
        
        if anomaly_start is None:
            if max_start_idx <= min_start_idx:
                start_idx = max(0, len(df) // 2)
            else:
                start_idx = np.random.randint(min_start_idx, max_start_idx)
        else:
            start_idx = df[df['timestamp'] >= anomaly_start].index[0]
        
        end_idx = min(
            start_idx + int(anomaly_duration_minutes / self.time_step_minutes),
            len(df)
        )
        
        anomaly_labels.iloc[start_idx:end_idx] = True
        
        if anomaly_type == 'thruster':
            # Sudden velocity change
            df_modified.loc[start_idx:end_idx, 'vx'] += magnitude
            df_modified.loc[start_idx:end_idx, 'vy'] += magnitude * 0.5
        
        elif anomaly_type == 'drag':
            # Gradual deceleration
            for idx in range(start_idx, end_idx):
                progress = (idx - start_idx) / (end_idx - start_idx)
                drag_factor = magnitude * progress
                df_modified.loc[idx, 'vx'] *= (1 - drag_factor)
                df_modified.loc[idx, 'vy'] *= (1 - drag_factor)
                df_modified.loc[idx, 'vz'] *= (1 - drag_factor)
        
        elif anomaly_type == 'sensor':
            # Random noise spike
            noise = np.random.normal(0, magnitude, (end_idx - start_idx, 3))
            df_modified.loc[start_idx:end_idx, 'x'] += noise[:, 0]
            df_modified.loc[start_idx:end_idx, 'y'] += noise[:, 1]
            df_modified.loc[start_idx:end_idx, 'z'] += noise[:, 2]
        
        return df_modified, anomaly_labels

