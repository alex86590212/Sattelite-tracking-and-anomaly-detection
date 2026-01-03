"""SGP4-based orbital propagation with drag and perturbations."""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
from sgp4.api import Satrec
from sgp4.conveniences import jday_datetime
import pandas as pd


class SGP4Propagator:
    """
    SGP4 orbital propagator with enhanced drag modeling.
    
    This provides the physics-based baseline that AI models
    will improve upon by learning residuals.
    """
    
    def __init__(self, sat: Satrec):
        """
        Initialize propagator with a satellite.
        
        Args:
            sat: Satrec object initialized from TLE
        """
        self.sat = sat
        self.epoch = None
        self._extract_epoch()
    
    def _extract_epoch(self):
        """Extract epoch from satellite object."""
        # SGP4 stores epoch as fractional days since 1900
        # This is a simplified extraction
        pass
    
    def propagate(
        self,
        time: datetime,
        include_drag: bool = True,
        include_perturbations: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Propagate satellite position and velocity to given time.
        
        Args:
            time: Target time for propagation
            include_drag: Whether to include atmospheric drag effects
            include_perturbations: Whether to include other perturbations
        
        Returns:
            Tuple of (position_km, velocity_km_s, error_code)
        """
        jd, fr = jday_datetime(time)
        error_code, r, v = self.sat.sgp4(jd, fr)
        
        # SGP4 already includes drag (via B* parameter) and perturbations
        # This wrapper allows for future enhancements
        
        return np.array(r), np.array(v), error_code
    
    def propagate_trajectory(
        self,
        start_time: datetime,
        end_time: datetime,
        time_step_minutes: float = 1.0
    ) -> pd.DataFrame:
        """
        Propagate trajectory over a time range.
        
        Args:
            start_time: Start time
            end_time: End time
            time_step_minutes: Time step in minutes
        
        Returns:
            DataFrame with trajectory data
        """
        timestamps = []
        positions = []
        velocities = []
        error_codes = []
        
        current_time = start_time
        
        while current_time <= end_time:
            r, v, error_code = self.propagate(current_time)
            
            timestamps.append(current_time)
            positions.append(r)
            velocities.append(v)
            error_codes.append(error_code)
            
            current_time += timedelta(minutes=time_step_minutes)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'x': [p[0] for p in positions],
            'y': [p[1] for p in positions],
            'z': [p[2] for p in positions],
            'vx': [v[0] for v in velocities],
            'vy': [v[1] for v in velocities],
            'vz': [v[2] for v in velocities],
            'error_code': error_codes
        })
        
        return df
    
    def compute_drag_effects(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        time: datetime,
        atmospheric_density: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute atmospheric drag acceleration.
        
        Args:
            position: Position vector (km)
            velocity: Velocity vector (km/s)
            time: Current time
            atmospheric_density: Atmospheric density (kg/m³), if None uses model
        
        Returns:
            Drag acceleration vector (km/s²)
        """
        # Simplified drag model
        # Real implementation would use atmospheric density models
        # (e.g., NRLMSISE-00, JB2008)
        
        altitude = np.linalg.norm(position) - 6378.137  # Earth radius in km
        
        if atmospheric_density is None:
            # Simplified exponential atmosphere model
            if altitude < 0:
                altitude = 0
            h0 = 500  # Reference altitude (km)
            rho0 = 1e-12  # Reference density (kg/m³)
            scale_height = 50  # Scale height (km)
            density = rho0 * np.exp(-(altitude - h0) / scale_height)
        else:
            density = atmospheric_density
        
        # Drag acceleration: a_drag = -0.5 * (Cd * A / m) * rho * v^2 * v_hat
        # Simplified: assume Cd*A/m = 0.01 m²/kg
        CdA_over_m = 0.01  # m²/kg
        v_mag = np.linalg.norm(velocity) * 1000  # Convert to m/s
        v_hat = velocity / (np.linalg.norm(velocity) + 1e-10)
        
        drag_accel = -0.5 * CdA_over_m * density * v_mag**2 * v_hat * 1e-3  # Convert to km/s²
        
        return drag_accel
    
    def compute_perturbations(
        self,
        position: np.ndarray,
        time: datetime
    ) -> Dict[str, np.ndarray]:
        """
        Compute various perturbation accelerations.
        
        Args:
            position: Position vector (km)
            time: Current time
        
        Returns:
            Dictionary of perturbation accelerations
        """
        perturbations = {}
        
        # J2 (oblateness) perturbation is already in SGP4
        # This method can be extended for additional perturbations:
        # - Solar radiation pressure
        # - Third-body effects (Moon, Sun)
        # - Tidal effects
        
        # Placeholder for future enhancements
        perturbations['solar_radiation'] = np.zeros(3)
        perturbations['third_body'] = np.zeros(3)
        
        return perturbations
    
    def predict_with_uncertainty(
        self,
        time: datetime,
        position_uncertainty: float = 0.001,
        velocity_uncertainty: float = 0.0001
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict position with uncertainty estimates.
        
        Args:
            time: Target time
            position_uncertainty: Position uncertainty (km)
            velocity_uncertainty: Velocity uncertainty (km/s)
        
        Returns:
            Tuple of (position, velocity, covariance_matrix)
        """
        r, v, error_code = self.propagate(time)
        
        # Simplified uncertainty model
        # Real implementation would propagate covariance matrix
        covariance = np.eye(6) * np.array([
            position_uncertainty**2,
            position_uncertainty**2,
            position_uncertainty**2,
            velocity_uncertainty**2,
            velocity_uncertainty**2,
            velocity_uncertainty**2
        ])
        
        return r, v, covariance

