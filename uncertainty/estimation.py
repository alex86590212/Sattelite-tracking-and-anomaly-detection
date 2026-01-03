"""Uncertainty estimation methods for orbit predictions."""

import numpy as np
import torch
from typing import Tuple, Optional, Dict
from scipy.stats import norm


class UncertaintyEstimator:
    """
    Estimate prediction uncertainty using various methods.
    """
    
    def __init__(self, method: str = 'ensemble'):
        """
        Initialize estimator.
        
        Args:
            method: 'ensemble', 'monte_carlo', 'quantile', or 'gaussian'
        """
        self.method = method
    
    def estimate_ensemble(
        self,
        predictions: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate uncertainty from ensemble predictions.
        
        Args:
            predictions: Ensemble predictions (n_models, n_samples, n_features)
            confidence: Confidence level
        
        Returns:
            (mean, lower_bound, upper_bound)
        """
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # Confidence intervals
        z_score = norm.ppf((1 + confidence) / 2)
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return mean, lower, upper
    
    def estimate_monte_carlo(
        self,
        model,
        x: torch.Tensor,
        num_samples: int = 100,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate uncertainty using Monte Carlo dropout.
        
        Args:
            model: PyTorch model with dropout
            x: Input tensor
            num_samples: Number of Monte Carlo samples
            confidence: Confidence level
        
        Returns:
            (mean, lower_bound, upper_bound)
        """
        model.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = model(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)  # (num_samples, batch, features)
        
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        z_score = norm.ppf((1 + confidence) / 2)
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return mean, lower, upper
    
    def estimate_quantile(
        self,
        predictions: np.ndarray,
        quantiles: Optional[list] = None
    ) -> Dict[str, np.ndarray]:
        """
        Estimate uncertainty using quantile regression.
        
        Args:
            predictions: Predictions from quantile models
            quantiles: List of quantiles (default: [0.05, 0.5, 0.95])
        
        Returns:
            Dictionary with quantile predictions
        """
        if quantiles is None:
            quantiles = [0.05, 0.5, 0.95]
        
        results = {}
        for i, q in enumerate(quantiles):
            results[f'q{q}'] = predictions[i]
        
        return results
    
    def propagate_uncertainty(
        self,
        initial_uncertainty: np.ndarray,
        time_horizon: float,
        dynamics_jacobian: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Propagate uncertainty forward in time.
        
        Args:
            initial_uncertainty: Initial covariance matrix
            time_horizon: Time horizon (hours)
            dynamics_jacobian: Jacobian of dynamics (optional)
        
        Returns:
            Propagated covariance matrix
        """
        if dynamics_jacobian is None:
            # Simplified: assume uncertainty grows linearly
            growth_factor = 1.0 + 0.1 * time_horizon
            return initial_uncertainty * growth_factor
        else:
            # Linearized propagation: P(t) = F @ P(0) @ F^T
            return dynamics_jacobian @ initial_uncertainty @ dynamics_jacobian.T


class ConfidenceIntervalCalculator:
    """Calculate confidence intervals for predictions."""
    
    @staticmethod
    def gaussian_confidence_intervals(
        mean: np.ndarray,
        std: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Gaussian confidence intervals.
        
        Args:
            mean: Mean predictions
            std: Standard deviations
            confidence: Confidence level
        
        Returns:
            (lower_bound, upper_bound)
        """
        z_score = norm.ppf((1 + confidence) / 2)
        lower = mean - z_score * std
        upper = mean + z_score * std
        return lower, upper
    
    @staticmethod
    def prediction_intervals(
        predictions: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals from sample distribution.
        
        Args:
            predictions: Sample predictions (n_samples, n_features)
            confidence: Confidence level
        
        Returns:
            (lower_bound, upper_bound)
        """
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(predictions, lower_percentile, axis=0)
        upper = np.percentile(predictions, upper_percentile, axis=0)
        
        return lower, upper

