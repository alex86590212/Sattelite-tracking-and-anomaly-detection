"""Bayesian surprise-based anomaly detection."""

import numpy as np
import torch
from typing import Tuple, Optional
from scipy.stats import multivariate_normal


class BayesianSurpriseDetector:
    """
    Detect anomalies using Bayesian surprise scores.
    
    Measures how "surprising" new observations are given the model's
    learned distribution.
    """
    
    def __init__(self, prior_weight: float = 0.1):
        """
        Initialize detector.
        
        Args:
            prior_weight: Weight for prior in surprise calculation
        """
        self.prior_weight = prior_weight
        self.posterior_mean = None
        self.posterior_cov = None
        self.prior_mean = None
        self.prior_cov = None
        self.observation_count = 0
    
    def update_posterior(
        self,
        observations: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ):
        """
        Update posterior distribution with new observations.
        
        Args:
            observations: New observations (n_samples, n_features)
            uncertainties: Observation uncertainties (n_samples, n_features)
        """
        if uncertainties is None:
            uncertainties = np.ones_like(observations) * 0.001
        
        n_samples, n_features = observations.shape
        
        # Initialize prior if needed
        if self.prior_mean is None:
            self.prior_mean = np.zeros(n_features)
            self.prior_cov = np.eye(n_features) * 10.0  # Wide prior
        
        if self.posterior_mean is None:
            self.posterior_mean = np.zeros(n_features)
            self.posterior_cov = np.eye(n_features) * 10.0
        
        # Bayesian update (simplified)
        for obs, unc in zip(observations, uncertainties):
            # Observation covariance
            obs_cov = np.diag(unc**2)
            
            # Kalman-like update
            K = self.posterior_cov @ np.linalg.inv(self.posterior_cov + obs_cov)
            self.posterior_mean = self.posterior_mean + K @ (obs - self.posterior_mean)
            self.posterior_cov = (np.eye(n_features) - K) @ self.posterior_cov
            
            self.observation_count += 1
    
    def compute_surprise(
        self,
        observations: np.ndarray,
        predictions: np.ndarray,
        prediction_uncertainties: np.ndarray
    ) -> np.ndarray:
        """
        Compute Bayesian surprise scores.
        
        Surprise = KL(posterior || prior) - KL(new_posterior || prior)
        
        Args:
            observations: New observations
            predictions: Model predictions
            prediction_uncertainties: Prediction uncertainties
        
        Returns:
            Surprise scores (higher = more surprising)
        """
        if self.posterior_mean is None:
            # Initialize with first observations
            self.update_posterior(observations, prediction_uncertainties)
            return np.zeros(len(observations))
        
        surprise_scores = []
        
        for obs, pred, unc in zip(observations, predictions, prediction_uncertainties):
            # Prior: current posterior
            prior_mean = self.posterior_mean.copy()
            prior_cov = self.posterior_cov.copy()
            
            # Likelihood: prediction with uncertainty
            likelihood_mean = pred
            likelihood_cov = np.diag(unc**2)
            
            # Posterior after seeing observation
            # Simplified Bayesian update
            K = prior_cov @ np.linalg.inv(prior_cov + likelihood_cov)
            posterior_mean = prior_mean + K @ (obs - prior_mean)
            posterior_cov = (np.eye(len(prior_mean)) - K) @ prior_cov
            
            # Compute KL divergence: KL(posterior || prior)
            try:
                kl_div = self._kl_divergence(
                    posterior_mean, posterior_cov,
                    prior_mean, prior_cov
                )
            except:
                kl_div = 0.0
            
            surprise_scores.append(kl_div)
            
            # Update posterior
            self.posterior_mean = posterior_mean
            self.posterior_cov = posterior_cov
        
        return np.array(surprise_scores)
    
    def _kl_divergence(
        self,
        mean1: np.ndarray,
        cov1: np.ndarray,
        mean2: np.ndarray,
        cov2: np.ndarray
    ) -> float:
        """Compute KL divergence between two Gaussians."""
        try:
            # KL(q || p) = 0.5 * (tr(Σp^-1 Σq) + (μp - μq)^T Σp^-1 (μp - μq) - k + ln(det(Σp)/det(Σq)))
            k = len(mean1)
            cov2_inv = np.linalg.inv(cov2 + np.eye(k) * 1e-6)
            
            diff = mean2 - mean1
            kl = 0.5 * (
                np.trace(cov2_inv @ cov1) +
                diff.T @ cov2_inv @ diff -
                k +
                np.log(np.linalg.det(cov2 + np.eye(k) * 1e-6) / 
                       (np.linalg.det(cov1 + np.eye(k) * 1e-6) + 1e-10))
            )
            return max(0, kl)  # KL is non-negative
        except:
            return 0.0
    
    def detect(
        self,
        observations: np.ndarray,
        predictions: np.ndarray,
        prediction_uncertainties: np.ndarray,
        surprise_threshold: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using surprise scores.
        
        Args:
            observations: New observations
            predictions: Model predictions
            prediction_uncertainties: Prediction uncertainties
            surprise_threshold: Threshold for anomaly flagging
        
        Returns:
            Tuple of (anomaly_flags, surprise_scores)
        """
        surprise_scores = self.compute_surprise(
            observations, predictions, prediction_uncertainties
        )
        
        anomaly_flags = surprise_scores > surprise_threshold
        
        return anomaly_flags, surprise_scores

