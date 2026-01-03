"""Bayesian Neural Network for uncertainty-aware orbit prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.
    
    Implements variational inference for weight distributions.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        posterior_std_init: float = 0.1
    ):
        super(BayesianLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Mean and log variance of weights
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(
            torch.ones(out_features, in_features) * np.log(posterior_std_init**2)
        )
        
        # Bias
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(
            torch.ones(out_features) * np.log(posterior_std_init**2)
        )
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with weight sampling.
        
        Args:
            x: Input tensor
            sample: Whether to sample weights (True) or use mean (False)
        
        Returns:
            Tuple of (output, kl_divergence)
        """
        if sample:
            # Sample weights from posterior
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight_eps = torch.randn_like(self.weight_mu)
            weight = self.weight_mu + weight_std * weight_eps
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias_eps = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_std * bias_eps
        else:
            # Use mean weights
            weight = self.weight_mu
            bias = self.bias_mu
        
        output = F.linear(x, weight, bias)
        
        # Compute KL divergence (simplified)
        kl = self._kl_divergence()
        
        return output, kl
    
    def _kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior."""
        # KL(q(w) || p(w)) where q is Gaussian posterior, p is Gaussian prior
        # Formula: 0.5 * [tr(Σ_p^-1 Σ_q) + μ^T Σ_p^-1 μ - k + ln(det(Σ_p)/det(Σ_q))]
        # where Σ_p = σ²I, μ_p = 0, so this simplifies to:
        prior_var = self.prior_std ** 2
        
        # Weight KL: 0.5 * [sum(exp(logvar)/σ²) + sum(μ²/σ²) - k + k*log(σ²) - sum(logvar)]
        k_weights = self.in_features * self.out_features
        log_prior_var = torch.log(torch.tensor(prior_var, device=self.weight_mu.device, dtype=self.weight_mu.dtype))
        
        weight_kl = 0.5 * (
            torch.sum(torch.exp(self.weight_logvar) / prior_var) +
            torch.sum((self.weight_mu ** 2) / prior_var) -
            k_weights +
            k_weights * log_prior_var -
            torch.sum(self.weight_logvar)
        )
        
        # Bias KL
        k_bias = self.out_features
        bias_kl = 0.5 * (
            torch.sum(torch.exp(self.bias_logvar) / prior_var) +
            torch.sum((self.bias_mu ** 2) / prior_var) -
            k_bias +
            k_bias * log_prior_var -
            torch.sum(self.bias_logvar)
        )
        
        total_kl = weight_kl + bias_kl
        # Ensure KL is non-negative (should always be, but clamp for numerical stability)
        return torch.clamp(total_kl, min=0.0)


class BayesianLSTM(nn.Module):
    """
    Bayesian LSTM for uncertainty-aware orbit prediction.
    
    Provides both predictions and uncertainty estimates.
    """
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 6,
        prior_std: float = 1.0
    ):
        super(BayesianLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Standard LSTM (can be replaced with Bayesian LSTM)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Bayesian output layers
        self.bayesian_fc1 = BayesianLinear(hidden_size, hidden_size, prior_std)
        self.bayesian_fc2 = BayesianLinear(hidden_size, hidden_size // 2, prior_std)
        self.bayesian_fc3 = BayesianLinear(hidden_size // 2, output_size, prior_std)
    
    def forward(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x: Input tensor (batch, sequence_length, features)
            num_samples: Number of Monte Carlo samples for uncertainty
            sample: Whether to sample weights
        
        Returns:
            Tuple of (mean_prediction, std_prediction, total_kl)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        # Monte Carlo sampling for uncertainty
        predictions = []
        total_kl = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        for _ in range(num_samples):
            h1, kl1 = self.bayesian_fc1(last_output, sample=sample)
            h1 = F.relu(h1)
            total_kl = total_kl + kl1
            
            h2, kl2 = self.bayesian_fc2(h1, sample=sample)
            h2 = F.relu(h2)
            total_kl = total_kl + kl2
            
            pred, kl3 = self.bayesian_fc3(h2, sample=sample)
            total_kl = total_kl + kl3
            
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (num_samples, batch, output)
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        # Average KL across samples and ensure non-negative
        avg_kl = total_kl / num_samples
        avg_kl = torch.clamp(avg_kl, min=0.0)
        
        return mean_pred, std_pred, avg_kl


class BayesianOrbitPredictor(nn.Module):
    """
    Full Bayesian model for orbit prediction with uncertainty.
    
    Space agencies love this for mission-critical applications.
    """
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 6,
        prior_std: float = 1.0
    ):
        super(BayesianOrbitPredictor, self).__init__()
        
        self.bayesian_lstm = BayesianLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            prior_std=prior_std
        )
    
    def forward(
        self,
        x: torch.Tensor,
        num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty.
        
        Returns:
            (mean, std, kl_divergence)
        """
        return self.bayesian_lstm(x, num_samples=num_samples, sample=True)
    
    def predict_confidence_intervals(
        self,
        x: torch.Tensor,
        confidence: float = 0.95,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with confidence intervals.
        
        Args:
            x: Input tensor
            confidence: Confidence level (e.g., 0.95 for 95%)
            num_samples: Number of Monte Carlo samples
        
        Returns:
            (mean, lower_bound, upper_bound)
        """
        mean, std, kl = self.forward(x, num_samples=num_samples)
        
        # Assuming Gaussian distribution
        z_score = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return mean, lower, upper

