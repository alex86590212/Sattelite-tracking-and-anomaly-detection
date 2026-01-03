"""Autoencoder-based anomaly detection."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import torch.nn.functional as F


class OrbitAutoencoder(nn.Module):
    """
    Autoencoder for detecting anomalies in orbit telemetry.
    
    Learns normal patterns and flags deviations as anomalies.
    """
    
    def __init__(
        self,
        input_size: int = 6,
        encoding_dim: int = 32,
        hidden_dims: Optional[list] = None
    ):
        """
        Initialize autoencoder.
        
        Args:
            input_size: Number of input features
            encoding_dim: Dimension of latent encoding
            hidden_dims: List of hidden layer dimensions (default: [64, 32])
        """
        super(OrbitAutoencoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_size
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_size))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, features) or (batch, sequence, features)
        
        Returns:
            Tuple of (reconstructed, encoding)
        """
        # Handle sequence input
        if len(x.shape) == 3:
            batch_size, seq_len, features = x.shape
            x_flat = x.view(batch_size * seq_len, features)
            encoded_flat = self.encoder(x_flat)
            decoded_flat = self.decoder(encoded_flat)
            encoded = encoded_flat.view(batch_size, seq_len, -1)
            decoded = decoded_flat.view(batch_size, seq_len, features)
        else:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
        
        return decoded, encoded
    
    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute reconstruction error.
        
        Args:
            x: Input tensor
            reduction: 'mean', 'sum', or 'none'
        
        Returns:
            Reconstruction error
        """
        reconstructed, _ = self.forward(x)
        error = F.mse_loss(x, reconstructed, reduction='none')
        
        if reduction == 'mean':
            return error.mean()
        elif reduction == 'sum':
            return error.sum()
        else:
            return error


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for anomaly detection.
    
    Provides better uncertainty estimates than standard autoencoder.
    """
    
    def __init__(
        self,
        input_size: int = 6,
        encoding_dim: int = 32,
        hidden_dim: int = 64
    ):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, encoding_dim)
        self.fc_logvar = nn.Linear(hidden_dim, encoding_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_size)
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            (reconstructed, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def compute_anomaly_score(
        self,
        x: torch.Tensor,
        num_samples: int = 10
    ) -> torch.Tensor:
        """
        Compute anomaly score using reconstruction error and KL divergence.
        
        Args:
            x: Input tensor
            num_samples: Number of samples for Monte Carlo estimation
        
        Returns:
            Anomaly scores
        """
        mu, logvar = self.encode(x)
        
        # Reconstruction error
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        recon_error = F.mse_loss(x, reconstructed, reduction='none').mean(dim=-1)
        
        # KL divergence (from standard normal prior)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        
        # Combined anomaly score
        anomaly_score = recon_error + 0.1 * kl
        
        return anomaly_score

