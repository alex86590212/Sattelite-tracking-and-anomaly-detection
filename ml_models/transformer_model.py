"""Transformer model for long-horizon orbit prediction."""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class OrbitTransformer(nn.Module):
    """
    Transformer model for orbit prediction.
    
    Handles long-horizon predictions and missing telemetry well.
    """
    
    def __init__(
        self,
        input_size: int = 6,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_size: int = 6,
        max_seq_length: int = 1000
    ):
        """
        Initialize transformer model.
        
        Args:
            input_size: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            output_size: Number of output features
            max_seq_length: Maximum sequence length
        """
        super(OrbitTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, features)
            mask: Optional mask for missing data
        
        Returns:
            Output tensor of shape (batch, features)
        """
        # Project input to model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Transpose for transformer: (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        # Create padding mask if needed
        if mask is not None:
            # Convert boolean mask to attention mask
            # True values should be masked (set to -inf)
            src_key_padding_mask = mask.bool()
        else:
            src_key_padding_mask = None
        
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Use last timestep
        last_output = encoded[-1, :, :]  # (batch, d_model)
        
        # Project to output
        output = self.output_projection(last_output)
        
        return output


class TransformerResidualPredictor(nn.Module):
    """
    Transformer that predicts residuals to physics model.
    """
    
    def __init__(
        self,
        input_size: int = 12,  # physics + observed
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super(TransformerResidualPredictor, self).__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 6)  # Residual
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        
        if mask is not None:
            src_key_padding_mask = mask.bool()
        else:
            src_key_padding_mask = None
        
        encoded = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        last_output = encoded[-1, :, :]
        residual = self.output_projection(last_output)
        return residual

