"""LSTM/GRU models for orbit prediction."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import torch.nn.functional as F


class OrbitLSTM(nn.Module):
    """
    LSTM model for predicting orbit residuals.
    
    Learns non-linear drift and corrections to physics-based predictions.
    """
    
    def __init__(
        self,
        input_size: int = 6,  # x, y, z, vx, vy, vz
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 6,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of output features
            bidirectional: Whether to use bidirectional LSTM
        """
        super(OrbitLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, features)
        
        Returns:
            Output tensor of shape (batch, features)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc_layers(last_output)
        
        return output


class OrbitGRU(nn.Module):
    """
    GRU model for orbit prediction (lighter alternative to LSTM).
    """
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 6
    ):
        super(OrbitGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, h_n = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.fc_layers(last_output)
        return output


class ResidualPredictor(nn.Module):
    """
    Predicts residuals (corrections) to physics-based predictions.
    
    This hybrid approach combines physics (SGP4) with ML corrections.
    """
    
    def __init__(
        self,
        base_model: str = 'lstm',
        input_size: int = 12,  # physics_pred + observed features
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize residual predictor.
        
        Args:
            base_model: 'lstm' or 'gru'
            input_size: Input feature size (physics + observed)
            hidden_size: Hidden layer size
            num_layers: Number of layers
            dropout: Dropout rate
        """
        super(ResidualPredictor, self).__init__()
        
        if base_model.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 6)  # Residual for x, y, z, vx, vy, vz
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        rnn_out, _ = self.rnn(x)
        last_output = rnn_out[:, -1, :]
        residual = self.fc(last_output)
        return residual

