"""LSTM 基准模型. 序列到序列回归: (B, T, C_in) -> (B, T, C_out)."""

from __future__ import annotations

import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=128, num_layers=2,
                 dropout=0.1, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        d = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d, out_channels)

    def forward(self, x):            # (B, T, C_in)
        h, _ = self.lstm(x)          # (B, T, d)
        return self.head(self.dropout(h))
