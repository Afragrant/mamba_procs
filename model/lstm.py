"""LSTM sequence model.

A standard (optionally bidirectional) multi-layer LSTM with a linear head,
sharing the same interface as the TCN and Mamba models so the three are
interchangeable for sequence tasks.
"""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM model with a linear head for sequence tasks.

    Args:
        input_size: Number of input features per time step.
        output_size: Number of target outputs.
        hidden_size: Hidden dimension of the LSTM.
        n_layers: Number of stacked LSTM layers (network depth).
        dropout: Dropout probability between LSTM layers (ignored if n_layers == 1).
        bidirectional: If True, use a bidirectional LSTM.
        seq_to_seq: If True, predict at every time step; otherwise use the last step.

    Input/Output shape: input ``(batch, sequence_length, input_size)``; output
    ``(batch, output_size)`` when ``seq_to_seq`` is False, else
    ``(batch, sequence_length, output_size)``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
        seq_to_seq: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        num_directions = 2 if bidirectional else 1
        self.linear = nn.Linear(hidden_size * num_directions, output_size)
        self.seq_to_seq = seq_to_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # out: (batch, sequence_length, hidden_size * num_directions).
        out, _ = self.lstm(x)
        if self.seq_to_seq:
            return self.linear(out)
        return self.linear(out[:, -1])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch, seq_len, features = 16, 100, 8
    model = LSTM(
        input_size=features,
        output_size=1,
        hidden_size=128,
        n_layers=2,
    ).to(device)
    sample = torch.randn(batch, seq_len, features, device=device)
    out = model(sample)
    print(model)
    print(f"input:  {tuple(sample.shape)}")
    print(f"output: {tuple(out.shape)}")
