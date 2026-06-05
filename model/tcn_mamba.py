"""TCN-Mamba hybrid sequence model.

A TCN front-end extracts local temporal features through dilated causal
convolutions, then a stack of Mamba3 blocks models long-range dependencies on
the resulting representation. Shares the interface of the standalone TCN, Mamba
and LSTM models so all four are interchangeable.
"""

import torch
import torch.nn as nn

from Mamba import MambaBlock
from TCN import TemporalConvNet


class TCNMamba(nn.Module):
    """TCN feature extractor followed by Mamba blocks and a linear head.

    Args:
        input_size: Number of input features per time step.
        output_size: Number of target outputs.
        tcn_channels: Output channels per TCN block; its length sets TCN depth.
            The last value becomes the model dimension fed to the Mamba blocks.
        kernel_size: TCN convolution kernel size.
        n_mamba_layers: Number of stacked Mamba blocks.
        d_state: Mamba SSM state expansion factor.
        expand: Mamba block expansion factor.
        headdim: Mamba head dimension; ``expand * tcn_channels[-1]`` must be
            divisible by it.
        dropout: Dropout probability used in both the TCN and Mamba parts.
        seq_to_seq: If True, predict at every time step; otherwise use the last step.

    Input/Output shape: input ``(batch, sequence_length, input_size)``; output
    ``(batch, output_size)`` when ``seq_to_seq`` is False, else
    ``(batch, sequence_length, output_size)``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tcn_channels: list[int],
        kernel_size: int = 3,
        n_mamba_layers: int = 4,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.2,
        seq_to_seq: bool = False,
    ):
        super().__init__()
        d_model = tcn_channels[-1]
        self.tcn = TemporalConvNet(
            input_size, tcn_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.mamba_layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    headdim=headdim,
                    dropout=dropout,
                    layer_idx=i,
                )
                for i in range(n_mamba_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, output_size)
        self.seq_to_seq = seq_to_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TCN expects (batch, channels, sequence_length); restore (batch, seq, feat).
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.mamba_layers:
            x = layer(x)
        x = self.norm(x)
        if self.seq_to_seq:
            return self.linear(x)
        return self.linear(x[:, -1])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch, seq_len, features = 16, 100, 8
    model = TCNMamba(
        input_size=features,
        output_size=1,
        tcn_channels=[64, 64, 128],
        kernel_size=3,
        n_mamba_layers=4,
    ).to(device)
    sample = torch.randn(batch, seq_len, features, device=device)
    out = model(sample)
    print(model)
    print(f"input:  {tuple(sample.shape)}")
    print(f"output: {tuple(out.shape)}")
