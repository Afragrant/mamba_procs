"""Mamba sequence model.

A stack of Mamba3 selective state-space blocks (Mamba3 from the ``mamba_ssm``
package) with input projection and a linear head, mirroring the interface of
the TCN model so the two are interchangeable for sequence tasks.
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba3


class MambaBlock(nn.Module):
    """Pre-norm residual wrapper around a single Mamba3 mixer."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.0,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            layer_idx=layer_idx,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual: x + Mixer(Norm(x)).
        return x + self.dropout(self.mixer(self.norm(x)))


class Mamba(nn.Module):
    """Mamba model with a linear head for sequence tasks.

    Args:
        input_size: Number of input features per time step.
        output_size: Number of target outputs.
        d_model: Hidden dimension of the Mamba blocks. With ``expand`` the inner
            width is ``expand * d_model`` and must be divisible by ``headdim``.
        n_layers: Number of stacked Mamba blocks (network depth).
        d_state: SSM state expansion factor.
        expand: Block expansion factor.
        headdim: Head dimension; ``expand * d_model`` must be divisible by it.
        dropout: Dropout probability on each block's residual branch.
        seq_to_seq: If True, predict at every time step; otherwise use the last step.

    Input/Output shape: input ``(batch, sequence_length, input_size)``; output
    ``(batch, output_size)`` when ``seq_to_seq`` is False, else
    ``(batch, sequence_length, output_size)``.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        dropout: float = 0.0,
        seq_to_seq: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    headdim=headdim,
                    dropout=dropout,
                    layer_idx=i,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, output_size)
        self.seq_to_seq = seq_to_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if self.seq_to_seq:
            return self.linear(x)
        return self.linear(x[:, -1])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch, seq_len, features = 16, 100, 8
    model = Mamba(
        input_size=features,
        output_size=1,
        d_model=128,
        n_layers=4,
    ).to(device)
    sample = torch.randn(batch, seq_len, features, device=device)
    out = model(sample)
    print(model)
    print(f"input:  {tuple(sample.shape)}")
    print(f"output: {tuple(out.shape)}")
