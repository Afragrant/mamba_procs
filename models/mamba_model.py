"""Mamba3 基准模型 (基于 mamba_ssm 的 Mamba3).

输入投影到 d_model, 堆叠 N 个残差 Mamba3 块, 再投影到输出通道.
(B, T, C_in) -> (B, T, C_out). 需 CUDA.

注: Mamba3 结构参数为 d_model / d_state / expand / headdim (无 d_conv).
"""

from __future__ import annotations

import torch.nn as nn
from mamba_ssm import Mamba3


class Mamba3Block(nn.Module):
    """残差: x + Mamba3(LayerNorm(x)), 后接 Dropout."""

    def __init__(self, d_model, d_state, expand, headdim, dropout, layer_idx):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba3(d_model=d_model, d_state=d_state, expand=expand,
                            headdim=headdim, layer_idx=layer_idx)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class Mamba3Model(nn.Module):
    def __init__(self, in_channels, out_channels, d_model=128, n_layers=4,
                 d_state=64, expand=2, headdim=64, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, d_model)
        self.blocks = nn.ModuleList([
            Mamba3Block(d_model, d_state, expand, headdim, dropout, layer_idx=i)
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_channels)

    def forward(self, x):            # (B, T, C_in)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))
