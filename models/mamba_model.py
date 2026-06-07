"""Mamba 基准模型 (基于 mamba_ssm 库).

输入投影到 d_model, 堆叠 N 个残差 Mamba 块, 再投影到输出通道.
(B, T, C_in) -> (B, T, C_out). 需 CUDA.
"""

from __future__ import annotations

import torch.nn as nn
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """残差: x + Mamba(LayerNorm(x)), 后接 Dropout."""

    def __init__(self, d_model, d_state, d_conv, expand, dropout, layer_idx):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv,
                           expand=expand, layer_idx=layer_idx)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class MambaModel(nn.Module):
    def __init__(self, in_channels, out_channels, d_model=128, n_layers=4,
                 d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout, layer_idx=i)
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_channels)

    def forward(self, x):            # (B, T, C_in)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))
