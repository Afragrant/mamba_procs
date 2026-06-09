"""TCN (Temporal Convolutional Network) 基准模型.

膨胀因果卷积 + 残差块 (Bai et al. 2018). 序列到序列回归:
输入 (B, T, C_in) -> 输出 (B, T, C_out).
"""

from __future__ import annotations

import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """切除因果卷积右侧多余的 padding, 保持输出长度 = 输入长度."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, : -self.chomp] if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_in, n_out, kernel_size, padding=pad, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(n_out, n_out, kernel_size, padding=pad, dilation=dilation))
        self.net = nn.Sequential(
            self.conv1,
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            self.conv2,
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """膨胀 TCN 主干, 输入/输出均为 (B, C, T)."""

    def __init__(self, n_in, channels, kernel_size, dropout):
        super().__init__()
        layers = []
        prev = n_in
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(prev, ch, kernel_size, dilation=2**i, dropout=dropout))
            prev = ch
        self.network = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, x):  # (B, C, T)
        return self.network(x)


class TCNModel(nn.Module):
    """TCN 序列回归: (B, T, C_in) -> (B, T, C_out)."""

    def __init__(self, in_channels, out_channels, channels=(64, 64, 64, 64), kernel_size=5, dropout=0.1):
        super().__init__()
        self.tcn = TCN(in_channels, channels, kernel_size, dropout)
        self.head = nn.Conv1d(self.tcn.out_channels, out_channels, 1)

    def forward(self, x):  # (B, T, C_in)
        x = x.transpose(1, 2)  # (B, C_in, T)
        h = self.tcn(x)  # (B, C, T)
        y = self.head(h)  # (B, C_out, T)
        return y.transpose(1, 2)  # (B, T, C_out)
