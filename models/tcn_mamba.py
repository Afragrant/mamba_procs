"""提出模型: TCN + Mamba + 蒙特卡洛 Dropout (TCNMambaMC).

设计动机:
  - TCN 前端: 膨胀因果卷积提取局部 (跨内/接头附近) 的快速波动特征;
  - Mamba 主干: 选择性状态空间模型, 高效建模沿线长程依赖 (跨间周期、累积效应);
  - MC Dropout: 推理时保持 Dropout 激活, 多次前向采样近似贝叶斯后验, 给出
    弓网接触力预测的均值与置信区间 (不确定性量化), 服务可靠性评估.

(B, T, C_in) -> (B, T, C_out). 需 CUDA (mamba_ssm).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .mamba_model import MambaBlock
from .tcn import TCN


class TCNMambaMC(nn.Module):
    def __init__(self, in_channels, out_channels, tcn_channels=(64, 64), tcn_kernel=5,
                 d_model=128, n_mamba=4, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.mc_p = dropout
        # 前端 TCN
        self.tcn = TCN(in_channels, tcn_channels, tcn_kernel, dropout)
        # 投影到 Mamba 维度
        self.proj = nn.Linear(self.tcn.out_channels, d_model)
        self.proj_drop = nn.Dropout(dropout)
        # Mamba 主干
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout, layer_idx=i)
            for i in range(n_mamba)
        ])
        self.norm = nn.LayerNorm(d_model)
        # MC Dropout 头
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, out_channels)

    def forward(self, x):                  # (B, T, C_in)
        h = self.tcn(x.transpose(1, 2))    # (B, C, T)
        h = h.transpose(1, 2)              # (B, T, C)
        h = self.proj_drop(self.proj(h))   # (B, T, d_model)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return self.head(self.head_drop(h))

    # ----------------------------------------------------------------- #
    # 蒙特卡洛 Dropout 推理
    # ----------------------------------------------------------------- #
    def enable_mc_dropout(self):
        """仅将 Dropout 层切到 train 模式 (保持其余层 eval), 以便 MC 采样."""
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def mc_predict(self, x, n_samples: int = 50):
        """返回 (mean, std): 形状均为 (B, T, C_out). 在归一化空间给出预测分布."""
        self.enable_mc_dropout()
        preds = torch.stack([self(x) for _ in range(n_samples)], dim=0)  # (S, B, T, C)
        return preds.mean(0), preds.std(0)


def build_tcn_mamba(in_channels, out_channels, **kwargs):
    return TCNMambaMC(in_channels, out_channels, **kwargs)
