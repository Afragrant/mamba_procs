"""提出模型: PC-TCN-Mamba (MCd) —— 物理约束 (Physics-Constrained) TCN-Mamba + MC Dropout.

与 TCNMambaMC 共享主干 (TCN 前端 + Mamba + MC Dropout), 区别只在输出层:
主干输出 2 个归一化通道 (gap_n, y_cat_n), 再由 pc.py 的精确本构关系硬推导出
物理三通道 (与其它模型同接口):
    gap = y_panto - y_cat
    y_panto = gap + y_cat
    Fc = CONTACT_KC * relu(gap)          (CONTACT_KC = 82300)

设计意义:
  - 直接预测 gap (无抵消量), 治理 "Fc = 大位移之差" 的灾难性抵消;
  - relu 把接触/离线开关精确编码: Fc≥0, 离线 (gap≤0) 时 Fc 恒为 0;
  - 三个物理量天然一致 (由同一 gap 推出), 无需额外一致性损失;
  - 对外仍输出归一化的 [Fc, y_panto, y_cat] 三通道, train/evaluate/指标/MC 全部复用.

forward 需要各通道与 gap 的 Min-Max 统计量, 通过 set_norm_stats() 注入 (存为 buffer,
随权重一并保存). (B, T, C_in) -> (B, T, 3). 需 CUDA.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config as C

from .mamba_model import MambaBlock
from .tcn import TCN


class PCTCNMambaMC(nn.Module):
    def __init__(self, in_channels, out_channels=3, tcn_channels=(64, 64), tcn_kernel=5,
                 d_model=128, n_mamba=4, d_state=16, d_conv=4, expand=2, dropout=0.15, kc=None):
        super().__init__()
        assert out_channels == 3, 'PC 模型对外固定输出 3 个物理通道'
        self.mc_p = dropout
        self.kc = float(C.CONTACT_KC if kc is None else kc)

        self.tcn = TCN(in_channels, tcn_channels, tcn_kernel, dropout)
        self.proj = nn.Linear(self.tcn.out_channels, d_model)
        self.proj_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout, layer_idx=i)
            for i in range(n_mamba)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 2)   # -> (gap_n, y_cat_n)

        # 归一化常量 (Min-Max 边界), 由 set_norm_stats 填充; 默认恒等变换
        for nm in ('gap', 'fc', 'yp', 'yc'):
            self.register_buffer(f'{nm}_lo', torch.tensor(0.0))
            self.register_buffer(f'{nm}_hi', torch.tensor(1.0))

    # ----------------------------------------------------------------- #
    def set_norm_stats(self, stats: dict):
        """注入 Min-Max 统计量. stats 需含 y_min/y_max (Fc,y_panto,y_cat) 与 gap_min/gap_max."""
        ymin, ymax = stats['y_min'], stats['y_max']
        def t(v):
            return torch.as_tensor(float(v))
        self.fc_lo.copy_(t(ymin[0])); self.fc_hi.copy_(t(ymax[0]))
        self.yp_lo.copy_(t(ymin[1])); self.yp_hi.copy_(t(ymax[1]))
        self.yc_lo.copy_(t(ymin[2])); self.yc_hi.copy_(t(ymax[2]))
        self.gap_lo.copy_(t(stats['gap_min'])); self.gap_hi.copy_(t(stats['gap_max']))

    @staticmethod
    def _denorm(n, lo, hi):
        return (n + 1.0) / 2.0 * (hi - lo) + lo

    @staticmethod
    def _norm(x, lo, hi):
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    # ----------------------------------------------------------------- #
    def _backbone(self, x):                # (B, T, C_in) -> (B, T, 2)
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        h = self.proj_drop(self.proj(h))
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return self.head(self.head_drop(h))

    def forward(self, x):
        out2 = self._backbone(x)
        gap_n, ycat_n = out2[..., 0], out2[..., 1]
        gap = self._denorm(gap_n, self.gap_lo, self.gap_hi)
        ycat = self._denorm(ycat_n, self.yc_lo, self.yc_hi)
        # 物理硬约束推导
        y_panto = gap + ycat
        Fc = self.kc * torch.relu(gap)
        # 还原成归一化的 [Fc, y_panto, y_cat] 三通道 (与其它模型同接口)
        Fc_n = self._norm(Fc, self.fc_lo, self.fc_hi)
        yp_n = self._norm(y_panto, self.yp_lo, self.yp_hi)
        return torch.stack([Fc_n, yp_n, ycat_n], dim=-1)

    # ----------------------------------------------------------------- #
    # MC Dropout (与 TCNMambaMC 一致); 对 gap 采样经 relu 推导, 天然给出
    # 物理一致、非负的接触力不确定带.
    # ----------------------------------------------------------------- #
    def enable_mc_dropout(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def mc_predict(self, x, n_samples: int = 50):
        self.enable_mc_dropout()
        preds = torch.stack([self(x) for _ in range(n_samples)], dim=0)
        return preds.mean(0), preds.std(0)


def build_pc_tcn_mamba(in_channels, out_channels, **kwargs):
    return PCTCNMambaMC(in_channels, out_channels, **kwargs)
