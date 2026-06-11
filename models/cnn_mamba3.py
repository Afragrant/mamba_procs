"""提出模型: CNN-Mamba3 —— CNN 前端 + Mamba3 主干.

对标 Hu et al.(2025) 的 CNN-LSTM, 把 LSTM 换成更高效的 Mamba3 选择性状态空间模型:
  - CNN 前端: 几层 1D 卷积 (非膨胀、等长 padding) 提取局部特征 (区别于膨胀因果 TCN 基准);
  - Mamba3 主干: 残差 Mamba3 块建模沿线长程依赖;
  - 输出头: 线性映射到 3 个输出通道.

(B, T, C_in) -> (B, T, C_out). 需 CUDA.
"""

from __future__ import annotations

import torch.nn as nn
from mamba_ssm import Mamba3

from .mamba_model import Mamba3Block

# Optuna 搜索用: 卷积宽度/核大小的离散选项 (Optuna 类别参数只能存字符串 code,
# build_tuned 再据 code 还原成元组). tune.py 与 build_tuned 共用.
CNN_CHANNEL_OPTS = {'a': (64, 128, 64), 'b': (64, 64, 64), 'c': (32, 64, 128)}
CNN_KERNEL_OPTS = {'a': (5, 5, 3), 'b': (7, 5, 3), 'c': (3, 3, 3)}


class CNNFrontEnd(nn.Module):
    """等长 1D 卷积特征提取, 输入/输出均为 (B, C, T).

    每层 Conv -> BatchNorm -> ReLU -> Dropout: BatchNorm1d 稳定特征尺度、加速收敛.
    注: padding=k//2 为对称(双向)卷积; 本任务序列轴是空间位置且输入全已知,
    非时序预测, 故无"未来泄露"问题, 双向上下文反而更贴合梁的物理.
    """

    def __init__(self, in_channels, channels=(64, 128, 64), kernels=(5, 5, 3), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_channels
        for ch, k in zip(channels, kernels):
            # padding=k//2 仅对奇数核 + stride=1 保证等长输出; 偶数核会输出 T+1,
            # 破坏后续残差/对齐. 显式断言, fail-fast 而非静默错位.
            assert k % 2 == 1, f'卷积核必须为奇数 (等长卷积要求), 收到 k={k}'
            layers += [nn.Conv1d(prev, ch, k, padding=k // 2),
                       nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout)]
            prev = ch
        self.net = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, x):              # (B, C, T)
        return self.net(x)


class BiMamba3Block(nn.Module):
    """双向残差 Mamba3 块: x + Dropout(scan_fwd + scan_bwd).

    Mamba3 是严格因果(从左到右)扫描; 但本任务序列轴是梁的空间位置, 某点形变受
    左右两侧共同影响, 无因果约束. 故正向扫一遍 + 反向(flip)扫一遍再相加, 让每个
    位置都能获得全局双向上下文 (Vision/Bi-Mamba 思路), 消除单向感受野的物理失真.

    关键: 共享内层 norm 与单一外层残差 (x + ...), 不会因复用残差块而重复叠加 skip.
    tie_weights=True (默认): 正反向共享同一组 Mamba3 权重 -> 参数量与单向块逐字节相同,
        是隔离"双向性"这一因素最公平的消融 (单向 vs 双向唯一差异 = 是否反向再扫一遍),
        且零额外参数. tie_weights=False: 正反向各一组权重 (Vim 风格), 容量更大但消融
        会混入参数量因素.
    """

    def __init__(self, d_model, d_state, expand, headdim, dropout, layer_idx, tie_weights=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = Mamba3(d_model=d_model, d_state=d_state, expand=expand,
                          headdim=headdim, layer_idx=2 * layer_idx)
        self.bwd = self.fwd if tie_weights else Mamba3(
            d_model=d_model, d_state=d_state, expand=expand,
            headdim=headdim, layer_idx=2 * layer_idx + 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                        # (B, T, d_model)
        h = self.norm(x)
        y_f = self.fwd(h)
        y_b = self.bwd(h.flip(1)).flip(1)        # 反向扫描后翻回原序
        return x + self.dropout(y_f + y_b)       # 单一残差, 不重复叠加


class CNNMamba3(nn.Module):
    def __init__(self, in_channels, out_channels, cnn_channels=(64, 128, 64),
                 cnn_kernels=(5, 5, 3), d_model=128, n_mamba=4, d_state=64,
                 expand=2, headdim=64, dropout=0.1, bidirectional=True, tie_weights=True):
        super().__init__()
        self.cnn = CNNFrontEnd(in_channels, cnn_channels, cnn_kernels, dropout)
        # 投影到 Mamba 维度: Linear -> GELU -> Dropout (加非线性, 缓解信息瓶颈)
        self.proj = nn.Linear(self.cnn.out_channels, d_model)
        self.proj_act = nn.GELU()
        self.proj_drop = nn.Dropout(dropout)
        # 默认双向扫描(权重共享): 空间轴无因果约束, 给每个位置全局双向上下文, 且参数量
        # 与单向版逐字节相同 (见 BiMamba3Block), 消融可干净隔离"双向性".
        if bidirectional:
            self.blocks = nn.ModuleList([
                BiMamba3Block(d_model, d_state, expand, headdim, dropout,
                              layer_idx=i, tie_weights=tie_weights)
                for i in range(n_mamba)
            ])
        else:
            self.blocks = nn.ModuleList([
                Mamba3Block(d_model, d_state, expand, headdim, dropout, layer_idx=i)
                for i in range(n_mamba)
            ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_channels)

    def forward(self, x):                       # (B, T, C_in)
        h = self.cnn(x.transpose(1, 2))         # (B, C, T)
        h = h.transpose(1, 2)                   # (B, T, C)
        h = self.proj_drop(self.proj_act(self.proj(h)))   # (B, T, d_model)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))


def build_cnn_mamba3(in_channels, out_channels, **kwargs):
    return CNNMamba3(in_channels, out_channels, **kwargs)
