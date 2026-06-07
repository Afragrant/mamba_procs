"""data_utils.py —— 数据加载、Min-Max 归一化、位置编码、torch Dataset/DataLoader.

归一化 (题述公式, 区间 [-1, 1]):
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
反归一化:
    x = (x_norm + 1) / 2 * (x_max - x_min) + x_min
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config as C


# --------------------------------------------------------------------------- #
# Min-Max 归一化 [-1, 1]
# --------------------------------------------------------------------------- #
def minmax_norm(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    rng = np.where((hi - lo) == 0, 1.0, hi - lo)
    return 2.0 * (x - lo) / rng - 1.0


def minmax_denorm(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    rng = hi - lo
    return (x + 1.0) / 2.0 * rng + lo


def denorm_targets(y_norm, y_min, y_max):
    """反归一化输出 (支持 numpy 或 torch)."""
    if isinstance(y_norm, torch.Tensor):
        y_min = torch.as_tensor(y_min, dtype=y_norm.dtype, device=y_norm.device)
        y_max = torch.as_tensor(y_max, dtype=y_norm.dtype, device=y_norm.device)
    return (y_norm + 1.0) / 2.0 * (y_max - y_min) + y_min


# --------------------------------------------------------------------------- #
# 位置编码: 1 线性通道 + 2*N_POS_FREQS 个 Fourier 通道, 形状 (T, N_POS)
# 所有模型共用, 使序列模型能生成沿线周期性响应 (对纯卷积 TCN 尤为必要).
# --------------------------------------------------------------------------- #
def build_positional_encoding(seq_len: int = C.SEQ_LEN, n_freqs: int = C.N_POS_FREQS) -> np.ndarray:
    if not C.POS_ENCODING:
        return np.zeros((seq_len, 0), dtype=np.float32)
    p01 = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    cols = [2.0 * p01 - 1.0]  # 线性, 映射到 [-1, 1]
    for k in range(n_freqs):
        f = np.pi * (2 ** k)
        cols.append(np.sin(f * p01))
        cols.append(np.cos(f * p01))
    return np.stack(cols, axis=-1).astype(np.float32)  # (T, 1+2*n_freqs)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CatenaryDataset(Dataset):
    """返回 (x_seq, y_seq):
       x_seq: (T, IN_CHANNELS)  = [5 归一化设计变量(沿T广播) | 位置编码]
       y_seq: (T, OUT_CHANNELS) = 3 个归一化输出序列
    """

    def __init__(self, design_norm: np.ndarray, targets_norm: np.ndarray, pos_enc: np.ndarray):
        self.design = torch.from_numpy(design_norm.astype(np.float32))      # (N, 5)
        self.targets = torch.from_numpy(targets_norm.astype(np.float32))    # (N, T, 3)
        self.pos = torch.from_numpy(pos_enc.astype(np.float32))             # (T, N_POS)
        self.T = self.targets.shape[1]

    def __len__(self):
        return self.design.shape[0]

    def __getitem__(self, i):
        d = self.design[i].unsqueeze(0).expand(self.T, -1)  # (T, 5)
        if self.pos.shape[1] > 0:
            x = torch.cat([d, self.pos], dim=-1)            # (T, 5+N_POS)
        else:
            x = d
        return x, self.targets[i]


# --------------------------------------------------------------------------- #
# 加载 npz -> 三个 DataLoader + 归一化统计量
# --------------------------------------------------------------------------- #
def load_dataset(npz_path, batch_size: int = C.BATCH_SIZE, num_workers: int = C.NUM_WORKERS):
    data = np.load(npz_path, allow_pickle=True)
    design = data['design']        # (N, 5) 原始物理值
    targets = data['targets']      # (N, T, 3) 原始物理值
    idx_tr, idx_va, idx_te = data['idx_train'], data['idx_val'], data['idx_test']
    x_min, x_max = data['x_min'], data['x_max']
    y_min, y_max = data['y_min'], data['y_max']

    design_n = minmax_norm(design, x_min, x_max).astype(np.float32)
    targets_n = minmax_norm(targets, y_min, y_max).astype(np.float32)
    pos = build_positional_encoding(targets.shape[1])

    def make(idx, shuffle):
        ds = CatenaryDataset(design_n[idx], targets_n[idx], pos)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True, drop_last=False)

    loaders = {
        'train': make(idx_tr, True),
        'val': make(idx_va, False),
        'test': make(idx_te, False),
    }
    stats = dict(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                 input_names=list(data['input_names']), output_names=list(data['output_names']))
    return loaders, stats
