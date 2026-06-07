"""metrics.py —— 性能验证指标 (严格按题述公式实现).

约定: y_true / y_pred 形状 (N, T, C) 或 (M, C), 为反归一化后的物理量.
逐输出通道计算, 并给出 'overall' (全通道展平) 汇总.

公式 (注: 题述命名与文献惯例略有出入, 此处忠实于题述):
  RRMSE = sqrt( (1/n) Σ (yR - yP)^2 )                         # 题述 "RRMSE"
  R^2   = 1 - Σ(yR-yP)^2 / Σ(yR - mean(yR))^2
  RDE_i = (i_pred - i_sim) / i_pred * 100%,  i ∈ {Std, Ave}   # 标准差/均值相对偏差
  RMSE  = (1/n) Σ (p - p~)^2                                   # 题述 "RMSE" (实为 MSE)
  RMAE  = (1/n) Σ |p - p~|                                     # 题述 "RMAE" (实为 MAE)
  Erel  = |p - p~| / |p|                                       # 逐点相对误差, 报告其均值
"""

from __future__ import annotations

import numpy as np

import config as C

_EPS = 1e-12


def _flat(y):
    return np.asarray(y, dtype=np.float64).reshape(-1)


def rrmse(y_true, y_pred) -> float:
    yt, yp = _flat(y_true), _flat(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2_score(y_true, y_pred) -> float:
    yt, yp = _flat(y_true), _flat(y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + _EPS))


def rde_std(y_true, y_pred) -> float:
    """标准差相对偏差 (%) = (std_pred - std_sim)/std_pred * 100."""
    s_pred = _flat(y_pred).std()
    s_sim = _flat(y_true).std()
    return float((s_pred - s_sim) / (s_pred + _EPS) * 100.0)


def rde_ave(y_true, y_pred) -> float:
    """均值相对偏差 (%) = (mean_pred - mean_sim)/mean_pred * 100."""
    m_pred = _flat(y_pred).mean()
    m_sim = _flat(y_true).mean()
    return float((m_pred - m_sim) / (m_pred + _EPS) * 100.0)


def rmse_msq(y_true, y_pred) -> float:
    """题述 RMSE = (1/n) Σ (p - p~)^2."""
    yt, yp = _flat(y_true), _flat(y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmae(y_true, y_pred) -> float:
    """题述 RMAE = (1/n) Σ |p - p~|."""
    yt, yp = _flat(y_true), _flat(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def erel_mean(y_true, y_pred) -> float:
    """逐点相对误差 Erel = |p - p~|/|p| 的均值 (跳过 |p|≈0 的点)."""
    yt, yp = _flat(y_true), _flat(y_pred)
    mask = np.abs(yt) > _EPS
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / np.abs(yt[mask])))


_METRIC_FNS = {
    'RRMSE': rrmse,
    'R2': r2_score,
    'RDE_Std%': rde_std,
    'RDE_Ave%': rde_ave,
    'RMSE': rmse_msq,
    'RMAE': rmae,
    'Erel': erel_mean,
}


def compute_all(y_true: np.ndarray, y_pred: np.ndarray, channel_names=None) -> dict:
    """返回 {通道名: {指标: 值}}, 含 'overall'.

    y_true / y_pred: (N, T, C). channel_names 默认 config.OUTPUT_NAMES.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if channel_names is None:
        channel_names = C.OUTPUT_NAMES
    nC = y_true.shape[-1]
    out = {}
    for c in range(nC):
        name = channel_names[c]
        yt, yp = y_true[..., c], y_pred[..., c]
        out[name] = {m: fn(yt, yp) for m, fn in _METRIC_FNS.items()}
    out['overall'] = {m: fn(y_true, y_pred) for m, fn in _METRIC_FNS.items()}
    return out


def format_table(results: dict) -> str:
    """把 compute_all 的结果格式化成对齐表格字符串."""
    metrics = list(_METRIC_FNS.keys())
    rows = list(results.keys())
    w = max(len(r) for r in rows) + 2
    header = 'channel'.ljust(w) + ''.join(m.rjust(12) for m in metrics)
    lines = [header, '-' * len(header)]
    for r in rows:
        line = r.ljust(w) + ''.join(f'{results[r][m]:12.5g}' for m in metrics)
        lines.append(line)
    return '\n'.join(lines)
