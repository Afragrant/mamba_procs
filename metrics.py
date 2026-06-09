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


def _safe_den(pred, ref):
    """分母下限保护: 题述 RDE 以"预测值"为分母, 当预测统计量≈0 时会爆炸.
    用仿真值 ref 的一个小比例做下限 (保号), 把退化情形限制在合理量级.
    """
    floor = 1e-3 * abs(ref) + _EPS
    if abs(pred) >= floor:
        return pred
    return floor if pred >= 0 else -floor


def rde_std(y_true, y_pred) -> float:
    """标准差相对偏差 (%) = (std_pred - std_sim)/std_pred * 100."""
    s_pred = _flat(y_pred).std()
    s_sim = _flat(y_true).std()
    return float((s_pred - s_sim) / _safe_den(s_pred, s_sim) * 100.0)


def rde_ave(y_true, y_pred) -> float:
    """均值相对偏差 (%) = (mean_pred - mean_sim)/mean_pred * 100."""
    m_pred = _flat(y_pred).mean()
    m_sim = _flat(y_true).mean()
    return float((m_pred - m_sim) / _safe_den(m_pred, m_sim) * 100.0)


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


# --------------------------------------------------------------------------- #
# 逐 case (per-case) 指标 —— 对标 Hu et al.(2025) 的口径
#   论文 Table 6: 对每个 case 单独算 RMSE/R², 再对所有 case 求 均值 与 标准差.
#   论文 Table 7: 对每个 case 算 RDE_Std / RDE_Ave, 报告所有 case 的 Max% 与 Ave%.
#   注: 论文 Eq.29 的 RMSE 含开方 (标准 RMSE), 此处逐 case 用标准 RMSE 以与论文一致;
#       与 compute_all 中"题述口径"(RMSE=MSE) 并存, 互不影响.
# --------------------------------------------------------------------------- #
def compute_percase(y_true: np.ndarray, y_pred: np.ndarray, channel_names=None) -> dict:
    """y_true / y_pred: (N, T, C). 返回 {通道: {指标统计}} (全向量化)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if channel_names is None:
        channel_names = C.OUTPUT_NAMES
    N, T, nC = y_true.shape
    out = {}
    for c in range(nC):
        a = y_true[:, :, c]        # (N, T)
        b = y_pred[:, :, c]
        ss_res = np.sum((a - b) ** 2, axis=1)
        ss_tot = np.sum((a - a.mean(axis=1, keepdims=True)) ** 2, axis=1)
        r2 = 1.0 - ss_res / (ss_tot + _EPS)
        rmse = np.sqrt(np.mean((a - b) ** 2, axis=1))   # 标准 RMSE (含开方, 同论文)
        rmae = np.mean(np.abs(a - b), axis=1)
        sp, ss = b.std(axis=1), a.std(axis=1)
        mp, ms = b.mean(axis=1), a.mean(axis=1)
        # 分母下限保护 (同 _safe_den, 向量化): 防止预测统计量≈0 时 RDE 爆炸
        den_s = np.where(np.abs(sp) >= 1e-3 * np.abs(ss) + _EPS, sp,
                         np.copysign(1e-3 * np.abs(ss) + _EPS, np.where(sp >= 0, 1.0, -1.0)))
        den_m = np.where(np.abs(mp) >= 1e-3 * np.abs(ms) + _EPS, mp,
                         np.copysign(1e-3 * np.abs(ms) + _EPS, np.where(mp >= 0, 1.0, -1.0)))
        rde_std = (sp - ss) / den_s * 100.0
        rde_ave = (mp - ms) / den_m * 100.0
        out[channel_names[c]] = {
            'R2_mean': float(r2.mean()), 'R2_std': float(r2.std()),
            'RMSE_mean': float(rmse.mean()), 'RMSE_std': float(rmse.std()),
            'RMAE_mean': float(rmae.mean()),
            'RDE_Std_Max%': float(np.abs(rde_std).max()), 'RDE_Std_Ave%': float(np.abs(rde_std).mean()),
            'RDE_Ave_Max%': float(np.abs(rde_ave).max()), 'RDE_Ave_Ave%': float(np.abs(rde_ave).mean()),
        }
    return out


def format_percase(results: dict) -> str:
    cols = ['R2_mean', 'R2_std', 'RMSE_mean', 'RMSE_std',
            'RDE_Std_Max%', 'RDE_Std_Ave%', 'RDE_Ave_Max%', 'RDE_Ave_Ave%']
    w = max(len(r) for r in results) + 2
    header = 'channel'.ljust(w) + ''.join(c.rjust(13) for c in cols)
    lines = [header, '-' * len(header)]
    for r in results:
        lines.append(r.ljust(w) + ''.join(f'{results[r][c]:13.5g}' for c in cols))
    return '\n'.join(lines)
