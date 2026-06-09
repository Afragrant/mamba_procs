"""evaluate.py —— 在测试集上评估模型, 计算全部指标, 画对比/不确定性图.

对提出模型 (tcn_mamba) 额外执行蒙特卡洛 Dropout 不确定性量化:
多次前向得到预测均值与标准差, 给出 95% 置信带与覆盖率.

用法:
    python evaluate.py --ckpt result/checkpoints/tcn_mamba.pt --data smoke
    python evaluate.py --ckpt result/checkpoints/lstm.pt --data smoke --n-plot 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import config as C
import metrics as M
from data_utils import denorm_targets, load_dataset
from models import build_model, build_tuned
from pc import _use_cjk_font


def resolve_data(arg: str):
    if arg == 'smoke':
        return C.DATASET_SMOKE
    if arg == 'full':
        return C.DATASET_FULL
    return arg


@torch.no_grad()
def collect_predictions(model, loader, device, y_min, y_max):
    """返回反归一化后的 (y_true, y_pred), 形状 (N, T, C)."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu()
        ys.append(denorm_targets(y, y_min, y_max).numpy())
        ps.append(denorm_targets(pred, y_min, y_max).numpy())
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def collect_mc(model, loader, device, y_min, y_max, n_samples):
    """MC Dropout: 返回 (y_true, mean, std) 反归一化后, 形状 (N, T, C)."""
    ys, means, stds = [], [], []
    yr = (np.asarray(y_max) - np.asarray(y_min)) / 2.0  # 反归一化对 std 的缩放
    for x, y in loader:
        x = x.to(device)
        m_n, s_n = model.mc_predict(x, n_samples=n_samples)
        ys.append(denorm_targets(y, y_min, y_max).numpy())
        means.append(denorm_targets(m_n.cpu(), y_min, y_max).numpy())
        stds.append(s_n.cpu().numpy() * yr)  # std 仅按尺度缩放, 不加偏移
    return np.concatenate(ys), np.concatenate(means), np.concatenate(stds)


def plot_predictions(y_true, y_pred, names, out_path, n_plot=3, std=None, band_mult=2.0):
    """band_mult: 置信带半宽 = band_mult · σ; 可为标量或逐通道数组 (标定后用)."""
    cjk = _use_cjk_font()
    T = y_true.shape[1]
    x = np.arange(T)
    n_plot = min(n_plot, y_true.shape[0])
    nC = y_true.shape[-1]
    mult = np.broadcast_to(np.asarray(band_mult, dtype=float), (nC,))
    fig, axes = plt.subplots(n_plot, nC, figsize=(4.5 * nC, 2.6 * n_plot), squeeze=False)
    for r in range(n_plot):
        for c in range(nC):
            ax = axes[r, c]
            ax.plot(x, y_true[r, :, c], color='C0', lw=1.0,
                    label='仿真' if cjk else 'Simulation')
            ax.plot(x, y_pred[r, :, c], color='C1', lw=1.0, ls='--',
                    label='预测' if cjk else 'Prediction')
            if std is not None:
                half = mult[c] * std[r, :, c]
                ax.fill_between(x, y_pred[r, :, c] - half, y_pred[r, :, c] + half,
                                color='C1', alpha=0.2,
                                label='95% 置信带' if cjk else '95% CI')
            if r == 0:
                ax.set_title(names[c])
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'图已保存: {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='smoke')
    ap.add_argument('--device', default=C.DEVICE)
    ap.add_argument('--n-plot', type=int, default=3)
    ap.add_argument('--mc-samples', type=int, default=C.MC_SAMPLES)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_name = ckpt['model_name']
    # 若权重来自 tune.py (含 best_params), 用相同的调优结构重建; 否则用默认结构.
    if ckpt.get('best_params'):
        print(f'检测到调优超参, 按 best_params 重建结构: {ckpt["best_params"]}')
        model = build_tuned(model_name, ckpt['best_params']).to(device)
    else:
        model = build_model(model_name).to(device)
    model.load_state_dict(ckpt['state_dict'])

    loaders, stats = load_dataset(resolve_data(args.data),
                                  batch_size=C.BATCH_SIZE, num_workers=2)
    if hasattr(model, 'set_norm_stats'):   # 物理约束模型需注入 Min-Max 统计量
        model.set_norm_stats(stats)
    y_min, y_max = stats['y_min'], stats['y_max']
    names = stats['output_names']

    print(f'\n评估模型: {model_name}  (测试集)')
    y_true, y_pred = collect_predictions(model, loaders['test'], device, y_min, y_max)

    print('\n[池化口径 / 题述公式] 全测试集逐通道:')
    results = M.compute_all(y_true, y_pred, names)
    print(M.format_table(results))

    print('\n[逐 case 口径 / 对标 Hu et al. 2025] 每 case 单独算再统计:')
    pc = M.compute_percase(y_true, y_pred, names)
    print(M.format_percase(pc))

    out_dir = Path(C.RESULT_DIR) / 'eval'
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f'{model_name}_metrics.npz',
             **{f'{ch}__{m}': results[ch][m] for ch in results for m in results[ch]},
             **{f'percase__{ch}__{m}': pc[ch][m] for ch in pc for m in pc[ch]})
    plot_predictions(y_true, y_pred, names, out_dir / f'{model_name}_pred.png',
                     n_plot=args.n_plot)

    # 提出模型: 蒙特卡洛 Dropout 不确定性量化 (含验证集后验标定)
    if hasattr(model, 'mc_predict'):
        nC = len(names)
        target = C.UQ_TARGET_COVERAGE
        z = 1.959963985  # 标准正态 95% 双侧分位

        # 1) 在验证集上求每通道标定因子 k, 使 (均值 ± k·σ) 的覆盖率≈target.
        #    用"覆盖率二分搜索 + σ下限"代替 |y-mean|/σ 分位: 物理约束模型在离线区
        #    Fc 恒为 0 且 MC 方差为 0, 直接除 σ 会爆炸; 此法数值稳健且不除 σ.
        if C.CALIBRATE_UQ:
            print(f'\n[MC 标定] 验证集 {args.mc_samples} 次采样, 目标覆盖率 {target:.0%} ...')
            yv, mv, sv = collect_mc(model, loaders['val'], device, y_min, y_max, args.mc_samples)
            k = np.empty(nC)
            for c in range(nC):
                scale = float(np.asarray(y_max)[c] - np.asarray(y_min)[c])
                sfloor = sv[..., c] + 1e-4 * scale          # σ 下限, 防止除 0/极端
                resid = np.abs(yv[..., c] - mv[..., c])
                lo, hi = 0.0, 1.0e6
                for _ in range(60):                          # 二分: 覆盖率关于 k 单调
                    mid = 0.5 * (lo + hi)
                    if np.mean(resid <= mid * sfloor) < target:
                        lo = mid
                    else:
                        hi = mid
                k[c] = hi
            print('  逐通道标定因子 k =', {names[c]: round(float(k[c]), 3) for c in range(nC)})
        else:
            k = np.full(nC, z)

        # 2) 测试集采样与评估
        print(f'\n[MC Dropout] 测试集 {args.mc_samples} 次前向采样 ...')
        yt, mean, std = collect_mc(model, loaders['test'], device, y_min, y_max, args.mc_samples)
        mc_results = M.compute_all(yt, mean, names)
        print('MC 均值预测指标:')
        print(M.format_table(mc_results))

        # 3) 标定前(±zσ) vs 标定后(±kσ) 的覆盖率与平均带宽 (MPIW)
        print(f'  {"通道":<10}{"标定前覆盖":>12}{"标定后覆盖":>12}{"平均σ":>12}{"标定后半宽":>14}')
        for c, nm in enumerate(names):
            cov0 = float(np.mean(np.abs(yt[..., c] - mean[..., c]) <= z * std[..., c]))
            cov1 = float(np.mean(np.abs(yt[..., c] - mean[..., c]) <= k[c] * std[..., c]))
            mpiw = float(np.mean(2 * k[c] * std[..., c]))
            print(f'  {nm:<10}{cov0:>12.3f}{cov1:>12.3f}{std[..., c].mean():>12.4g}{mpiw:>14.4g}')

        plot_predictions(yt, mean, names, out_dir / f'{model_name}_mc_uq.png',
                         n_plot=args.n_plot, std=std, band_mult=k)
        np.savez(out_dir / f'{model_name}_uq.npz', calib_k=k, target=target,
                 mean_sigma=std.reshape(-1, nC).mean(0))


if __name__ == '__main__':
    main()
