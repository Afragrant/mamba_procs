"""evaluate.py —— 在测试集上评估模型, 计算全部指标, 画预测对比图.

输出两套指标表: 池化口径(题述公式) 与 逐 case 口径(对标 Hu et al. 2025).

用法:
    python evaluate.py --ckpt result/checkpoints/cnn_mamba3.pt --data full
    python evaluate.py --ckpt result/checkpoints/lstm.pt --data full --n-plot 3
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


if __name__ == '__main__':
    main()
