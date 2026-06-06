"""Validate a single trained surrogate on the dataset's validation split.

Reports per-channel RMSE / MAE / NRMSE / R² in physical units and saves an
overlay plot (prediction vs ground truth) for a few validation samples.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import PCData, compute_metrics
from factory import build_model
from pc import _use_cjk_font


def load_model(ckpt_path: str, device: str):
    """Rebuild a model from a checkpoint saved by train.py."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Rebuild with the checkpoint's (possibly tuned) architecture so weights match.
    model = build_model(ckpt["model_name"], ckpt["input_size"], ckpt["output_size"],
                        seq_to_seq=True, hp=ckpt["config"].get("arch_hp"))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict(model, x, device, denorm_fn) -> np.ndarray:
    """Forward pass → physical-unit predictions (N, seq_len, 3)."""
    pred = model(x.to(device)).cpu().numpy()
    return denorm_fn(pred)


def plot_samples(pos, pred, true, names, idxs, title, out_path):
    cjk = _use_cjk_font()
    units = ["[N]", "[m]", "[m]"]
    labels = (["弓网接触力", "弓头位移", "接触网位移"] if cjk else names)
    pred_lbl, true_lbl = ("预测", "物理模型") if cjk else ("Prediction", "Physics")
    xlabel = "归一化位置" if cjk else "Normalized position"

    fig, axes = plt.subplots(len(names), len(idxs), figsize=(4 * len(idxs), 7), squeeze=False)
    for r in range(len(names)):
        for c, s in enumerate(idxs):
            ax = axes[r, c]
            ax.plot(pos, true[s, :, r], color="k", lw=1.4, label=true_lbl)
            ax.plot(pos, pred[s, :, r], color="C3", lw=1.0, ls="--", label=pred_lbl)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel(f"{labels[r]} {units[r]}")
            if r == 0:
                ax.set_title(f"sample {s}")
            if r == len(names) - 1:
                ax.set_xlabel(xlabel)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="./result/pc_dataset.npz")
    ap.add_argument("--n-plot", type=int, default=3)
    ap.add_argument("--out", default="./result/validation")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = PCData(args.data)
    model, ckpt = load_model(args.ckpt, device)
    name = ckpt["model_name"]

    pred = predict(model, data.x_val, device, data.denorm_y)
    true = data.Y_val_raw
    metrics = compute_metrics(pred, true, data.target_names)

    print(f"=== Validation: {name}  (val N={len(true)}) ===")
    for n in data.target_names:
        m = metrics[n]
        print(f"  {n:14s}  RMSE={m['rmse']:.4g}  MAE={m['mae']:.4g}  NRMSE={m['nrmse']:.4f}  R²={m['r2']:.4f}")
    print(f"  {'macro':14s}  NRMSE={metrics['_macro']['nrmse']:.4f}  R²={metrics['_macro']['r2']:.4f}")

    n_plot = min(args.n_plot, len(true))
    idxs = np.linspace(0, len(true) - 1, n_plot).astype(int)
    path = plot_samples(
        data.pos, pred, true, data.target_names, idxs,
        title=f"{name} — validation prediction vs physics",
        out_path=Path(args.out) / f"validate_{name}.png",
    )
    print(f"Figure saved → {path}")


if __name__ == "__main__":
    main()
