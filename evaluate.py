"""Compare all four surrogates and benchmark them against the physical model.

For each model (TCN / LSTM / Mamba / TCN-Mamba) this script:
  * loads its checkpoint, or trains it on the fly with ``--train``,
  * scores it on the validation split (RMSE / MAE / NRMSE / R², physical units),
  * times its inference (per sample, GPU-synchronized),
and then:
  * replays the physical model (``pc.run_simulation``) for one validation case
    and times it, overlaying every surrogate's prediction on the physics output,
  * reports the accuracy ranking (lower macro-NRMSE is better) and the
    inference-time speed-up of each surrogate over the physics solver.

The model with the lowest macro-NRMSE is declared best; the goal is to show
TCN-Mamba both matches the physics and is orders of magnitude faster.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import PCData, compute_metrics
from factory import MODELS, build_model
from pc import STABLE_END, STABLE_START, _use_cjk_font, rigid_overhead_contact_system_params, run_simulation
from train import save_checkpoint, train_model

PRESETS = [1, 2, 3, 4]


def find_preset(x_row: np.ndarray) -> int:
    """Recover which catenary preset an input feature row came from (L,rhoA,EI,KEQ,MEQ)."""
    for p in PRESETS:
        L, _N, rhoA, EI, KEQ, MEQ, _MZ, _L_MZ = rigid_overhead_contact_system_params(p)
        if np.allclose(x_row[:5], [L, rhoA, EI, KEQ, MEQ], rtol=1e-3, atol=1e-6):
            return p
    raise ValueError("input row does not match any preset")


def _resample(arr: np.ndarray, seq_len: int) -> np.ndarray:
    xp = np.linspace(0.0, 1.0, len(arr))
    return np.interp(np.linspace(0.0, 1.0, seq_len), xp, arr)


def physics_reference(preset: int, speed: float, nm: int, dt: float, seq_len: int):
    """Run the physical model for one case → (targets (seq_len,3), wall_time_s)."""
    t0 = time.perf_counter()
    res = run_simulation(
        rigid_overhead_contact_system=preset, pantograph=1, speed_kmh=speed,
        NM=nm, dt_base=dt, verbose=False,
    )
    wall = time.perf_counter() - t0
    y = np.stack(
        [
            _resample(res["contact_force_stable"], seq_len),
            _resample(res["y_pantograph_stable"], seq_len),
            _resample(res["y_rigid_overhead_contact_system_stable"], seq_len),
        ],
        axis=1,
    )
    return y, wall


@torch.no_grad()
def time_inference(model, x, device, reps: int = 50) -> float:
    """Per-sample inference time in seconds (GPU-synchronized, warmed up)."""
    model.eval()
    x = x.to(device)
    for _ in range(3):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / (reps * x.shape[0])


@torch.no_grad()
def predict(model, x, device, denorm_fn) -> np.ndarray:
    return denorm_fn(model(x.to(device)).cpu().numpy())


def get_model(name, data, device, ckpt_dir, do_train, epochs):
    """Load a checkpoint if present, otherwise (optionally) train and save it."""
    path = Path(ckpt_dir) / f"{name}.pt"
    if path.exists() and not do_train:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = build_model(name, data.input_size, data.output_size, seq_to_seq=True)
        model.load_state_dict(ckpt["state_dict"])
        train_time = ckpt["config"].get("train_time_s", float("nan"))
        return model.to(device).eval(), train_time
    t0 = time.perf_counter()
    model, history = train_model(name, data, device, epochs=epochs, verbose=True)
    train_time = time.perf_counter() - t0
    save_checkpoint(path, model, name, data,
                    config={"epochs": epochs, "train_time_s": train_time, "history": history})
    return model.eval(), train_time


def plot_overlay(pos, phys, preds: dict, names, sample_desc, out_path):
    cjk = _use_cjk_font()
    labels = (["弓网接触力 [N]", "弓头位移 [m]", "接触网位移 [m]"] if cjk else
              [f"{n}" for n in names])
    phys_lbl = "物理模型" if cjk else "Physics"
    xlabel = "归一化位置" if cjk else "Normalized position"
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 9), sharex=True)
    for r in range(len(names)):
        ax = axes[r]
        ax.plot(pos, phys[:, r], color="k", lw=2.0, label=phys_lbl, zorder=5)
        for i, (mname, pr) in enumerate(preds.items()):
            ax.plot(pos, pr[:, r], lw=1.0, ls="--", color=f"C{i}", label=mname)
        ax.set_ylabel(labels[r])
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=len(preds) + 1, fontsize=8)
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(f"Surrogates vs physics — {sample_desc}")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_summary(rows, phys_time, out_path):
    names = [r["name"] for r in rows]
    nrmse = [r["nrmse"] for r in rows]
    inf_t = [r["infer_s"] for r in rows]
    cjk = _use_cjk_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = [f"C{i}" for i in range(len(names))]
    ax1.bar(names, nrmse, color=colors)
    ax1.set_ylabel("macro NRMSE" + (" (越低越好)" if cjk else " (lower=better)"))
    ax1.set_title("精度对比" if cjk else "Accuracy")
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(names, inf_t, color=colors)
    ax2.axhline(phys_time, color="k", ls="--", label=("物理模型" if cjk else "Physics"))
    ax2.set_yscale("log")
    ax2.set_ylabel(("单样本耗时 [s]" if cjk else "Time per sample [s]") + " (log)")
    ax2.set_title("推理耗时对比" if cjk else "Inference time")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="./result/pc_dataset.npz")
    ap.add_argument("--ckpt-dir", default="./result/checkpoints")
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated subset")
    ap.add_argument("--train", action="store_true", help="(re)train instead of loading checkpoints")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--sample", type=int, default=0, help="val index for the physics overlay")
    ap.add_argument("--out", default="./result/evaluation")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = PCData(args.data)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out)

    # Physics reference + timing for the chosen validation sample.
    s = args.sample % len(data.X_val_raw)
    x_row = data.X_val_raw[s]
    preset, speed = find_preset(x_row), float(x_row[5])
    nm, dt = data.meta["nm"], data.meta["dt"]
    print(f"Physics replay: preset={preset}  speed={speed:.1f} km/h  (NM={nm}, dt={dt:.1e})")
    phys, phys_time = physics_reference(preset, speed, nm, dt, len(data.pos))
    print(f"  physics wall time = {phys_time:.3f} s\n")

    rows, preds = [], {}
    for name in models:
        model, train_time = get_model(name, data, device, args.ckpt_dir, args.train, args.epochs)
        pred = predict(model, data.x_val, device, data.denorm_y)
        metrics = compute_metrics(pred, data.Y_val_raw, data.target_names)
        infer_s = time_inference(model, data.x_val, device)
        preds[name] = pred[s]
        rows.append({
            "name": name,
            "nrmse": metrics["_macro"]["nrmse"],
            "r2": metrics["_macro"]["r2"],
            "infer_s": infer_s,
            "train_time": train_time,
            "per_target": {n: metrics[n] for n in data.target_names},
        })

    rows.sort(key=lambda r: r["nrmse"])
    best = rows[0]["name"]

    # --- report ---
    print("=" * 78)
    print(f"{'model':12s} {'macroNRMSE':>11s} {'macroR2':>9s} {'infer/s':>11s} {'speedup':>9s}")
    print("-" * 78)
    for r in rows:
        speedup = phys_time / r["infer_s"] if r["infer_s"] > 0 else float("inf")
        print(f"{r['name']:12s} {r['nrmse']:11.4f} {r['r2']:9.4f} {r['infer_s']:11.3e} {speedup:8.0f}x")
    print("=" * 78)
    print(f"Best (lowest macro-NRMSE): {best}")
    print(f"Physics time/run: {phys_time:.3f} s  →  {best} is "
          f"{phys_time / next(r['infer_s'] for r in rows if r['name'] == best):.0f}x faster\n")

    print("Per-target metrics:")
    for r in rows:
        print(f"  [{r['name']}]")
        for n, m in r["per_target"].items():
            print(f"    {n:14s} RMSE={m['rmse']:.4g} NRMSE={m['nrmse']:.4f} R²={m['r2']:.4f}")

    # --- figures ---
    desc = f"preset {preset}, {speed:.0f} km/h"
    p1 = plot_overlay(data.pos, phys, preds, data.target_names, desc, out_dir / "overlay_vs_physics.png")
    p2 = plot_summary(rows, phys_time, out_dir / "summary_accuracy_time.png")
    print(f"\nFigures saved → {p1}\n               {p2}")


if __name__ == "__main__":
    main()
