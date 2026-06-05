"""Train a sequence surrogate (TCN / LSTM / Mamba / TCN-Mamba) on the dataset.

Exposes ``train_model`` (reused by ``evaluate.py``) and a CLI that trains one
model and saves a checkpoint bundling the weights, config and normalization
stats so ``validate.py`` / ``evaluate.py`` can run standalone.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import PCData
from factory import build_model


def train_model(
    name: str,
    data: PCData,
    device: str,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 16,
    seed: int = 0,
    patience: int = 40,
    verbose: bool = True,
):
    """Train one model, returning (best_model, history) where history is val MSE.

    Stops early if the validation loss does not improve for ``patience`` epochs
    (``patience<=0`` disables it). The best-val weights are always restored, so a
    longer ``epochs`` ceiling is safe — training self-regulates.
    """
    torch.manual_seed(seed)
    model = build_model(name, data.input_size, data.output_size, seq_to_seq=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.MSELoss()

    loader = DataLoader(
        TensorDataset(data.x_train, data.y_train), batch_size=batch_size, shuffle=True
    )
    x_val, y_val = data.x_val.to(device), data.y_val.to(device)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0
    history = []
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[{name}] params={n_params:,}  train={len(data.x_train)}  val={len(data.x_val)}")

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()
        history.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}/{epochs}  val_mse={val_loss:.4e}  best={best_val:.4e} (@{best_epoch})")

        if patience > 0 and epochs_no_improve >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (no val improvement for {patience} epochs; "
                      f"best={best_val:.4e} @ epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def plot_history(history: list, name: str, out_path: Path):
    """Plot the validation-loss curve (log scale) and mark the best epoch."""
    import matplotlib.pyplot as plt

    best_epoch = int(np.argmin(history)) + 1
    best_val = float(min(history))
    epochs = np.arange(1, len(history) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, history, color="C0", lw=1.2, label="val MSE")
    ax.axvline(best_epoch, color="C3", ls="--", lw=1.0,
               label=f"best @ {best_epoch} ({best_val:.3e})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MSE (log)")
    ax.set_title(f"{name} — training curve")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path, best_epoch, best_val


def save_checkpoint(path: Path, model, name: str, data: PCData, config: dict):
    """Persist weights + config + normalization stats + grid for standalone use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": name,
            "state_dict": model.state_dict(),
            "input_size": data.input_size,
            "output_size": data.output_size,
            "config": config,
            "feature_names": data.feature_names,
            "target_names": data.target_names,
            "pos": data.pos,
            "x_mean": data.x_mean,
            "x_std": data.x_std,
            "y_mean": data.y_mean,
            "y_std": data.y_std,
        },
        path,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="tcn | lstm | mamba | tcn-mamba")
    ap.add_argument("--data", default="./result/pc_dataset.npz")
    ap.add_argument("--epochs", type=int, default=300, help="max epochs (early stopping may end sooner)")
    ap.add_argument("--patience", type=int, default=40, help="early-stop patience; 0 disables")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="./result/checkpoints")
    ap.add_argument("--no-plot", action="store_true", help="skip the loss-curve figure")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = PCData(args.data)

    t0 = time.perf_counter()
    model, history = train_model(
        args.model, data, device,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        seed=args.seed, patience=args.patience,
    )
    train_time = time.perf_counter() - t0
    print(f"[{args.model}] trained in {train_time:.1f} s  "
          f"best_val_mse={min(history):.4e}  ran {len(history)}/{args.epochs} epochs")

    out = Path(args.out_dir) / f"{args.model}.pt"
    save_checkpoint(
        out, model, args.model, data,
        config={"epochs": args.epochs, "patience": args.patience, "lr": args.lr,
                "batch_size": args.batch_size, "seed": args.seed,
                "train_time_s": train_time, "history": history},
    )
    print(f"Saved checkpoint → {out}")

    if not args.no_plot:
        fig_path, best_epoch, best_val = plot_history(
            history, args.model, Path("./result/training") / f"{args.model}_loss.png"
        )
        converged = len(history) < args.epochs or best_epoch < 0.9 * len(history)
        print(f"Loss curve → {fig_path}")
        print(f"  best epoch {best_epoch}/{len(history)} — "
              + ("looks converged (best is well before the end)." if converged
                 else "best is near the end → consider raising --epochs."))


if __name__ == "__main__":
    main()
