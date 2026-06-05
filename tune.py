"""Per-model hyperparameter optimization with Optuna.

For a fair "which architecture is best" comparison, each model is tuned to its
own optimum rather than reused at a shared hand-picked size.

Methodology (important): tuning must not see the test set. The 4 catenary
presets in ``X_val`` are the held-out final test and are NOT touched here.
Instead an **HPO-validation** split is carved from the LHS training set; every
trial trains on the HPO-train remainder and is scored on HPO-val. Only after a
model is retrained (train.py / evaluate.py) are the presets used.

Each model's best hyperparameters are written to
``result/tuning/<model>_best.json`` as ``{"arch": {...}, "train": {...}}``;
train.py / evaluate.py pick them up automatically via ``load_best_hp``.

A MedianPruner stops unpromising trials early (using the per-epoch val loss
reported through the training callback), keeping the search affordable.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import torch

from dataset import PCData
from factory import MODELS, canonical_name
from train import train_model


def make_hpo_split(data: PCData, frac: float, seed: int):
    """Carve an HPO-val subset out of the LHS training set (presets stay untouched)."""
    n = len(data.x_train)
    idx = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(n * frac)))
    vi, ti = idx[:n_val], idx[n_val:]
    train_t = (data.x_train[ti], data.y_train[ti])
    val_t = (data.x_train[vi], data.y_train[vi])
    return train_t, val_t


def suggest_params(name: str, trial: optuna.Trial) -> dict:
    """Sample a flat parameter dict for the given model (training + architecture)."""
    p = {
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    }
    key = canonical_name(name)
    if key == "tcn":
        p["tcn_width"] = trial.suggest_categorical("tcn_width", [32, 64, 96, 128])
        p["tcn_depth"] = trial.suggest_int("tcn_depth", 3, 6)
        p["kernel_size"] = trial.suggest_categorical("kernel_size", [2, 3, 5])
    elif key == "lstm":
        p["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128, 192, 256])
        p["n_layers"] = trial.suggest_int("n_layers", 1, 3)
    elif key == "mamba":
        # d_model multiple of 32 so expand*d_model is divisible by headdim=64.
        p["d_model"] = trial.suggest_categorical("d_model", [64, 96, 128, 192, 256])
        p["n_layers"] = trial.suggest_int("n_layers", 2, 6)
        p["d_state"] = trial.suggest_categorical("d_state", [64, 128])
    elif key == "tcn-mamba":
        p["tcnm_width"] = trial.suggest_categorical("tcnm_width", [64, 96, 128, 192])
        p["tcnm_depth"] = trial.suggest_int("tcnm_depth", 2, 4)
        p["n_mamba_layers"] = trial.suggest_int("n_mamba_layers", 2, 6)
        p["d_state"] = trial.suggest_categorical("d_state", [64, 128])
        p["kernel_size"] = trial.suggest_categorical("kernel_size", [2, 3, 5])
    return p


def params_to_hp(name: str, p: dict) -> tuple[dict, dict]:
    """Convert a flat param dict into (arch_hp, train_hp) for build_model / train_model."""
    train_hp = {"lr": p["lr"], "batch_size": p["batch_size"], "weight_decay": p["weight_decay"]}
    dropout = p["dropout"]
    key = canonical_name(name)
    if key == "tcn":
        arch = dict(num_channels=[p["tcn_width"]] * p["tcn_depth"],
                    kernel_size=p["kernel_size"], dropout=dropout)
    elif key == "lstm":
        arch = dict(hidden_size=p["hidden_size"], n_layers=p["n_layers"], dropout=dropout)
    elif key == "mamba":
        arch = dict(d_model=p["d_model"], n_layers=p["n_layers"], d_state=p["d_state"], dropout=dropout)
    elif key == "tcn-mamba":
        arch = dict(tcn_channels=[p["tcnm_width"]] * p["tcnm_depth"], kernel_size=p["kernel_size"],
                    n_mamba_layers=p["n_mamba_layers"], d_state=p["d_state"], dropout=dropout)
    return arch, train_hp


def tune_model(name, data, device, train_t, val_t, n_trials, tune_epochs, tune_patience, seed):
    """Run an Optuna study for one model; return (best_arch_hp, best_train_hp, study)."""

    def objective(trial: optuna.Trial) -> float:
        p = suggest_params(name, trial)
        arch_hp, train_hp = params_to_hp(name, p)

        def cb(epoch, val_loss):
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            _, history = train_model(
                name, data, device,
                epochs=tune_epochs, patience=tune_patience, seed=seed,
                lr=train_hp["lr"], batch_size=train_hp["batch_size"],
                weight_decay=train_hp["weight_decay"], hp=arch_hp,
                train_tensors=train_t, val_tensors=val_t,
                epoch_callback=cb, verbose=False,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        return float(min(history))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    arch_hp, train_hp = params_to_hp(name, study.best_params)
    return arch_hp, train_hp, study


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="./result/pc_dataset.npz")
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated subset")
    ap.add_argument("--trials", type=int, default=30, help="Optuna trials per model")
    ap.add_argument("--tune-epochs", type=int, default=150, help="epochs per trial (short budget)")
    ap.add_argument("--tune-patience", type=int, default=25)
    ap.add_argument("--hpo-val-frac", type=float, default=0.15, help="HPO-val fraction of LHS train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="./result/tuning")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = PCData(args.data)
    train_t, val_t = make_hpo_split(data, args.hpo_val_frac, args.seed)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Hyperparameter optimization (Optuna)")
    print(f"  HPO-train: {len(train_t[0])}   HPO-val: {len(val_t[0])}   "
          f"(presets held out as final test)")
    print(f"  trials/model: {args.trials}   tune-epochs: {args.tune_epochs}")
    print("=" * 70)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    summary = {}
    for name in models:
        key = canonical_name(name)
        t0 = time.perf_counter()
        arch_hp, train_hp, study = tune_model(
            name, data, device, train_t, val_t,
            args.trials, args.tune_epochs, args.tune_patience, args.seed,
        )
        elapsed = time.perf_counter() - t0
        n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        out = out_dir / f"{key}_best.json"
        out.write_text(json.dumps(
            {"arch": arch_hp, "train": train_hp,
             "best_hpo_val_mse": study.best_value, "trials": len(study.trials)},
            indent=2,
        ))
        summary[key] = study.best_value
        print(f"[{key}] {elapsed:.0f}s  trials={len(study.trials)} (pruned {n_pruned})  "
              f"best_hpo_val_mse={study.best_value:.4e}")
        print(f"        arch={arch_hp}")
        print(f"        train={train_hp}  → saved {out}")

    print("=" * 70)
    print("Best HPO-val MSE per model (lower=better):")
    for k, v in sorted(summary.items(), key=lambda kv: kv[1]):
        print(f"  {k:12s} {v:.4e}")
    print("Now retrain on full train + score on presets:  python evaluate.py --train")


if __name__ == "__main__":
    main()
