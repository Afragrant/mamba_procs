"""Generate a pantograph–catenary dataset for the TCN-Mamba surrogate model.

The surrogate maps the rigid overhead contact system parameters
``(L, rhoA, EI, KEQ, MEQ)`` together with the train ``speed`` to the
stable-window response of the coupled system: contact force, pantograph-head
displacement and catenary displacement.

Split design (this is deliberate, for honest generalization testing):

* **Training set** — Latin-Hypercube samples drawn *continuously* over the
  parameter ranges in ``PARAM_RANGES`` (the ranges bound the four presets). The
  model therefore learns the parameter→response mapping over a whole region, not
  just four points.
* **Validation / test set** — the four rigid catenary presets (1–4) swept across
  speeds. These exact configurations never appear in training, so scoring on
  them measures generalization to the real catenaries of interest.

Each physics run (``run_simulation`` from ``pc.py``, pantograph fixed to type 1)
keeps only the stable window, resampled to a fixed sequence length. Normalization
statistics are computed from the TRAIN split only.

Outputs (single ``.npz``): X_train/Y_train (LHS), X_val/Y_val (presets),
feature_names, target_names, pos, x_mean/x_std/y_mean/y_std, generation meta.
"""

# Keep BLAS single-threaded so multiprocessing workers don't oversubscribe cores.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pc import rigid_overhead_contact_system_params, run_simulation

PANTOGRAPH = 1  # fixed per requirement
PRESETS = [1, 2, 3, 4]  # held-out validation/test catenaries

# Continuous training-sample ranges. Bound the four presets so the LHS region
# encloses them. scale 'log' draws log-uniformly (KEQ spans ~6e4 .. 6.7e7).
PARAM_RANGES = {
    "L": (7.5, 8.6, "lin"),
    "rhoA": (7.0, 8.2, "lin"),
    "EI": (1.6e5, 2.8e5, "lin"),
    "KEQ": (6.0e4, 6.7e7, "log"),
    "MEQ": (2.5, 7.5, "lin"),
    "speed_kmh": (180.0, 250.0, "lin"),
}
FEATURE_NAMES = list(PARAM_RANGES.keys())
TARGET_NAMES = ["contact_force", "y_pantograph", "y_catenary"]


def sample_lhs(n: int, seed: int) -> np.ndarray:
    """Latin-Hypercube sample of all 6 input features → (n, 6)."""
    try:
        from scipy.stats import qmc

        unit = qmc.LatinHypercube(d=len(PARAM_RANGES), seed=seed).random(n)
    except Exception:
        unit = np.random.default_rng(seed).random((n, len(PARAM_RANGES)))

    cols = []
    for j, (lo, hi, scale) in enumerate(PARAM_RANGES.values()):
        u = unit[:, j]
        if scale == "log":
            cols.append(np.exp(u * (np.log(hi) - np.log(lo)) + np.log(lo)))
        else:
            cols.append(u * (hi - lo) + lo)
    return np.stack(cols, axis=1).astype(np.float32)


def preset_features(preset: int) -> tuple[float, float, float, float, float]:
    L, _N, rhoA, EI, KEQ, MEQ, _MZ, _L_MZ = rigid_overhead_contact_system_params(preset)
    return L, rhoA, EI, KEQ, MEQ


def build_train_tasks(n_train, seq_len, nm, dt, seed):
    """LHS continuous tasks. mode='lhs' → run with catenary_params override."""
    X = sample_lhs(n_train, seed)
    return [("lhs", tuple(X[i, :5]), float(X[i, 5]), X[i], seq_len, nm, dt) for i in range(n_train)]


def build_test_tasks(speeds_per_cat, seq_len, nm, dt, seed):
    """Preset tasks. mode='preset' → run via rigid_overhead_contact_system index."""
    rng = np.random.default_rng(seed + 10_000)
    tasks = []
    for preset in PRESETS:
        feats = preset_features(preset)
        speeds = rng.uniform(*PARAM_RANGES["speed_kmh"][:2], size=speeds_per_cat)
        for sp in speeds:
            x = np.array([*feats, sp], dtype=np.float32)
            tasks.append(("preset", preset, float(sp), x, seq_len, nm, dt))
    return tasks


def _resample(arr: np.ndarray, seq_len: int) -> np.ndarray:
    xp = np.linspace(0.0, 1.0, len(arr))
    return np.interp(np.linspace(0.0, 1.0, seq_len), xp, arr)


def _run_one(task: tuple) -> np.ndarray | None:
    """Worker: run one simulation → resampled (seq_len, 3) targets, or None on failure."""
    mode, payload, speed, _x, seq_len, nm, dt = task
    kw = dict(pantograph=PANTOGRAPH, speed_kmh=speed, NM=nm, dt_base=dt, verbose=False)
    if mode == "lhs":
        kw["catenary_params"] = payload
    else:
        kw["rigid_overhead_contact_system"] = payload
    try:
        res = run_simulation(**kw)
    except Exception:
        return None

    y = np.stack(
        [
            _resample(res["contact_force_stable"], seq_len),
            _resample(res["y_pantograph_stable"], seq_len),
            _resample(res["y_rigid_overhead_contact_system_stable"], seq_len),
        ],
        axis=1,
    )
    if not np.all(np.isfinite(y)):
        return None
    return y.astype(np.float32)


def generate(tasks: list, workers: int, desc: str) -> tuple[np.ndarray, np.ndarray]:
    """Run a task list in parallel → (X, Y), dropping failed runs."""
    n = len(tasks)
    Y = [None] * n
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_run_one, t): i for i, t in enumerate(tasks)}
        for fut in tqdm(as_completed(futures), total=n, desc=desc, unit="run"):
            Y[futures[fut]] = fut.result()
    keep = [i for i in range(n) if Y[i] is not None]
    if len(keep) < n:
        print(f"  dropped {n - len(keep)} failed/invalid runs ({desc})")
    X = np.stack([tasks[i][3] for i in keep], axis=0)
    Yk = np.stack([Y[i] for i in keep], axis=0)
    return X, Yk


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=256, help="LHS continuous training samples")
    ap.add_argument("--speeds-per-cat", type=int, default=16, help="test speeds per preset (×4 presets)")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--nm", type=int, default=200, help="retained catenary modes")
    ap.add_argument("--dt", type=float, default=1e-5, help="base time step [s]")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="./result/pc_dataset.npz")
    args = ap.parse_args()

    train_tasks = build_train_tasks(args.n_train, args.seq_len, args.nm, args.dt, args.seed)
    test_tasks = build_test_tasks(args.speeds_per_cat, args.seq_len, args.nm, args.dt, args.seed)

    print("=" * 60)
    print("Pantograph–catenary dataset generation")
    print("=" * 60)
    print(f"  train (LHS)   : {len(train_tasks)} continuous samples over {FEATURE_NAMES}")
    print(f"  val/test      : presets {PRESETS} × {args.speeds_per_cat} speeds = {len(test_tasks)}")
    print(f"  speed range   : {PARAM_RANGES['speed_kmh'][0]:.0f}–{PARAM_RANGES['speed_kmh'][1]:.0f} km/h")
    print(f"  seq_len       : {args.seq_len}   modes NM: {args.nm}   dt: {args.dt:.1e}")
    print(f"  pantograph    : {PANTOGRAPH} (fixed)   workers: {args.workers}")

    X_train, Y_train = generate(train_tasks, args.workers, "Train(LHS)")
    X_val, Y_val = generate(test_tasks, args.workers, "Val(presets)")
    print(f"  collected     : train X{tuple(X_train.shape)}  val X{tuple(X_val.shape)}")

    # Normalization stats from the TRAIN split only.
    x_mean, x_std = X_train.mean(0), X_train.std(0) + 1e-8
    y_mean = Y_train.reshape(-1, 3).mean(0)
    y_std = Y_train.reshape(-1, 3).std(0) + 1e-8

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        feature_names=np.array(FEATURE_NAMES),
        target_names=np.array(TARGET_NAMES),
        pos=np.linspace(0.0, 1.0, args.seq_len).astype(np.float32),
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
        # generation settings — used by evaluate.py to time the physics fairly.
        meta_nm=np.int64(args.nm),
        meta_dt=np.float64(args.dt),
        meta_seq_len=np.int64(args.seq_len),
        meta_pantograph=np.int64(PANTOGRAPH),
        meta_val_is_presets=np.int64(1),
    )
    print(f"\nSaved → {out}")
    print(f"  train (LHS)   : X{tuple(X_train.shape)}  Y{tuple(Y_train.shape)}")
    print(f"  val (presets) : X{tuple(X_val.shape)}  Y{tuple(Y_val.shape)}")


if __name__ == "__main__":
    main()
