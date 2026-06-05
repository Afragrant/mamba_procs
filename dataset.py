"""Data preprocessing for the pantograph–catenary surrogate.

Loads the ``.npz`` produced by ``pc_to_data.py`` and builds model-ready tensors:

* the 6 scalar input features are standardized (train-split mean/std) and
  broadcast across the sequence,
* the normalized position grid is expanded into **Fourier positional features**
  (the raw ramp plus ``sin``/``cos`` at harmonics 1..F) and appended. A single
  linear ramp makes the models fight spectral bias when reconstructing the
  oscillatory contact force; a Fourier basis gives them ready-made high-frequency
  components keyed by the parameters, which is the dominant accuracy lever here,
* targets are standardized per channel.

Input tensor shape: ``(N, seq_len, 6 + 1 + 2*n_fourier)``.
Target tensor shape: ``(N, seq_len, 3)``.
"""

import numpy as np
import torch

# Number of Fourier harmonics for the position encoding. The stable window spans
# ~0.4·N≈12 spans, so the span-passing fundamental is ~12 cycles; a few dozen
# harmonics cover that and its overtones (sharp peaks near supports).
DEFAULT_N_FOURIER = 32


class PCData:
    """Holds normalized train/val tensors plus stats and raw (physical) targets."""

    def __init__(self, path: str, n_fourier: int = DEFAULT_N_FOURIER):
        d = np.load(path, allow_pickle=True)
        self.n_fourier = n_fourier
        self.feature_names = [str(s) for s in d["feature_names"]]
        self.target_names = [str(s) for s in d["target_names"]]
        self.pos = d["pos"].astype(np.float32)

        self.x_mean = d["x_mean"].astype(np.float32)
        self.x_std = d["x_std"].astype(np.float32)
        self.y_mean = d["y_mean"].astype(np.float32)
        self.y_std = d["y_std"].astype(np.float32)

        self.x_train = torch.from_numpy(self._build_inputs(d["X_train"]))
        self.x_val = torch.from_numpy(self._build_inputs(d["X_val"]))
        self.y_train = torch.from_numpy(self._norm_y(d["Y_train"]))
        self.y_val = torch.from_numpy(self._norm_y(d["Y_val"]))

        # Raw arrays kept for physical-unit evaluation and physics replay.
        self.X_train_raw = d["X_train"].astype(np.float32)
        self.X_val_raw = d["X_val"].astype(np.float32)
        self.Y_val_raw = d["Y_val"].astype(np.float32)

        # Optional generation settings (present in newer datasets).
        self.meta = {
            "nm": int(d["meta_nm"]) if "meta_nm" in d.files else 200,
            "dt": float(d["meta_dt"]) if "meta_dt" in d.files else 1e-5,
            "pantograph": int(d["meta_pantograph"]) if "meta_pantograph" in d.files else 1,
        }

        self.input_size = self.x_train.shape[-1]
        self.output_size = self.y_train.shape[-1]

    def _pos_features(self) -> np.ndarray:
        """Positional channels: raw ramp + sin/cos at harmonics 1..n_fourier → (seq, P)."""
        pos = self.pos  # (seq,) in [0, 1]
        feats = [pos[:, None]]
        if self.n_fourier > 0:
            freqs = np.arange(1, self.n_fourier + 1, dtype=np.float32)
            ang = 2.0 * np.pi * np.outer(pos, freqs)  # (seq, n_fourier)
            feats.append(np.sin(ang))
            feats.append(np.cos(ang))
        return np.concatenate(feats, axis=1).astype(np.float32)

    def _build_inputs(self, X: np.ndarray) -> np.ndarray:
        xn = (X - self.x_mean) / self.x_std  # (N, 6)
        n, seq = X.shape[0], len(self.pos)
        xb = np.broadcast_to(xn[:, None, :], (n, seq, xn.shape[1]))
        pf = self._pos_features()  # (seq, P)
        pfb = np.broadcast_to(pf[None, :, :], (n, seq, pf.shape[1]))
        return np.concatenate([xb, pfb], axis=2).astype(np.float32)

    def _norm_y(self, Y: np.ndarray) -> np.ndarray:
        return ((Y - self.y_mean) / self.y_std).astype(np.float32)

    def denorm_y(self, Yn: np.ndarray) -> np.ndarray:
        """Map normalized targets/predictions back to physical units."""
        return Yn * self.y_std + self.y_mean


def compute_metrics(pred: np.ndarray, true: np.ndarray, names: list[str]) -> dict:
    """Per-channel RMSE / MAE / NRMSE / R² in physical units, plus a macro summary.

    ``pred`` / ``true``: arrays of shape ``(N, seq_len, C)`` in physical units.
    NRMSE normalizes RMSE by the channel's standard deviation, giving a
    scale-free score for fair comparison/selection across channels and models.
    """
    out = {}
    for c, name in enumerate(names):
        p = pred[..., c].ravel()
        t = true[..., c].ravel()
        err = p - t
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        t_std = float(t.std()) + 1e-12
        nrmse = rmse / t_std
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((t - t.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        out[name] = {"rmse": rmse, "mae": mae, "nrmse": nrmse, "r2": r2}

    out["_macro"] = {
        "nrmse": float(np.mean([out[n]["nrmse"] for n in names])),
        "r2": float(np.mean([out[n]["r2"] for n in names])),
    }
    return out
