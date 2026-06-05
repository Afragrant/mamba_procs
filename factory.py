"""Model factory: build any of the four sequence models by name.

The model definitions live under ``model/``. ``TCN-Mamba.py`` has a hyphen in
its name (not importable normally) and itself does ``from TCN import ...`` /
``from Mamba import ...``, so we put ``model/`` on ``sys.path`` and load the
modules under the names those imports expect.
"""

import importlib.util
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

MODELS = ["tcn", "lstm", "mamba", "tcn-mamba"]

# Default architecture hyperparameters per model. Keys are exactly the model
# constructors' keyword arguments, so a tuned ``hp`` dict (from tune.py) can
# override any subset of them.
ARCH_DEFAULTS = {
    "tcn": dict(num_channels=[64, 64, 64, 64], kernel_size=3, dropout=0.1),
    "lstm": dict(hidden_size=128, n_layers=2, dropout=0.1),
    "mamba": dict(d_model=128, n_layers=4, d_state=128, dropout=0.0),
    "tcn-mamba": dict(tcn_channels=[64, 64, 128], kernel_size=3, n_mamba_layers=4,
                      d_state=128, dropout=0.1),
}


def canonical_name(name: str) -> str:
    key = name.lower()
    if key in ("tcn-mamba", "tcn_mamba", "tcnmamba"):
        return "tcn-mamba"
    if key in MODELS:
        return key
    raise ValueError(f"unknown model '{name}', choose from {MODELS}")


def _load_module(name: str, filename: str):
    """Load (and cache in sys.modules) a module from model/<filename>."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODEL_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(name: str, input_size: int, output_size: int, seq_to_seq: bool = True, hp: dict | None = None):
    """Construct a model. ``hp`` overrides the per-model ARCH_DEFAULTS (keys must
    match the constructor kwargs, e.g. num_channels / d_model / n_mamba_layers)."""
    key = canonical_name(name)
    cfg = {**ARCH_DEFAULTS[key], **(hp or {})}
    if key == "tcn":
        m = _load_module("TCN", "tcn.py")
        return m.TCN(input_size, output_size, seq_to_seq=seq_to_seq, **cfg)
    if key == "lstm":
        m = _load_module("LSTM", "lstm.py")
        return m.LSTM(input_size, output_size, seq_to_seq=seq_to_seq, **cfg)
    if key == "mamba":
        m = _load_module("Mamba", "mamba.py")
        return m.Mamba(input_size, output_size, seq_to_seq=seq_to_seq, **cfg)
    if key == "tcn-mamba":
        # tcn_mamba.py does `from TCN import ...` / `from Mamba import ...`, so its
        # dependencies must be registered under those names before it is loaded.
        _load_module("TCN", "tcn.py")
        _load_module("Mamba", "mamba.py")
        m = _load_module("TCN_Mamba", "tcn_mamba.py")
        return m.TCNMamba(input_size, output_size, seq_to_seq=seq_to_seq, **cfg)
    raise ValueError(f"unknown model '{name}', choose from {MODELS}")
