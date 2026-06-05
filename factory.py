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


def _load_module(name: str, filename: str):
    """Load (and cache in sys.modules) a module from model/<filename>."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODEL_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(name: str, input_size: int, output_size: int, seq_to_seq: bool = True):
    """Construct a model with consistent I/O dimensions and sensible defaults."""
    key = name.lower()
    if key == "tcn":
        m = _load_module("TCN", "tcn.py")
        return m.TCN(
            input_size, output_size,
            num_channels=[64, 64, 64, 64], kernel_size=3, dropout=0.1,
            seq_to_seq=seq_to_seq,
        )
    if key == "lstm":
        m = _load_module("LSTM", "lstm.py")
        return m.LSTM(
            input_size, output_size,
            hidden_size=128, n_layers=2, dropout=0.1,
            seq_to_seq=seq_to_seq,
        )
    if key == "mamba":
        m = _load_module("Mamba", "mamba.py")
        return m.Mamba(
            input_size, output_size,
            d_model=128, n_layers=4,
            seq_to_seq=seq_to_seq,
        )
    if key in ("tcn-mamba", "tcn_mamba", "tcnmamba"):
        # tcn_mamba.py does `from TCN import ...` / `from Mamba import ...`, so its
        # dependencies must be registered under those names before it is loaded.
        _load_module("TCN", "tcn.py")
        _load_module("Mamba", "mamba.py")
        m = _load_module("TCN_Mamba", "tcn_mamba.py")
        return m.TCNMamba(
            input_size, output_size,
            tcn_channels=[64, 64, 128], kernel_size=3, n_mamba_layers=4, dropout=0.1,
            seq_to_seq=seq_to_seq,
        )
    raise ValueError(f"unknown model '{name}', choose from {MODELS}")
