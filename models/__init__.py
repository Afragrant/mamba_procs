"""序列回归模型集合.

基准: TCN, LSTM, Mamba3 (mamba_ssm)
提出: CNN-Mamba3 (CNN 前端 + Mamba3 主干)

统一接口: 输入 (B, T, IN_CHANNELS) -> 输出 (B, T, OUT_CHANNELS).
"""

import config as C

from .lstm import LSTMModel
from .tcn import TCNModel

# mamba_model / cnn_mamba3 依赖 mamba_ssm (需 CUDA), 故延迟到使用时再导入.


def build_model(name: str):
    name = name.lower()
    cfg = C.MODEL_CFG
    if name == 'tcn':
        return TCNModel(C.IN_CHANNELS, C.OUT_CHANNELS, **cfg['tcn'])
    if name == 'lstm':
        return LSTMModel(C.IN_CHANNELS, C.OUT_CHANNELS, **cfg['lstm'])
    if name in ('mamba3', 'mamba'):
        from .mamba_model import Mamba3Model
        return Mamba3Model(C.IN_CHANNELS, C.OUT_CHANNELS, **cfg['mamba3'])
    if name in ('cnn_mamba3', 'cnnmamba3', 'proposed'):
        from .cnn_mamba3 import build_cnn_mamba3
        return build_cnn_mamba3(C.IN_CHANNELS, C.OUT_CHANNELS, **cfg['cnn_mamba3'])
    raise ValueError(f'未知模型: {name}; 可选 tcn / lstm / mamba3 / cnn_mamba3')


def build_tuned(name: str, p: dict):
    """从 Optuna 搜索得到的扁平超参字典构建模型 (tune.py / evaluate.py 共用)."""
    name = name.lower()
    ci, co = C.IN_CHANNELS, C.OUT_CHANNELS
    if name == 'tcn':
        return TCNModel(ci, co, channels=tuple([p['width']] * p['depth']),
                        kernel_size=p['kernel'], dropout=p['dropout'])
    if name == 'lstm':
        return LSTMModel(ci, co, hidden_size=p['hidden'], num_layers=p['layers'],
                         dropout=p['dropout'], bidirectional=True)
    if name in ('mamba3', 'mamba'):
        from .mamba_model import Mamba3Model
        return Mamba3Model(ci, co, d_model=p['d_model'], n_layers=p['n_layers'],
                           d_state=p['d_state'], headdim=p['headdim'], dropout=p['dropout'])
    if name in ('cnn_mamba3', 'cnnmamba3', 'proposed'):
        from .cnn_mamba3 import CNN_CHANNEL_OPTS, CNN_KERNEL_OPTS, CNNMamba3
        ch = p['cnn_channels']
        ker = p['cnn_kernels']
        ch = CNN_CHANNEL_OPTS[ch] if isinstance(ch, str) else tuple(ch)
        ker = CNN_KERNEL_OPTS[ker] if isinstance(ker, str) else tuple(ker)
        # bidirectional / tie_weights 为消融实验条件 (非 Optuna 搜索项), 缺省即提出模型.
        return CNNMamba3(ci, co, cnn_channels=ch, cnn_kernels=ker, d_model=p['d_model'],
                         n_mamba=p['n_mamba'], d_state=p['d_state'], headdim=p['headdim'],
                         dropout=p['dropout'],
                         bidirectional=p.get('bidirectional', True),
                         tie_weights=p.get('tie_weights', True))
    raise ValueError(f'未知模型: {name}')


MODEL_NAMES = ('tcn', 'lstm', 'mamba3', 'cnn_mamba3')
