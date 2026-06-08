"""序列回归模型集合.

基准: TCN, LSTM, Mamba (mamba_ssm)
提出: TCN+Mamba+MC-Dropout (TCNMambaMC) —— 带蒙特卡洛 Dropout 不确定性量化

统一接口: 输入 (B, T, IN_CHANNELS) -> 输出 (B, T, OUT_CHANNELS).
"""

import config as C

from .lstm import LSTMModel
from .tcn import TCNModel

# mamba_model / tcn_mamba 依赖 mamba_ssm (需 CUDA), 故延迟到使用时再导入.


def build_model(name: str):
    name = name.lower()
    cfg = C.MODEL_CFG
    if name == 'tcn':
        return TCNModel(C.IN_CHANNELS, C.OUT_CHANNELS, **cfg['tcn'])
    if name == 'lstm':
        return LSTMModel(C.IN_CHANNELS, C.OUT_CHANNELS, **cfg['lstm'])
    if name == 'mamba':
        from .mamba_model import MambaModel
        return MambaModel(C.IN_CHANNELS, C.OUT_CHANNELS, **cfg['mamba'])
    if name in ('tcn_mamba', 'tcnmamba', 'proposed'):
        from .tcn_mamba import build_tcn_mamba
        return build_tcn_mamba(C.IN_CHANNELS, C.OUT_CHANNELS, **cfg['tcn_mamba'])
    raise ValueError(f'未知模型: {name}; 可选 tcn / lstm / mamba / tcn_mamba')


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
    if name == 'mamba':
        from .mamba_model import MambaModel
        return MambaModel(ci, co, d_model=p['d_model'], n_layers=p['n_layers'],
                          d_state=p['d_state'], dropout=p['dropout'])
    if name in ('tcn_mamba', 'tcnmamba', 'proposed'):
        from .tcn_mamba import TCNMambaMC
        return TCNMambaMC(ci, co, tcn_channels=tuple([p['tcn_width']] * p['tcn_depth']),
                          tcn_kernel=p['kernel'], d_model=p['d_model'], n_mamba=p['n_mamba'],
                          d_state=p['d_state'], dropout=p['dropout'])
    raise ValueError(f'未知模型: {name}')


MODEL_NAMES = ('tcn', 'lstm', 'mamba', 'tcn_mamba')
