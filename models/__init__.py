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


MODEL_NAMES = ('tcn', 'lstm', 'mamba', 'tcn_mamba')
