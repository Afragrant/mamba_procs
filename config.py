"""项目全局配置 —— 基于 TCN-Mamba 的刚性接触网可替代(代理)模型.

集中管理: 设计变量取值范围、仿真保真度、序列长度、数据划分、归一化区间、
训练与模型超参数、文件路径. 其余脚本统一从这里读取, 避免散落的魔数.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
RESULT_DIR = ROOT / 'result'
CKPT_DIR = RESULT_DIR / 'checkpoints'
for _d in (DATA_DIR, RESULT_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 数据集文件 (由 pc_to_data.py 生成)
DATASET_FULL = DATA_DIR / 'dataset_full.npz'      # 20000 样本正式数据集
DATASET_SMOKE = DATA_DIR / 'dataset_smoke.npz'    # 小样本冒烟测试数据集
NORM_STATS = DATA_DIR / 'norm_stats.npz'          # 归一化统计量(随数据集一并写出)

# --------------------------------------------------------------------------- #
# 5 个设计变量 (输入) 及其取值范围 —— 拉丁超立方采样的边界
#   KEQ : 支撑等效刚度 (题述 ks)  [N/m]
#   L   : 单跨长度                [m]
#   V   : 列车速度                [km/h]
#   EI  : 抗弯刚度                [N·m^2]
#   rhoA: 线密度                  [kg/m]
# 顺序固定为 (KEQ, L, V, EI, rhoA), 全项目一致.
# --------------------------------------------------------------------------- #
INPUT_NAMES = ('KEQ', 'L', 'V', 'EI', 'rhoA')
INPUT_RANGES = {
    'KEQ': (6.0e4, 6.7e7),
    'L': (6.0, 10.0),
    'V': (160.0, 250.0),
    'EI': (1.5e5, 4.0e5),
    'rhoA': (3.0, 10.0),
}

# 跨数量级的变量改在对数空间做 LHS, 使各数量级均衡采样 (低刚度区不被饿死).
# KEQ 跨约 3 个数量级 (1100 倍) -> 对数采样; EI 仅 2.67 倍, 线性即可, 不在此列.
# 注: 仅影响采样分布; 归一化仍按题述线性 Min-Max 公式 (二者独立).
LOG_SAMPLE_VARS = {'KEQ'}

# 可选: 对标论文 Hu et al.(2025) Table 5 的窄采样范围 (KS 仅 1 个数量级, CF 远更易学).
# 题述范围 KEQ∈[6e4,6.7e7] 跨 3 个数量级、含高刚度离线工况, 比论文难约 100 倍.
# 置 True 即切到论文范围; ⚠️ 改动后必须重新生成数据集 (pc_to_data.py --full).
USE_PAPER_KS_RANGE = False
KS_RANGE_PAPER = (1.0e4, 1.0e5)
if USE_PAPER_KS_RANGE:
    INPUT_RANGES['KEQ'] = KS_RANGE_PAPER

# 3 个输出 (时间/空间序列)
#   Fc      : 弓网接触力      [N]
#   y_panto : 弓头位移        [m]
#   y_cat   : 刚性接触网位移  [m]
OUTPUT_NAMES = ('Fc', 'y_panto', 'y_cat')

# 受电弓固定为第 1 种 (DSA380); 接触网跨数固定 30 跨.
PANTOGRAPH_TYPE = 1
N_SPANS = 30

# --------------------------------------------------------------------------- #
# 仿真保真度 (用户选定: NM=100, dt=4e-5; 稳定段 30%~70%)
# --------------------------------------------------------------------------- #
NM = 100
DT_BASE = 4.0e-5
STABLE_START = 0.30
STABLE_END = 0.70

# 每个样本的稳定段重采样到固定序列长度 T.
# 稳定段恒好覆盖 0.4 * N_SPANS = 12 跨 (起点恰为第 9*L 跨边界, 相位对齐).
SEQ_LEN = 512

# --------------------------------------------------------------------------- #
# 数据集规模与划分
# --------------------------------------------------------------------------- #
N_SAMPLES_FULL = 20000
SPLIT_FULL = {'train': 17000, 'val': 1500, 'test': 1500}

N_SAMPLES_SMOKE = 200
SPLIT_SMOKE = {'train': 170, 'val': 15, 'test': 15}

LHS_SEED = 20260607

# --------------------------------------------------------------------------- #
# 归一化: Min-Max 到 [-1, 1]
#   x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
# 输入用已知 LHS 边界; 输出用训练集统计量.
# --------------------------------------------------------------------------- #
NORM_LO, NORM_HI = -1.0, 1.0

# --------------------------------------------------------------------------- #
# 位置编码 (序列模型实现细节, 不计入 5 个设计变量)
# 5 个设计变量在序列上为常量, 纯卷积 TCN 对常量输入会输出常量; 因此为所有模型
# 附加一组确定性的 Fourier 位置编码, 使其能生成沿线周期性响应, 保证公平对比.
# 通道数 = 1(线性) + 2 * N_POS_FREQS(sin/cos).
# --------------------------------------------------------------------------- #
POS_ENCODING = True
N_POS_FREQS = 6

# 模型输入通道数: 5 设计变量 (+ 位置编码)
N_DESIGN = len(INPUT_NAMES)
N_POS = (1 + 2 * N_POS_FREQS) if POS_ENCODING else 0
IN_CHANNELS = N_DESIGN + N_POS
OUT_CHANNELS = len(OUTPUT_NAMES)

# --------------------------------------------------------------------------- #
# 训练超参数
# --------------------------------------------------------------------------- #
DEVICE = 'cuda'  # mamba_ssm 需要 CUDA
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 15
NUM_WORKERS = 4
SEED = 42

# ReduceLROnPlateau
LR_FACTOR = 0.5
LR_PATIENCE = 6
LR_MIN = 1e-6

# 各输出通道损失权重 (Fc, y_panto, y_cat).
# 接触力 Fc 高频、最难学且最关键, 故加大权重, 把模型容量向 Fc 倾斜.
CHANNEL_LOSS_WEIGHTS = (4.0, 1.0, 1.0)

# 物理约束 (Physics-Constrained) 模型 PC-TCN-Mamba 用的接触弹簧刚度常量.
# pc.py: contact_force = KS * relu(y_panto - y_cat), KS=82300 (固定接触界面参数,
# 与设计变量"支撑刚度 KEQ"无关). PC 模型把该精确本构关系内嵌进输出层:
#   gap = y_panto - y_cat;  Fc = CONTACT_KC * relu(gap)
# 从而保证 Fc≥0、离线精确为零、三量物理一致, 并直接优化无抵消的 gap.
CONTACT_KC = 82300.0

# MC Dropout (提出模型: TCN+Mamba+蒙特卡洛)
# p 提到 0.15: 增大预测方差的"可变性", 使不确定性更有信息量 (需重训生效).
MC_DROPOUT_P = 0.15
MC_SAMPLES = 50  # 推理时蒙特卡洛前向次数
# 不确定性后验标定: 在验证集上求每通道缩放因子 k, 使 (均值 ± k·σ) 覆盖率≈目标值.
CALIBRATE_UQ = True
UQ_TARGET_COVERAGE = 0.95

# 各模型默认结构超参数
MODEL_CFG = {
    'tcn': dict(channels=(64, 64, 64, 64), kernel_size=5, dropout=0.1),
    'lstm': dict(hidden_size=128, num_layers=2, dropout=0.1, bidirectional=True),
    'mamba': dict(d_model=128, n_layers=4, d_state=16, d_conv=4, expand=2, dropout=0.1),
    'tcn_mamba': dict(
        tcn_channels=(64, 64),
        tcn_kernel=5,
        d_model=128,
        n_mamba=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=MC_DROPOUT_P,
    ),
    # 提出模型: 物理约束 PC-TCN-Mamba (结构同 tcn_mamba, 仅输出层换成物理推导)
    'pc_tcn_mamba': dict(
        tcn_channels=(64, 64),
        tcn_kernel=5,
        d_model=128,
        n_mamba=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=MC_DROPOUT_P,
    ),
}
