# 基于 TCN-Mamba 的刚性接触网可替代(代理)模型

用深度序列模型替代刚性接触网–受电弓有限元/模态时域仿真 (`pc.py`)，
以 5 个设计变量直接预测稳定段 (30%–70%) 的 **弓网接触力、弓头位移、刚性接触网位移**
三条响应序列。基准模型为 TCN / LSTM / Mamba，提出模型为
**TCN + Mamba + 蒙特卡洛 Dropout (不确定性量化)**。

---

## 1. 项目结构

```
pc.py            原始物理仿真 (模态叠加 + Newmark-β)，未改动，作为数据真值
config.py        全局配置: 变量范围、保真度、序列长度、划分、超参数、路径
pc_to_data.py    由 pc.py 改写: 拉丁超立方采样 + 多进程仿真 + 重采样 + 写 npz
data_utils.py    Min-Max[-1,1] 归一化、位置编码、torch Dataset/DataLoader
models/
  tcn.py         TCN 基准 (膨胀因果卷积 + 残差块)
  lstm.py        LSTM 基准 (双向)
  mamba_model.py Mamba 基准 (mamba_ssm)
  tcn_mamba.py   提出模型 TCNMambaMC (TCN 前端 + Mamba 主干 + MC Dropout)
metrics.py       RRMSE / R² / RDE_Std / RDE_Ave / RMSE / RMAE / Erel
train.py         训练任一模型 (AdamW + ReduceLROnPlateau + 早停)
evaluate.py      测试集评估 + 指标表 + 预测图 + MC Dropout 置信带与覆盖率
data/            生成的数据集与归一化统计量
result/          checkpoints / eval 图表
```

## 2. 输入 / 输出定义

| 设计变量 (输入, 5) | 符号 | 范围 | 单位 |
|---|---|---|---|
| 支撑等效刚度 (题述 ks) | KEQ | 6e4 ~ 6.7e7 | N/m |
| 单跨长度 | L | 6 ~ 10 | m |
| 列车速度 | V | 160 ~ 250 | km/h |
| 抗弯刚度 | EI | 1.5e5 ~ 4e5 | N·m² |
| 线密度 | rhoA | 3 ~ 10 | kg/m |

受电弓固定第 1 种 (DSA380)，跨数固定 30。

| 输出 (3, 序列) | 符号 |
|---|---|
| 弓网接触力 | Fc |
| 弓头位移 | y_panto |
| 刚性接触网位移 | y_cat |

每个样本取稳定段 (30%–70%) 三条响应，**重采样到固定长度 T=512**。稳定段起点恰为
第 9·L 跨边界（`0.3·L·N = 9L`），故所有样本相位严格对齐、恒覆盖 12 跨，跨保真度一致。

## 3. 归一化

Min-Max 到 [-1, 1]（题述公式）：

```
x_norm = 2·(x - x_min)/(x_max - x_min) - 1
```

输入用已知 LHS 边界，输出用 **训练集** 每通道全局 min/max（防止信息泄漏），
统计量随数据集写入 `data/norm_stats.npz`。

## 4. 序列建模与位置编码

5 个设计变量对单个样本是常量。纯卷积 TCN 对常量输入只能输出常量，无法生成沿线
周期响应；因此为 **所有模型** 统一附加一组确定性 Fourier 位置编码
（1 线性 + 2·6 个 sin/cos 通道），作为模型实现细节、**不计入 5 个设计变量**，
保证四个模型在同一输入下公平对比。模型输入通道数 = 5 + 13 = 18。

## 5. 提出模型为什么能加蒙特卡洛

可以，且推荐采用 **MC Dropout（蒙特卡洛 Dropout，Gal & Ghahramani 2016）** 作为
贝叶斯近似：训练正常带 Dropout；**推理时保持 Dropout 激活**，对同一输入做 N 次前向，
得到接触力等输出的 **预测均值 ± 标准差（置信带）**。这把确定性代理升级为
**带不确定性量化 (UQ) 的代理**，工程意义明确——给出弓网接触力预测的可信区间，
直接服务受流质量可靠性评估，且几乎不增训练成本。故提出模型定为
**TCN + Mamba + 蒙特卡洛 Dropout (`TCNMambaMC`)**，而非纯 TCN+Mamba。
（`evaluate.py` 会报告 95% 置信带覆盖率与平均 σ。）

## 6. 使用流程

```bash
# 0) 环境 (已配置 .venv: torch+cu130, mamba_ssm, scipy, numpy, matplotlib)
#    mamba/tcn_mamba 需要 CUDA。

# 1) 生成数据集 (拉丁超立方 + 多进程仿真)
python pc_to_data.py --smoke              # 200 样本冒烟 -> data/dataset_smoke.npz
python pc_to_data.py --full --workers 22  # 20000 正式集 -> data/dataset_full.npz

# 2) 训练 (四个模型分别训练)
python train.py --model tcn       --data full
python train.py --model lstm      --data full
python train.py --model mamba     --data full
python train.py --model tcn_mamba --data full

# 3) 评估 (指标表 + 预测图; 提出模型额外输出 MC UQ)
python evaluate.py --ckpt result/checkpoints/tcn_mamba.pt --data full
```

> **正式数据集耗时**：单样本仿真 (NM=100, dt=4e-5) 约 30–60 s 计算量。`pc_to_data.py`
> 已强制 `OMP_NUM_THREADS=1` 避免 BLAS 超额订阅；24 核约 22 worker 下，20000 样本
> 预计约 6–10 小时（建议放后台运行）。冒烟集 200 样本约 10 分钟。

## 7. 评价指标

`metrics.py` 严格按题述公式实现，逐输出通道 + overall：
RRMSE、R²、RDE_Std%、RDE_Ave%、RMSE（题述为 MSE 形式）、RMAE（MAE 形式）、
Erel（逐点相对误差均值）。

> 说明：题述 RRMSE/RMSE/RMAE 的命名与文献惯例略有出入（题述 RRMSE 实为 RMSE、
> RMSE 实为 MSE、RMAE 实为 MAE），代码 **忠实于题述公式** 并在 `metrics.py` 注明。

---

详见 [发表分析_中文EI.md](发表分析_中文EI.md)（可发表性评估与提升建议）。
