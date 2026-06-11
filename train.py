"""train.py —— 训练任一模型 (tcn / lstm / mamba3 / cnn_mamba3).

归一化空间 MSE 损失, AdamW, ReduceLROnPlateau, 梯度裁剪, 早停, 保存最优权重.

用法:
    python train.py --model cnn_mamba3 --data full
    python train.py --model lstm --data full --epochs 100
    python train.py --model mamba3 --data data/dataset_5000.npz
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

import config as C
from data_utils import load_dataset
from models import build_model


def resolve_data(arg: str):
    if arg == 'smoke':
        return C.DATASET_SMOKE
    if arg == 'full':
        return C.DATASET_FULL
    return arg


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WeightedMSELoss(nn.Module):
    """逐通道加权 MSE (+ 可选物理一致性 + 可选 Fc 频域损失).

    weights (C,) 对应 (Fc, y_panto, y_cat). 输出已归一化到 [-1,1], 各通道尺度可比,
    权重直接反映"重视程度": 接触力 Fc 最难最关键, 权重最大.

    y_min/y_max (C,): 输出反归一化区间. 提供后才能启用物理项 (须在物理空间算 Fc).
    lambda_phys: 物理一致性权重. Fc 应 = KS·relu(y_panto - y_cat); 把三通道反归一化到
        物理空间算出重构 Fc, 再归一化回 [-1,1] 与模型的 Fc 输出比. 0 则关闭.
    lambda_freq: Fc 频域幅值损失 (ortho 归一 rfft 的 |·| 的 L1), 抑制过平滑. 0 则关闭.
    """

    def __init__(self, weights, y_min=None, y_max=None,
                 lambda_phys=0.0, lambda_freq=0.0, ks=None):
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float32)
        self.register_buffer('w', w / w.mean())  # 归一化, 不改变整体损失量级
        self.lambda_phys = float(lambda_phys)
        self.lambda_freq = float(lambda_freq)
        self.ks = float(ks) if ks is not None else 0.0
        if y_min is not None and y_max is not None:
            self.register_buffer('y_min', torch.as_tensor(y_min, dtype=torch.float32))
            self.register_buffer('y_max', torch.as_tensor(y_max, dtype=torch.float32))
        else:
            self.y_min = self.y_max = None

    def _denorm(self, y_norm, c):              # (B,T) 归一化通道 c -> 物理值
        return (y_norm + 1.0) / 2.0 * (self.y_max[c] - self.y_min[c]) + self.y_min[c]

    def _norm_fc(self, fc_phys):               # 物理 Fc -> 归一化 [-1,1] (通道 0)
        rng = self.y_max[0] - self.y_min[0]
        return 2.0 * (fc_phys - self.y_min[0]) / rng - 1.0

    def forward(self, pred, target):           # (B, T, C); 通道序 (Fc, y_panto, y_cat)
        se = (pred - target) ** 2
        loss = (se * self.w).mean()
        # 方案三: 物理一致性. 用预测的位移之差重构 Fc, 逼模型三通道自洽 -> 把 Fc 的梯度
        # 沿物理关系灌入 yp、yc 之差, 缓解灾难性抵消.
        if self.lambda_phys > 0 and self.y_min is not None:
            yp = self._denorm(pred[..., 1], 1)
            yc = self._denorm(pred[..., 2], 2)
            fc_recon = self._norm_fc(self.ks * torch.relu(yp - yc))
            loss = loss + self.lambda_phys * ((pred[..., 0] - fc_recon) ** 2).mean()
        # 方案二: Fc 频域幅值损失. ortho 归一使幅值与信号同量级, λ 可控.
        if self.lambda_freq > 0:
            pf = torch.fft.rfft(pred[..., 0], dim=1, norm='ortho').abs()
            tf = torch.fft.rfft(target[..., 0], dim=1, norm='ortho').abs()
            loss = loss + self.lambda_freq * (pf - tf).abs().mean()
        return loss


def make_loss(stats):
    """按 config 与数据集统计量构建训练损失 (train.py / tune.py 共用, 保证目标一致)."""
    return WeightedMSELoss(C.CHANNEL_LOSS_WEIGHTS,
                           y_min=stats['y_min'], y_max=stats['y_max'],
                           lambda_phys=C.LAMBDA_PHYS, lambda_freq=C.LAMBDA_FREQ,
                           ks=C.CONTACT_KS)


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total += loss_fn(pred, y).item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def train(model_name, data_path, epochs, batch_size, lr, device, out_ckpt):
    set_seed(C.SEED)
    loaders, stats = load_dataset(data_path, batch_size=batch_size)
    model = build_model(model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'模型 {model_name}: {n_params:,} 参数  | 设备 {device}')

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=C.LR_FACTOR, patience=C.LR_PATIENCE, min_lr=C.LR_MIN)
    loss_fn = make_loss(stats).to(device)
    print(f'通道损失权重 (Fc, y_panto, y_cat) = {C.CHANNEL_LOSS_WEIGHTS}; '
          f'λ_phys={C.LAMBDA_PHYS}, λ_freq={C.LAMBDA_FREQ}')

    best_val, best_state, patience = float('inf'), None, 0
    history = {'train': [], 'val': []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        run, n = 0.0, 0
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            run += loss.item() * x.size(0)
            n += x.size(0)
        tr = run / max(n, 1)
        va = evaluate_loss(model, loaders['val'], loss_fn, device)
        sched.step(va)
        history['train'].append(tr)
        history['val'].append(va)

        flag = ''
        if va < best_val - 1e-9:
            best_val, patience = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = ' *'
        else:
            patience += 1
        lr_now = opt.param_groups[0]['lr']
        print(f'  ep {ep:3d}/{epochs}  train {tr:.6f}  val {va:.6f}  lr {lr_now:.2e}{flag}')
        if patience >= C.EARLY_STOP_PATIENCE:
            print(f'  早停于 epoch {ep} (val {C.EARLY_STOP_PATIENCE} 轮无改善)')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    el = time.time() - t0
    print(f'训练完成: {el / 60:.1f} min, 最优 val_loss={best_val:.6f}')

    torch.save({
        'model_name': model_name,
        'state_dict': model.state_dict(),
        'config': {k: getattr(C, k) for k in
                   ('IN_CHANNELS', 'OUT_CHANNELS', 'SEQ_LEN', 'MODEL_CFG')},
        'stats': stats,
        'history': history,
        'best_val': best_val,
        'data_path': str(data_path),
    }, out_ckpt)
    print(f'已保存权重: {out_ckpt}')
    return out_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    choices=['tcn', 'lstm', 'mamba3', 'cnn_mamba3'])
    ap.add_argument('--data', default='smoke', help='smoke / full / npz 路径')
    ap.add_argument('--epochs', type=int, default=C.EPOCHS)
    ap.add_argument('--batch-size', type=int, default=C.BATCH_SIZE)
    ap.add_argument('--lr', type=float, default=C.LR)
    ap.add_argument('--device', default=C.DEVICE)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    out = args.out or (C.CKPT_DIR / f'{args.model}.pt')
    train(args.model, resolve_data(args.data), args.epochs,
          args.batch_size, args.lr, device, out)


if __name__ == '__main__':
    main()
