"""train.py —— 训练任一模型 (tcn / lstm / mamba / tcn_mamba).

归一化空间 MSE 损失, AdamW, ReduceLROnPlateau, 梯度裁剪, 早停, 保存最优权重.

用法:
    python train.py --model tcn_mamba --data smoke
    python train.py --model lstm --data full --epochs 100
    python train.py --model mamba --data data/dataset_5000.npz
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
    """逐输出通道加权 MSE. weights 形状 (C,), 对应 (Fc, y_panto, y_cat).

    输出已归一化到 [-1,1], 各通道尺度可比, 故权重直接反映"重视程度":
    接触力 Fc 最难且最关键, 权重最大.
    """

    def __init__(self, weights):
        super().__init__()
        w = torch.as_tensor(weights, dtype=torch.float32)
        self.register_buffer('w', w / w.mean())  # 归一化, 不改变整体损失量级

    def forward(self, pred, target):           # (B, T, C)
        se = (pred - target) ** 2
        return (se * self.w).mean()


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
    if hasattr(model, 'set_norm_stats'):   # 物理约束模型需注入 Min-Max 统计量
        model.set_norm_stats(stats)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'模型 {model_name}: {n_params:,} 参数  | 设备 {device}')

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=C.LR_FACTOR, patience=C.LR_PATIENCE, min_lr=C.LR_MIN)
    loss_fn = WeightedMSELoss(C.CHANNEL_LOSS_WEIGHTS).to(device)
    print(f'通道损失权重 (Fc, y_panto, y_cat) = {C.CHANNEL_LOSS_WEIGHTS}')

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
                    choices=['tcn', 'lstm', 'mamba', 'tcn_mamba', 'pc_tcn_mamba'])
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
