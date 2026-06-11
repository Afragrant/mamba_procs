"""tune.py —— 用 Optuna 做超参/结构搜索

对指定模型搜索学习率、宽度、深度、dropout 等, 以验证集加权 MSE 为目标,
带中位数剪枝. 找到后保存最优超参 (json); 可选用最优超参全量重训并存权重.

用法:
    python tune.py --model cnn_mamba3 --data full --trials 40 --tune-epochs 25
    python tune.py --model lstm --data full --trials 30
    python tune.py --model cnn_mamba3 --data full --trials 40 --final-train
"""

from __future__ import annotations

import argparse
import json

import optuna
import torch
import torch.nn as nn

import config as C
from data_utils import load_dataset
from models import build_tuned
from train import evaluate_loss, make_loss, resolve_data, set_seed


# --------------------------------------------------------------------------- #
# 由 trial 采样的超参构建模型 (容量范围比默认大, 以追平论文的大模型)
# --------------------------------------------------------------------------- #
def suggest_params(trial, name):
    p = {'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True), 'dropout': trial.suggest_float('dropout', 0.05, 0.5)}
    if name == 'tcn':
        p['width'] = trial.suggest_categorical('width', [64, 96, 128, 192])
        p['depth'] = trial.suggest_int('depth', 3, 6)
        p['kernel'] = trial.suggest_categorical('kernel', [3, 5, 7])
    elif name == 'lstm':
        p['hidden'] = trial.suggest_categorical('hidden', [128, 256, 384, 512])
        p['layers'] = trial.suggest_int('layers', 1, 3)
    elif name == 'mamba3':
        p['d_model'] = trial.suggest_categorical('d_model', [64, 128, 192, 256])
        p['n_layers'] = trial.suggest_int('n_layers', 2, 6)
        p['d_state'] = trial.suggest_categorical('d_state', [32, 64, 128])
        p['headdim'] = trial.suggest_categorical('headdim', [32, 64])
    elif name == 'cnn_mamba3':
        # 存 code 字符串, build_tuned 据 CNN_*_OPTS 还原成元组
        p['cnn_channels'] = trial.suggest_categorical('cnn_channels', ['a', 'b', 'c'])
        p['cnn_kernels'] = trial.suggest_categorical('cnn_kernels', ['a', 'b', 'c'])
        p['d_model'] = trial.suggest_categorical('d_model', [64, 128, 192, 256])
        p['n_mamba'] = trial.suggest_int('n_mamba', 2, 6)
        p['d_state'] = trial.suggest_categorical('d_state', [32, 64, 128])
        p['headdim'] = trial.suggest_categorical('headdim', [32, 64])
    return p


build_from_params = build_tuned  # 复用 models 中的统一构建器


def train_short(model, loaders, lr, device, epochs, stats, trial=None, restore_best=False):
    """训练 epochs 轮, 返回最优 val_loss; 支持 Optuna 剪枝与早停.

    restore_best=True 时, 结束前把模型权重回滚到验证集最优 (供 --final-train 保存).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=C.LR_FACTOR, patience=C.LR_PATIENCE, min_lr=C.LR_MIN
    )
    loss_fn = make_loss(stats).to(device)
    best, best_state, patience = float('inf'), None, 0
    for ep in range(epochs):
        model.train()
        for x, y in loaders['train']:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
        val = evaluate_loss(model, loaders['val'], loss_fn, device)
        sched.step(val)
        if val < best - 1e-9:
            best, patience = val, 0
            if restore_best:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if trial is not None:
            trial.report(val, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if patience >= C.EARLY_STOP_PATIENCE:
            break
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['tcn', 'lstm', 'mamba3', 'cnn_mamba3'])
    ap.add_argument('--data', default='full')
    ap.add_argument('--trials', type=int, default=40)
    ap.add_argument('--tune-epochs', type=int, default=25)
    ap.add_argument('--final-epochs', type=int, default=C.EPOCHS, help='--final-train 时的训练轮数')
    ap.add_argument('--device', default=C.DEVICE)
    ap.add_argument('--final-train', action='store_true', help='用最优超参全量重训并保存权重')
    # cnn_mamba3 消融条件 (固定, 不进搜索空间). 缺省 = 提出模型(双向+共享权重).
    ap.add_argument(
        '--bidirectional',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='cnn_mamba3: 双向扫描 (--no-bidirectional 为单向消融基线)',
    )
    ap.add_argument(
        '--tie-weights',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='cnn_mamba3: 双向时正反向共享权重 (--no-tie-weights 为 Vim 独立权重)',
    )
    ap.add_argument('--tag', default='', help='产物名后缀, 区分消融配置 (如 _uni / _bisep)')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    set_seed(C.SEED)
    loaders, stats = load_dataset(resolve_data(args.data), batch_size=C.BATCH_SIZE)

    # 架构消融条件: 注入到超参字典随 trial 一起构建, 但不由 Optuna 采样.
    arch = {}
    if args.model == 'cnn_mamba3':
        arch = {'bidirectional': args.bidirectional, 'tie_weights': args.tie_weights}

    def objective(trial):
        p = suggest_params(trial, args.model)
        p.update(arch)
        model = build_from_params(args.model, p).to(device)
        return train_short(model, loaders, p['lr'], device, args.tune_epochs, stats, trial=trial)

    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=args.trials)

    print('\n===== 搜索完成 =====')
    print('最优 val_loss:', study.best_value)
    print('最优超参:', study.best_params)
    if arch:
        print('架构消融条件:', arch)
    # best_params 合入固定的架构消融条件, 供 evaluate.py 按相同结构重建.
    best_params = {**study.best_params, **arch}
    out = C.CKPT_DIR / f'{args.model}{args.tag}_best_params.json'
    out.write_text(json.dumps(best_params, indent=2, ensure_ascii=False))
    print('已保存:', out)

    if args.final_train:
        print('\n用最优超参全量重训 ...')
        p = best_params
        model = build_from_params(args.model, p).to(device)
        best = train_short(model, loaders, p['lr'], device, args.final_epochs, stats, restore_best=True)
        ckpt = C.CKPT_DIR / f'{args.model}{args.tag}.pt'
        torch.save(
            {
                'model_name': args.model,
                'state_dict': model.state_dict(),
                'stats': stats,
                'best_val': best,
                'best_params': p,
                'data_path': str(resolve_data(args.data)),
            },
            ckpt,
        )
        print(f'最优超参全量重训完成, val={best:.6f}, 已存 {ckpt}')
        print('⚠️ 注意: 该权重用搜索时的网络结构, evaluate.py 需用相同结构加载 (见下).')


if __name__ == '__main__':
    main()
