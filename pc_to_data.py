"""pc_to_data.py —— 由 pc.py 改写, 生成刚性接触网代理模型的训练数据集.

流程:
  1. 拉丁超立方设计 (scipy.stats.qmc.LatinHypercube) 在 5 个设计变量范围内采样;
  2. 对每个参数组合调用 pc.run_simulation, 取稳定段 (30%~70%) 的 3 个输出序列;
  3. 将稳定段重采样到固定长度 SEQ_LEN (稳定段恒覆盖 0.4*N_SPANS=12 跨, 相位对齐);
  4. 写出 .npz: 设计参数 (N,5)、目标序列 (N,T,3)、划分索引、归一化统计量.

多进程并行 (默认用全部物理核). 单样本耗时 ~5s (NM=100, dt=4e-5),
20000 样本在 24 核上约 5 小时.

用法:
    python pc_to_data.py --smoke           # 200 样本冒烟测试 -> data/dataset_smoke.npz
    python pc_to_data.py --full            # 20000 样本正式集 -> data/dataset_full.npz
    python pc_to_data.py --n 1000 --out data/foo.npz --workers 12
"""

from __future__ import annotations

import os

# 每个仿真本身是小稠密线性代数; 多进程并行时必须限制 BLAS 线程数为 1,
# 否则 20 进程 × 多线程会严重超额订阅 CPU. 必须在 import numpy 之前设置.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

import config as C
from pc import run_simulation

# 固定的支撑等效质量 MEQ (随预设默认值, 不作为设计变量)
MEQ_FIXED = 7.0


# --------------------------------------------------------------------------- #
# 采样
# --------------------------------------------------------------------------- #
def latin_hypercube(n: int, seed: int) -> np.ndarray:
    """返回 (n, 5) 设计参数, 列顺序 = config.INPUT_NAMES = (KEQ, L, V, EI, rhoA).

    C.LOG_SAMPLE_VARS 中的变量在对数空间均匀采样 (各数量级均衡), 其余线性采样.
    """
    sampler = qmc.LatinHypercube(d=C.N_DESIGN, seed=seed)
    unit = sampler.random(n)  # (n, 5) in [0,1]
    out = np.empty_like(unit)
    for j, name in enumerate(C.INPUT_NAMES):
        lo, hi = C.INPUT_RANGES[name]
        if name in C.LOG_SAMPLE_VARS:
            out[:, j] = 10.0 ** (np.log10(lo) + unit[:, j] * (np.log10(hi) - np.log10(lo)))
        else:
            out[:, j] = lo + unit[:, j] * (hi - lo)
    return out


def _resample(arr: np.ndarray, t: int) -> np.ndarray:
    """把任意长度序列线性重采样到长度 t."""
    src = np.linspace(0.0, 1.0, num=len(arr))
    dst = np.linspace(0.0, 1.0, num=t)
    return np.interp(dst, src, arr)


# --------------------------------------------------------------------------- #
# 单样本仿真 (多进程 worker, 必须为模块级函数)
# --------------------------------------------------------------------------- #
def simulate_one(args: tuple[int, np.ndarray]) -> tuple[int, np.ndarray | None]:
    idx, design = args
    keq, l, v, ei, rhoa = (float(x) for x in design)
    try:
        # run_simulation 的 catenary_params 顺序 = (L, rhoA, EI, KEQ, MEQ)
        res = run_simulation(
            catenary_params=(l, rhoa, ei, keq, MEQ_FIXED),
            pantograph=C.PANTOGRAPH_TYPE,
            speed_kmh=v,
            NM=C.NM,
            N_spans=C.N_SPANS,
            dt_base=C.DT_BASE,
            verbose=False,
        )
        fc = _resample(res['contact_force_stable'], C.SEQ_LEN)
        yp = _resample(res['y_pantograph_stable'], C.SEQ_LEN)
        yc = _resample(res['y_rigid_overhead_contact_system_stable'], C.SEQ_LEN)
        seq = np.stack([fc, yp, yc], axis=-1).astype(np.float32)  # (T, 3)
        if not np.all(np.isfinite(seq)):
            return idx, None
        return idx, seq
    except Exception:
        return idx, None


# --------------------------------------------------------------------------- #
# 归一化统计量 (输入用 LHS 边界; 输出用训练集统计量)
# --------------------------------------------------------------------------- #
def compute_norm_stats(design_train: np.ndarray, targets_train: np.ndarray) -> dict:
    x_min = np.array([C.INPUT_RANGES[k][0] for k in C.INPUT_NAMES], dtype=np.float64)
    x_max = np.array([C.INPUT_RANGES[k][1] for k in C.INPUT_NAMES], dtype=np.float64)
    # 输出每通道全局 min/max (在 train 上)
    y_min = targets_train.reshape(-1, C.OUT_CHANNELS).min(axis=0).astype(np.float64)
    y_max = targets_train.reshape(-1, C.OUT_CHANNELS).max(axis=0).astype(np.float64)
    return dict(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


# --------------------------------------------------------------------------- #
# 生成
# --------------------------------------------------------------------------- #
def generate(n: int, split: dict, out_path, workers: int, seed: int):
    print('=' * 64)
    print(f'生成数据集: N={n}, 划分={split}')
    print(f'  保真度: NM={C.NM}, dt={C.DT_BASE:.0e}, 稳定段 {C.STABLE_START:.0%}-{C.STABLE_END:.0%}')
    print(f'  序列长度 SEQ_LEN={C.SEQ_LEN}, 受电弓={C.PANTOGRAPH_TYPE}, 跨数={C.N_SPANS}')
    print(f'  并行 workers={workers}')
    print('=' * 64)

    design = latin_hypercube(n, seed)
    targets = np.empty((n, C.SEQ_LEN, C.OUT_CHANNELS), dtype=np.float32)
    done = np.zeros(n, dtype=bool)

    t0 = time.time()
    tasks = list(enumerate(design))
    with Pool(processes=workers) as pool:
        completed = 0
        for idx, seq in pool.imap_unordered(simulate_one, tasks, chunksize=4):
            completed += 1
            if seq is not None:
                targets[idx] = seq
                done[idx] = True
            if completed % max(1, n // 50) == 0 or completed == n:
                el = time.time() - t0
                rate = completed / el
                eta = (n - completed) / rate if rate > 0 else 0
                print(f'  [{completed:>6}/{n}] ok={done.sum():>6} '
                      f'{el / 60:6.1f} min  ETA {eta / 60:6.1f} min', flush=True)

    # 补采样替换失败样本: 对失败位重新抽取设计参数并重跑, 直到全部成功
    if not done.all():
        print(f'  失败 {int((~done).sum())} 个样本, 重新采样补齐 ...')
        attempt = 0
        while not done.all() and attempt < 10:
            attempt += 1
            fail_idx = np.where(~done)[0]
            new_design = latin_hypercube(len(fail_idx), seed + 1000 + attempt)
            design[fail_idx] = new_design  # 用新参数覆盖失败位
            redo = [(int(i), design[i]) for i in fail_idx]
            with Pool(processes=workers) as pool:
                for ridx, seq in pool.imap_unordered(simulate_one, redo):
                    if seq is not None:
                        targets[ridx] = seq
                        done[ridx] = True
        if not done.all():
            raise RuntimeError(f'仍有 {int((~done).sum())} 个样本无法完成仿真')

    el = time.time() - t0
    print(f'仿真完成: {el / 60:.1f} min ({el / n:.2f} s/样本)')

    # 打乱并划分
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    design, targets = design[perm], targets[perm]
    n_tr, n_va, n_te = split['train'], split['val'], split['test']
    assert n_tr + n_va + n_te == n, '划分数量之和必须等于 N'
    idx_tr = np.arange(0, n_tr)
    idx_va = np.arange(n_tr, n_tr + n_va)
    idx_te = np.arange(n_tr + n_va, n)

    stats = compute_norm_stats(design[idx_tr], targets[idx_tr])

    np.savez_compressed(
        out_path,
        design=design.astype(np.float32),
        targets=targets.astype(np.float32),
        idx_train=idx_tr, idx_val=idx_va, idx_test=idx_te,
        input_names=np.array(C.INPUT_NAMES), output_names=np.array(C.OUTPUT_NAMES),
        seq_len=C.SEQ_LEN, nm=C.NM, dt=C.DT_BASE,
        x_min=stats['x_min'], x_max=stats['x_max'],
        y_min=stats['y_min'], y_max=stats['y_max'],
    )
    np.savez(C.NORM_STATS, **stats,
             input_names=np.array(C.INPUT_NAMES), output_names=np.array(C.OUTPUT_NAMES))
    print(f'已写出: {out_path}  (train={n_tr}, val={n_va}, test={n_te})')
    print(f'归一化统计量: {C.NORM_STATS}')
    print(f'  输出通道 y_min={stats["y_min"]}, y_max={stats["y_max"]}')


def main():
    ap = argparse.ArgumentParser(description='刚性接触网代理模型数据集生成')
    ap.add_argument('--smoke', action='store_true', help='生成小样本冒烟数据集')
    ap.add_argument('--full', action='store_true', help='生成 20000 正式数据集')
    ap.add_argument('--n', type=int, default=None, help='自定义样本数')
    ap.add_argument('--out', type=str, default=None, help='输出 npz 路径')
    ap.add_argument('--workers', type=int, default=max(1, (__import__('os').cpu_count() or 2) - 2))
    ap.add_argument('--seed', type=int, default=C.LHS_SEED)
    args = ap.parse_args()

    if args.full:
        n, split, out = C.N_SAMPLES_FULL, C.SPLIT_FULL, args.out or C.DATASET_FULL
    elif args.smoke:
        n, split, out = C.N_SAMPLES_SMOKE, C.SPLIT_SMOKE, args.out or C.DATASET_SMOKE
    elif args.n:
        n = args.n
        n_va = n_te = max(1, n // 10)
        split = {'train': n - 2 * n_va, 'val': n_va, 'test': n_te}
        out = args.out or (C.DATA_DIR / f'dataset_{n}.npz')
    else:
        ap.error('需指定 --smoke / --full / --n')

    generate(n, split, out, args.workers, args.seed)


if __name__ == '__main__':
    main()
