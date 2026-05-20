#!/usr/bin/env python3
# 功能：按网格或随机方式启动多组训练实验以进行超参数扫参。
"""
用于消融实验的超参数扫参运行器。

按顺序或并行启动多组训练，每组训练都有独立实验名和配置覆盖。

支持网格搜索和随机搜索。

示例：比较平滑系数：
    python scripts/sweep.py \
        --mode grid \
        --param cmd_filter_alpha 0.0 0.4 0.7 0.9 \
        --total_steps 300000 \
        --n_envs 8

示例：随机搜索 10 组 LR/batch 组合：
    python scripts/sweep.py \
        --mode random \
        --n_trials 10 \
        --param lr 1e-4 1e-3 \
        --param batch_size 256 1024 \
        --total_steps 300000
"""

import argparse
import copy
import itertools
import os
import subprocess
import sys
import random
import math
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter sweep for ablation studies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",        choices=["grid", "random"], default="grid")
    p.add_argument("--n_trials",    type=int, default=10,
                   help="Number of random search trials (only for --mode random)")
    p.add_argument("--param",       action="append", nargs="+", metavar=("NAME", "VAL"),
                   help="Parameter to sweep: --param name v1 v2 ...")
    p.add_argument("--total_steps", type=int, default=300_000)
    p.add_argument("--n_envs",      type=int, default=8)
    p.add_argument("--stage",       type=int, default=1)
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--log_dir",     type=str, default="./output/logs/sweep")
    p.add_argument("--parallel",    type=int, default=1,
                   help="Number of runs to launch in parallel (use with care — GPU memory)")
    return p.parse_args()


def build_configs_grid(params: list) -> list:
    """构建网格搜索的所有组合。"""
    names  = [p[0]    for p in params]
    values = [p[1:]   for p in params]
    combos = list(itertools.product(*values))
    return [{n: v for n, v in zip(names, combo)} for combo in combos]


def build_configs_random(params: list, n_trials: int) -> list:
    """采样随机组合。"""
    configs = []
    for _ in range(n_trials):
        cfg = {}
        for p in params:
            name   = p[0]
            lo, hi = float(p[1]), float(p[2])
            # LR 类参数使用 log-uniform，其它参数使用线性采样
            if "lr" in name or "learning_rate" in name:
                val = 10 ** random.uniform(math.log10(lo), math.log10(hi))
            else:
                val = random.uniform(lo, hi)
            cfg[name] = val
        configs.append(cfg)
    return configs


PARAM_TO_FLAG = {
    "cmd_filter_alpha": None,       # env config 中的参数，暂无直接 CLI flag
    "lr":               "--lr",
    "learning_rate":    "--lr",
    "batch_size":       "--batch_size",
    "n_envs":           "--n_envs",
}


def config_to_exp_name(cfg: dict, timestamp: str, idx: int) -> str:
    parts = [f"sweep_{timestamp}"]
    for k, v in cfg.items():
        val_str = f"{float(v):.5g}".replace("-", "m")
        parts.append(f"{k}_{val_str}")
    parts.append(f"run{idx:03d}")
    return "_".join(parts)[:80]   # 保持名称较短


def launch_run(cfg: dict, args, idx: int, timestamp: str) -> subprocess.Popen:
    exp_name = config_to_exp_name(cfg, timestamp, idx)
    cmd = [
        sys.executable, os.path.join(ROOT, "scripts", "train.py"),
        "--exp_name",    exp_name,
        "--total_steps", str(args.total_steps),
        "--n_envs",      str(args.n_envs),
        "--stage",       str(args.stage),
        "--device",      args.device,
        "--log_dir",     args.log_dir,
    ]

    # 将已知参数映射到 CLI flag
    for name, val in cfg.items():
        flag = PARAM_TO_FLAG.get(name)
        if flag:
            cmd += [flag, str(val)]
        # 其它 env-config 参数需要 monkey-patching（高级用法）

    log_file = os.path.join(args.log_dir, f"{exp_name}.log")
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"\n[{idx+1:3d}] Launching: {exp_name}")
    print(f"     Config : {cfg}")
    print(f"     Log    : {log_file}")

    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

    return proc


def main():
    args = parse_args()
    if not args.param:
        print("ERROR: No parameters specified. Use --param name v1 v2 ...")
        sys.exit(1)

    timestamp = datetime.now().strftime("%m%d_%H%M")

    if args.mode == "grid":
        configs = build_configs_grid(args.param)
        print(f"Grid search: {len(configs)} combinations")
    else:
        configs = build_configs_random(args.param, args.n_trials)
        print(f"Random search: {args.n_trials} trials")

    if args.parallel == 1:
        # 顺序运行
        for i, cfg in enumerate(configs):
            proc = launch_run(cfg, args, i, timestamp)
            ret  = proc.wait()
            print(f"  Run {i+1} finished (exit code {ret})")
    else:
        # 并行运行（限制进程池大小）
        pool = []
        for i, cfg in enumerate(configs):
            while len(pool) >= args.parallel:
                # 轮询已结束的进程
                pool = [p for p in pool if p.poll() is None]
            pool.append(launch_run(cfg, args, i, timestamp))
        for p in pool:
            p.wait()

    print(f"\nSweep complete. All logs in: {args.log_dir}")
    print(f"View all runs: tensorboard --logdir {args.log_dir.replace('sweep','')}")


if __name__ == "__main__":
    main()
