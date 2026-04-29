#!/usr/bin/env python3
# 功能：解析命令行参数并启动 SAC 课程学习训练流程。
"""
训练入口。

示例：
    # 只训练起始课程
    python scripts/train.py

    # 从指定课程开始，并恢复某个 checkpoint
    python scripts/train.py --stage 3 --resume models/drone_landing/ckpt_0000500000

    # 允许在课程之间手动继续
    python scripts/train.py --stage 1 --max_stage 4

    # 快速 smoke test（8 个环境，100k 步）
    python scripts/train.py --n_envs 8 --total_steps 100000 --exp_name smoke_test

    # 禁用 GPU（调试）
    python scripts/train.py --device cpu --n_envs 4
"""

import argparse
import sys
import os

# ── 将项目根目录加入 Python path ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.env_config import EnvConfig
from configs.train_config import CurriculumConfig, SACConfig, TrainConfig
from curriculum.strategies import registered_stage_ids
from training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(
        description="Train drone landing RL agent (SAC + Curriculum)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    stage_ids = list(registered_stage_ids())
    p.add_argument("--stage",        type=int,   default=stage_ids[0],
                   choices=stage_ids,  help="Starting curriculum stage")
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint (without .zip)")
    p.add_argument("--n_envs",       type=int,   default=defaults.n_envs,
                   help="Number of parallel environments")
    p.add_argument("--device",       type=str,   default=defaults.device,
                   choices=["cuda", "cpu"],
                   help="Training device")
    p.add_argument("--exp_name",     type=str,   default=defaults.exp_name,
                   help="Experiment name for logging subdirectory")
    p.add_argument("--seed",         type=int,   default=defaults.seed,
                   help="Random seed")
    p.add_argument("--total_steps",  type=int,   default=None,
                   help="Override timestep budget for each selected stage")
    p.add_argument("--max_stage",   type=int,   default=defaults.curriculum.max_stage,
                   choices=stage_ids,
                   help="Highest stage allowed for manual continuation")
    p.add_argument("--batch_size",   type=int,   default=defaults.sac.batch_size,
                   help="SAC replay buffer sample batch size")
    p.add_argument("--lr",           type=float, default=defaults.sac.learning_rate,
                   help="SAC learning rate")
    p.add_argument("--log_dir",      type=str,   default=defaults.log_dir,
                   help="TensorBoard and runtime log root directory")
    p.add_argument("--model_dir",    type=str,   default=defaults.model_dir,
                   help="Checkpoint and final model root directory")
    p.add_argument("--csv_dir",      type=str,   default=defaults.csv_dir,
                   help="CSV output root directory")
    args = p.parse_args()
    if args.max_stage < args.stage:
        p.error("--max_stage must be greater than or equal to --stage")
    return args


def main() -> None:
    args = parse_args()

    # ── 环境配置（使用默认值；物理参数优先在 env_config.py 中调整）
    env_config = EnvConfig()

    # ── 训练配置
    sac_cfg = SACConfig(
        learning_rate = args.lr,
        batch_size = args.batch_size,
    )

    cur_cfg = CurriculumConfig(max_stage = args.max_stage)
    if args.total_steps is not None:
        # 在手动课程模式中，--total_steps 是每个阶段的预算。
        cur_cfg.stage_timesteps = [args.total_steps] * len(cur_cfg.stage_timesteps)

    train_config = TrainConfig(
        n_envs = args.n_envs,
        device = args.device,
        exp_name = args.exp_name,
        seed = args.seed,
        log_dir = args.log_dir,
        model_dir = args.model_dir,
        csv_dir = args.csv_dir,
        sac = sac_cfg,
        curriculum = cur_cfg,
    )

    # ── 启动
    trainer = Trainer(
        env_config = env_config,
        train_config = train_config,
        resume_path = args.resume,
        start_stage = args.stage,
    )
    trainer.train()


if __name__ == "__main__":
    main()
