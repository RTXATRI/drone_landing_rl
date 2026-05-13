#!/usr/bin/env python3
# 功能：解析命令行参数并启动 SAC 课程学习训练流程。
"""
训练入口。

示例：
    # 只训练起始课程
    python scripts/train.py

    # 从指定课程开始，并恢复某个 checkpoint
    python scripts/train.py --stage 3 --resume output/models/drone_landing/ckpt_0000500000

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


def _parse_stage_envs(value: str) -> dict:
    """Parse CLI values like '1=50,2=50,3=32'."""
    result = {}
    if not value:
        return result
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "Expected comma-separated STAGE=NUM pairs, e.g. 1=50,2=50"
            )
        stage_text, count_text = item.split("=", 1)
        try:
            stage = int(stage_text.strip())
            count = int(count_text.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Stage and env count must be integers, e.g. 1=50,2=50"
            ) from exc
        if count < 1:
            raise argparse.ArgumentTypeError("Eval env count must be >= 1")
        result[stage] = count
    return result


def parse_args() -> argparse.Namespace:
    from configs.train_config import TrainConfig
    from curriculum.strategies import registered_stage_ids

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
    p.add_argument("--mix_ratio",   type=float, default=0.0,
                   help="穿插训练比例：使用前一课程策略的回合占比（0.0-1.0，默认 0.0 不开启）")
    p.add_argument("--csv_dir",      type=str,   default=defaults.csv_dir,
                   help="CSV output root directory")
    p.add_argument(
        "--reward_step_csv",
        action="store_true",
        default=defaults.reward_step_csv_enabled,
        help="Enable per-step reward breakdown CSV logging",
    )
    p.add_argument(
        "--best_model_selection",
        action=argparse.BooleanOptionalAction,
        default=defaults.best_model_selection_enabled,
        help="Enable per-stage best-model candidate collection and final evaluation",
    )
    p.add_argument("--best_grid_count", type=int, default=defaults.best_model_grid_count,
                   help="Number of evenly spaced per-stage best-model grid candidates")
    p.add_argument("--best_interval_top_m", type=int,
                   default=defaults.best_model_interval_top_m,
                   help="Number of high-reward candidates kept in each grid interval")
    p.add_argument("--best_eval_episodes", type=int,
                   default=defaults.best_model_eval_episodes,
                   help="Episodes per candidate in final best-model evaluation")
    p.add_argument("--best_eval_envs", type=_parse_stage_envs,
                   default=dict(defaults.best_model_eval_envs_by_stage),
                   help="Per-stage parallel eval envs, e.g. 1=50,2=50,3=32")
    p.add_argument("--best_keep_top_n", type=int,
                   default=defaults.best_model_keep_top_n,
                   help="Number of final top models to keep after best-model evaluation")
    p.add_argument("--lr_decay", action=argparse.BooleanOptionalAction,
                   default=defaults.sac.lr_decay,
                   help="Enable learning rate decay (constant -> cosine -> flat)")
    p.add_argument("--lr_decay_start", type=float,
                   default=defaults.sac.lr_decay_start,
                   help="Training progress to start LR decay (0.0-1.0, e.g. 0.7=70%% done)")
    p.add_argument("--lr_decay_end", type=float,
                   default=defaults.sac.lr_decay_end,
                   help="Training progress to reach min LR (0.0-1.0, e.g. 0.95=95%% done)")
    p.add_argument("--lr_decay_min_ratio", type=float,
                   default=defaults.sac.lr_decay_min_ratio,
                   help="Minimum LR ratio after decay")
    args = p.parse_args()
    if args.max_stage < args.stage:
        p.error("--max_stage must be greater than or equal to --stage")
    if args.best_grid_count < 1:
        p.error("--best_grid_count must be >= 1")
    if args.best_interval_top_m < 0:
        p.error("--best_interval_top_m must be >= 0")
    if args.best_eval_episodes < 1:
        p.error("--best_eval_episodes must be >= 1")
    if args.best_keep_top_n < 1:
        p.error("--best_keep_top_n must be >= 1")
    if args.lr_decay:
        if not (0.0 < args.lr_decay_start < args.lr_decay_end <= 1.0):
            p.error("--lr_decay_start must be in (0, --lr_decay_end) and --lr_decay_end in (0, 1]")
        if not (0.0 < args.lr_decay_min_ratio < 1.0):
            p.error("--lr_decay_min_ratio must be in (0, 1)")
    unknown_best_eval_stages = sorted(set(args.best_eval_envs) - set(stage_ids))
    if unknown_best_eval_stages:
        p.error(f"--best_eval_envs contains unknown stages: {unknown_best_eval_stages}")
    merged_best_eval_envs = dict(defaults.best_model_eval_envs_by_stage)
    merged_best_eval_envs.update(args.best_eval_envs)
    args.best_eval_envs = merged_best_eval_envs
    return args


def main() -> None:
    from configs.env_config import EnvConfig
    from configs.train_config import CurriculumConfig, SACConfig, TrainConfig
    from training.trainer import Trainer

    args = parse_args()

    # ── 环境配置（使用默认值；物理参数优先在 env_config.py 中调整）
    env_config = EnvConfig()

    # ── 训练配置
    sac_cfg = SACConfig(
        learning_rate = args.lr,
        batch_size = args.batch_size,
        lr_decay = args.lr_decay,
        lr_decay_start = args.lr_decay_start,
        lr_decay_end = args.lr_decay_end,
        lr_decay_min_ratio = args.lr_decay_min_ratio,
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
        reward_step_csv_enabled = args.reward_step_csv,
        best_model_selection_enabled = args.best_model_selection,
        best_model_grid_count = args.best_grid_count,
        best_model_interval_top_m = args.best_interval_top_m,
        best_model_eval_episodes = args.best_eval_episodes,
        best_model_eval_envs_by_stage = args.best_eval_envs,
        best_model_keep_top_n = args.best_keep_top_n,
        sac = sac_cfg,
        curriculum = cur_cfg,
    )

    # ── 启动
    trainer = Trainer(
        env_config = env_config,
        train_config = train_config,
        resume_path = args.resume,
        start_stage = args.stage,
        mix_ratio = args.mix_ratio,
    )
    trainer.train()


if __name__ == "__main__":
    main()
