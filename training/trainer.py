# 功能：编排并行环境创建、SAC 模型初始化/恢复、训练和最终保存。
"""
训练编排器。

职责：
  - 构建向量化环境（SubprocVecEnv）以进行 CPU 并行采样
  - 初始化或恢复 SAC 模型（通过 torch 使用 GPU 加速）
  - 组合 callbacks，并按阶段启动 model.learn()
  - 在手动切换课程阶段前询问用户
  - 保存最终模型并记录课程学习总结
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from configs.env_config import EnvConfig
from configs.train_config import TrainConfig
from curriculum.curriculum_manager import CurriculumManager, STAGE_SHORT_LABELS
from curriculum.strategies import create_strategy, episode_info_keywords
from envs.drone_landing_env import DroneLandingEnv
from training.callbacks import (
    CheckpointCallback,
    CSVLoggingCallback,
    CurriculumCallback,
)
from utils.logger import setup_logger

logger = logging.getLogger(__name__)

EPISODE_INFO_KEYWORDS = episode_info_keywords()


# ─────────────────────────────────────────────────────────────────────────────
# 环境工厂辅助函数（spawn 模式下必须位于模块顶层以便 pickle）
# ─────────────────────────────────────────────────────────────────────────────

def _make_env_fn(env_config: EnvConfig, rank: int, seed: int, stage: int):
    """返回一个创建单个环境实例的闭包。"""
    def _init():
        env = DroneLandingEnv(
            env_config,
            strategy=create_strategy(stage, env_config),
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# 训练器
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    端到端训练流程。

    示例：
        trainer = Trainer(env_config, train_config)
        trainer.train()
    """

    def __init__(
        self,
        env_config: EnvConfig,
        train_config: TrainConfig,
        resume_path: Optional[str] = None,
        start_stage: int = 1,
    ):
        self.env_cfg      = env_config
        self.cfg          = train_config
        self.resume_path  = resume_path
        self.start_stage  = start_stage

        # 日志设置
        run_dir = os.path.join(train_config.log_dir, train_config.exp_name)
        setup_logger(run_dir)

        # 设备解析
        if train_config.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable — falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = train_config.device

        logger.info(f"Training device : {self.device}")
        if self.device == "cuda":
            logger.info(f"GPU             : {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM            : "
                        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 公共接口 ─────────────────────────────────────────────────────────────

    def train(self) -> None:
        """运行完整训练流程。"""
        selected_stages = range(self.start_stage, self.cfg.curriculum.max_stage + 1)
        planned_steps = sum(self._stage_budget(stage) for stage in selected_stages)

        logger.info("=" * 60)
        logger.info(f"  Experiment : {self.cfg.exp_name}")
        logger.info(f"  Envs       : {self.cfg.n_envs} parallel (SubprocVecEnv)")
        logger.info(f"  Device     : {self.device}")
        logger.info(f"  Stages     : {self.start_stage} → {self.cfg.curriculum.max_stage}")
        logger.info(f"  Max steps  : {planned_steps:,} if all selected stages continue")
        logger.info("=" * 60)

        # 1. 构建并行环境
        vec_env = self._build_vec_env(self.cfg.n_envs, self.start_stage)

        # 2. 构建或加载模型
        model = self._build_model(vec_env)

        # 3. 构建课程管理器和回调
        curriculum = CurriculumManager(self.cfg.curriculum)
        curriculum.set_stage(self.start_stage)
        callbacks = self._build_callbacks(curriculum)

        # 4. 按阶段训练；阶段切换由用户手动决定。
        reset_num_timesteps = self.resume_path is None
        try:
            for stage in selected_stages:
                curriculum.set_stage(stage)
                stage_obs = self._activate_stage(vec_env, stage)
                model._last_obs = stage_obs
                model._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)

                stage_steps = self._stage_budget(stage)
                logger.info("-" * 60)
                logger.info(
                    f"Stage {stage} {STAGE_SHORT_LABELS[stage]} started | "
                    f"budget={stage_steps:,} timesteps"
                )
                logger.info("-" * 60)

                model.learn(
                    total_timesteps = stage_steps,
                    callback = callbacks,
                    tb_log_name = self.cfg.exp_name,
                    reset_num_timesteps = reset_num_timesteps,
                    progress_bar = True,
                )
                reset_num_timesteps = False

                self._log_stage_finished(curriculum, stage, stage_steps)
                if stage < self.cfg.curriculum.max_stage:
                    if not self._confirm_next_stage(stage + 1):
                        logger.info("Manual curriculum stopped by user.")
                        break
        except KeyboardInterrupt:
            logger.info("Training interrupted by user — saving current model…")
        finally:
            self._save_final_model(model)
            curriculum.log_summary()
            vec_env.close()

    # ── 私有构建函数 ─────────────────────────────────────────────────────────

    def _build_vec_env(self, n_envs: int, stage: int) -> VecMonitor:
        logger.info(f"Spawning {n_envs} environment processes (stage {stage})…")
        fns = [
            _make_env_fn(self.env_cfg, rank=i, seed=self.cfg.seed, stage=stage)
            for i in range(n_envs)
        ]
        # spawn 可以避免 PyBullet 共享 C 库在 fork 下的相关问题
        vec_env = SubprocVecEnv(fns, start_method="spawn")
        vec_env = VecMonitor(vec_env, info_keywords=EPISODE_INFO_KEYWORDS)
        logger.info("Environments ready.")
        return vec_env

    def _activate_stage(self, vec_env: VecMonitor, stage: int) -> np.ndarray:
        """广播手动选择的阶段，并开始一批干净的新 episode。"""
        logger.info(f"Activating Stage {stage} {STAGE_SHORT_LABELS[stage]} on all envs.")
        vec_env.env_method("set_strategy", create_strategy(stage, self.env_cfg))
        return vec_env.reset()

    def _stage_budget(self, stage: int) -> int:
        """返回从 1 开始编号的课程阶段对应的配置步数预算。"""
        return int(self.cfg.curriculum.stage_timesteps[stage - 1])

    def _build_model(self, vec_env: VecMonitor) -> SAC:
        sac = self.cfg.sac
        tb_path = os.path.join(self.cfg.log_dir, self.cfg.exp_name)

        if self.resume_path and os.path.exists(self.resume_path + ".zip"):
            logger.info(f"Resuming from {self.resume_path}.zip")
            model = SAC.load(
                self.resume_path,
                env=vec_env,
                device=self.device,
                tensorboard_log=tb_path,
            )
        else:
            logger.info("Initializing new SAC model…")
            model = SAC(
                policy="MlpPolicy",
                env=vec_env,
                learning_rate=sac.learning_rate,
                buffer_size=sac.buffer_size,
                learning_starts=sac.learning_starts,
                batch_size=sac.batch_size,
                tau=sac.tau,
                gamma=sac.gamma,
                train_freq=sac.train_freq,
                gradient_steps=sac.gradient_steps,
                ent_coef=sac.ent_coef,
                target_update_interval=sac.target_update_interval,
                use_sde=sac.use_sde,
                policy_kwargs=sac.policy_kwargs,
                tensorboard_log=tb_path,
                device=self.device,
                verbose=1,
                seed=self.cfg.seed,
            )

        n_params = sum(p.numel() for p in model.policy.parameters())
        logger.info(f"Policy parameters: {n_params:,}")
        return model

    def _build_callbacks(self, curriculum: CurriculumManager):
        curriculum_cb = CurriculumCallback(
            manager=curriculum,
            check_freq=self.cfg.curriculum.eval_freq,
            verbose=1,
        )
        csv_cb = CSVLoggingCallback(
            csv_dir=os.path.join(self.cfg.csv_dir, self.cfg.exp_name),
            flush_freq=2_000,
            env_config=self.env_cfg,
            verbose=0,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=self.cfg.save_freq,
            save_dir=os.path.join(self.cfg.model_dir, self.cfg.exp_name),
            verbose=1,
        )
        return [curriculum_cb, csv_cb, ckpt_cb]

    def _log_stage_finished(
        self,
        curriculum: CurriculumManager,
        stage: int,
        planned_steps: int,
    ) -> None:
        """打印简洁的阶段结束总结，供用户手动查看。"""
        stats = curriculum.stage_stats[stage]
        sr_window = self.cfg.curriculum.window_size

        logger.info("-" * 60)
        logger.info(f"Stage {stage} {STAGE_SHORT_LABELS[stage]} finished")
        logger.info(
            f"Steps: {curriculum.stage_steps():,}/{planned_steps:,} | "
            f"Episodes: {stats.episodes:,}"
        )
        logger.info(
            f"SuccessRate{sr_window}: {curriculum.rolling_success_rate():.1%} | "
            f"AvgR{sr_window}: {curriculum.rolling_avg_reward():+.1f} | "
            f"AvgLen{sr_window}: {curriculum.rolling_avg_length():.0f}"
        )
        logger.info("-" * 60)

    def _confirm_next_stage(self, next_stage: int) -> bool:
        """询问用户是否继续；非交互运行默认选择 N。"""
        if not sys.stdin.isatty():
            logger.info(
                "Non-interactive input detected; defaulting to N and saving the current model."
            )
            return False

        prompt = f"Continue to Stage {next_stage} {STAGE_SHORT_LABELS[next_stage]}? [Y/N]: "
        while True:
            try:
                answer = input(prompt).strip().lower()
            except EOFError:
                logger.info("Input stream closed; defaulting to N.")
                return False

            if answer in ("y", "yes"):
                return True
            if answer in ("", "n", "no"):
                return False
            logger.info("Please enter Y or N.")

    def _save_final_model(self, model: SAC) -> None:
        """将当前策略保存到约定的 final model 路径。"""
        final_dir = os.path.join(self.cfg.model_dir, self.cfg.exp_name)
        os.makedirs(final_dir, exist_ok=True)
        final_path = os.path.join(final_dir, "model_final")
        model.save(final_path)
        logger.info(f"Final model saved → {final_path}.zip")
