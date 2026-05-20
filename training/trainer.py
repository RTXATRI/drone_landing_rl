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
from typing import Optional, TYPE_CHECKING

import numpy as np

from configs.env_config import EnvConfig
from configs.train_config import TrainConfig
from curriculum.curriculum_manager import CurriculumManager, STAGE_SHORT_LABELS
from curriculum.strategies import episode_info_keywords
from utils.logger import setup_logger

if TYPE_CHECKING:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecMonitor
    from training.model_selection import BestModelCandidateCallback

logger = logging.getLogger(__name__)

EPISODE_INFO_KEYWORDS = episode_info_keywords()


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
        mix_ratio: float = 0.0,
    ):
        self.env_cfg      = env_config
        self.cfg          = train_config
        self.resume_path  = resume_path
        self.start_stage  = start_stage
        self.mix_ratio    = float(mix_ratio)
        self._best_model_cb: Optional["BestModelCandidateCallback"] = None

        # 日志设置
        run_dir = os.path.join(train_config.log_dir, train_config.exp_name)
        setup_logger(run_dir)

        # 设备解析
        import torch

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

        # 3. 构建课程管理器和 best-model 候选采样器
        curriculum = CurriculumManager(self.cfg.curriculum)
        curriculum.set_stage(self.start_stage)
        self._build_best_model_callback()
        callbacks = self._build_callbacks(curriculum)

        # 4. 按阶段训练；阶段切换由用户手动决定。
        reset_num_timesteps = self.resume_path is None
        try:
            for stage in selected_stages:
                if vec_env is None:
                    vec_env = self._build_vec_env(self.cfg.n_envs, stage)
                    model.set_env(vec_env)

                curriculum.set_stage(stage)
                stage_obs = self._activate_stage(vec_env, stage)
                model._last_obs = stage_obs
                model._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)

                stage_steps = self._stage_budget(stage)
                stage_start_step = int(model.num_timesteps)
                if self._best_model_cb is not None:
                    self._best_model_cb.start_stage(
                        stage=stage,
                        start_step=stage_start_step,
                        stage_budget=stage_steps,
                    )
                logger.info("-" * 60)
                logger.info(
                    f"Stage {stage} {STAGE_SHORT_LABELS[stage]} started | "
                    f"budget={stage_steps:,} timesteps"
                )
                logger.info("-" * 60)

                self._apply_stage_lr_schedule(
                    model,
                    stage=stage,
                    stage_start_step=stage_start_step,
                    stage_budget=stage_steps,
                )

                model.learn(
                    total_timesteps = stage_steps,
                    callback = callbacks,
                    log_interval = self.cfg.sb3_log_interval_episodes,
                    tb_log_name = self.cfg.exp_name,
                    reset_num_timesteps = reset_num_timesteps,
                    progress_bar = True,
                )
                reset_num_timesteps = False

                if self._best_model_cb is not None:
                    self._best_model_cb.save_stage_final(stage, model)
                self._log_stage_finished(curriculum, stage, stage_steps)
                if self._best_model_cb is not None and vec_env is not None:
                    vec_env.close()
                    vec_env = None
                self._evaluate_best_model_candidates(stage)
                if stage < self.cfg.curriculum.max_stage:
                    if not self._confirm_next_stage(stage + 1):
                        logger.info("Manual curriculum stopped by user.")
                        break
        except KeyboardInterrupt:
            logger.info("Training interrupted by user — saving current model…")
        finally:
            self._save_final_model(model)
            curriculum.log_summary()
            if vec_env is not None:
                vec_env.close()

    # ── 私有构建函数 ─────────────────────────────────────────────────────────

    def _build_vec_env(self, n_envs: int, stage: int) -> "VecMonitor":
        from stable_baselines3.common.vec_env import VecMonitor
        from training.env_factory import make_training_env_fn
        from training.spawn_vec_env import SpawnSafeSubprocVecEnv

        logger.info(f"Spawning {n_envs} environment processes (stage {stage})…")
        fns = [
            make_training_env_fn(self.env_cfg, rank=i, seed=self.cfg.seed, stage=stage)
            for i in range(n_envs)
        ]
        # spawn 可以避免 PyBullet 共享 C 库在 fork 下的相关问题
        vec_env = SpawnSafeSubprocVecEnv(fns, start_method="spawn")
        vec_env = VecMonitor(vec_env, info_keywords=EPISODE_INFO_KEYWORDS)
        logger.info("Environments ready.")
        return vec_env

    def _activate_stage(self, vec_env: "VecMonitor", stage: int) -> np.ndarray:
        """广播手动选择的阶段，并开始一批干净的新 episode。"""
        from curriculum.strategies import create_strategy

        logger.info(f"Activating Stage {stage} {STAGE_SHORT_LABELS[stage]} on all envs.")
        if self.mix_ratio > 0.0 and stage > 1:
            # 预缓存前置课程策略（修复 Bug 3：直接从当前课程启动时前置不存在）
            vec_env.env_method("set_strategy", create_strategy(stage - 1, self.env_cfg))
        vec_env.env_method("set_strategy", create_strategy(stage, self.env_cfg))
        if self.mix_ratio > 0.0 and stage > 1:
            vec_env.env_method("enable_mix_training", self.mix_ratio)
        return vec_env.reset()

    def _stage_budget(self, stage: int) -> int:
        """返回从 1 开始编号的课程阶段对应的配置步数预算。"""
        return int(self.cfg.curriculum.stage_timesteps[stage - 1])

    def _make_lr_schedule(self, *, stage_start_step: int, stage_budget: int):
        """构造按当前课程阶段局部进度计算的学习率调度函数。"""
        import math

        sac = self.cfg.sac
        base_lr = float(sac.learning_rate)
        decay_start = float(sac.lr_decay_start)
        decay_end = float(sac.lr_decay_end)
        min_ratio = float(sac.lr_decay_min_ratio)
        stage_start_step = int(stage_start_step)
        stage_budget = max(1, int(stage_budget))
        stage_end_step = stage_start_step + stage_budget

        def lr_schedule(progress_remaining: float) -> float:
            current_global = (1.0 - float(progress_remaining)) * float(stage_end_step)
            stage_progress = (
                (current_global - float(stage_start_step)) / float(stage_budget)
            )
            stage_progress = float(np.clip(stage_progress, 0.0, 1.0))

            if stage_progress <= decay_start:
                return base_lr
            if stage_progress >= decay_end:
                return base_lr * min_ratio
            t = (stage_progress - decay_start) / (decay_end - decay_start)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * t))
            return base_lr * (min_ratio + (1.0 - min_ratio) * cosine_factor)

        return lr_schedule

    def _apply_stage_lr_schedule(
        self,
        model: "SAC",
        *,
        stage: int,
        stage_start_step: int,
        stage_budget: int,
    ) -> None:
        """将当前阶段局部学习率调度应用到 SB3 模型和 optimizer。"""
        sac = self.cfg.sac

        if sac.lr_decay:
            lr_schedule = self._make_lr_schedule(
                stage_start_step=stage_start_step,
                stage_budget=stage_budget,
            )
            model.learning_rate = lr_schedule
            model._setup_lr_schedule()

            logger.info(
                f"Applied stage-local LR schedule: stage={stage}, "
                f"start={stage_start_step:,}, budget={stage_budget:,}"
            )
        else:
            model.learning_rate = sac.learning_rate
            model._setup_lr_schedule()

    def _build_model(self, vec_env: "VecMonitor") -> "SAC":
        from stable_baselines3 import SAC

        sac = self.cfg.sac
        tb_path = os.path.join(self.cfg.log_dir, self.cfg.exp_name)

        if sac.lr_decay:
            logger.info(
                f"LR decay enabled: constant until {sac.lr_decay_start:.0%} done, "
                f"cosine to {sac.lr_decay_min_ratio:.0%} at "
                f"{sac.lr_decay_end:.0%} done (stage-local)"
            )

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

    def _build_best_model_callback(self) -> None:
        if self.cfg.best_model_selection_enabled:
            from training.model_selection import BestModelCandidateCallback

            self._best_model_cb = BestModelCandidateCallback(
                save_root=os.path.join(self.cfg.model_dir, self.cfg.exp_name),
                grid_count=self.cfg.best_model_grid_count,
                interval_top_m=self.cfg.best_model_interval_top_m,
                verbose=1,
            )
        else:
            self._best_model_cb = None

    def _build_callbacks(self, curriculum: CurriculumManager):
        from training.callbacks import (
            CheckpointCallback,
            CSVLoggingCallback,
            CurriculumCallback,
        )

        n_envs = max(1, int(self.cfg.n_envs))
        curriculum_cb = CurriculumCallback(
            manager=curriculum,
            check_freq=self.cfg.curriculum.eval_freq * n_envs,
            status_interval_episodes=self.cfg.curriculum_status_interval_episodes,
            verbose=1,
        )
        csv_cb = CSVLoggingCallback(
            csv_dir=os.path.join(self.cfg.csv_dir, self.cfg.exp_name),
            flush_freq=2_000 * n_envs,
            env_config=self.env_cfg,
            reward_step_csv_enabled=self.cfg.reward_step_csv_enabled,
            verbose=0,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=self.cfg.save_freq,
            save_dir=os.path.join(self.cfg.model_dir, self.cfg.exp_name),
            verbose=1,
        )
        callbacks = [curriculum_cb, csv_cb, ckpt_cb]
        if self._best_model_cb is not None:
            callbacks.append(self._best_model_cb)
        return callbacks

    def _evaluate_best_model_candidates(self, stage: int) -> None:
        """阶段正常结束后，对该阶段候选模型进行统一复评。"""
        if self._best_model_cb is None:
            return
        from training.model_selection import evaluate_stage_candidates

        candidates = self._best_model_cb.candidates_for_stage(stage)
        try:
            evaluate_stage_candidates(
                candidates=candidates,
                stage=stage,
                env_config=self.env_cfg,
                train_config=self.cfg,
                device=self.device,
            )
        except Exception:
            logger.exception("Best-model evaluation failed for stage %s.", stage)

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
        summary_fields = [
            f"SuccessRate{sr_window}: {curriculum.rolling_success_rate():.1%}",
        ]
        if "train_score" in curriculum.current_episode_metric_keys():
            summary_fields.append(
                f"AvgTrainScore{sr_window}: "
                f"{curriculum.rolling_metric('train_score'):.1f}"
            )
        summary_fields.extend([
            f"AvgLen{sr_window}: {curriculum.rolling_avg_length():.0f}",
            f"AvgReward{sr_window}: {curriculum.rolling_avg_reward():+.1f}",
        ])
        logger.info(" | ".join(summary_fields))
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

    def _save_final_model(self, model: "SAC") -> None:
        """将当前策略保存到约定的 final model 路径。"""
        final_dir = os.path.join(self.cfg.model_dir, self.cfg.exp_name)
        os.makedirs(final_dir, exist_ok=True)
        final_path = os.path.join(final_dir, "model_final")
        model.save(final_path)
        logger.info(f"Final model saved → {final_path}.zip")
