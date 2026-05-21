# 功能：定义手动课程指标、CSV 日志导出和模型 checkpoint 的 SB3 回调。
"""
自定义 Stable-Baselines3 callbacks。

训练器会组合多个 callback：

  StageProgressCallback — 用自管 Rich 风格 tqdm 显示当前课程阶段训练进度。

  CurriculumCallback     — 跟踪 episode 结果，并把手动课程阶段指标写入 TensorBoard。

  CSVLoggingCallback     — 将每个 episode 和每步的指标导出到 CSV，
                           供 MATLAB/Python 后处理使用。

  CheckpointCallback     — 按固定间隔保存模型 checkpoint。
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import warnings
from typing import Dict, List

from stable_baselines3.common.callbacks import BaseCallback

from curriculum.curriculum_manager import CurriculumManager, STAGE_SHORT_LABELS
from curriculum.strategies import create_strategy
from configs.env_config import EnvConfig

logger = logging.getLogger(__name__)


def _progress_tqdm():
    """Return a Rich-styled tqdm when available, otherwise plain tqdm."""
    try:
        from tqdm import TqdmExperimentalWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", TqdmExperimentalWarning)
            from tqdm.rich import tqdm
        return tqdm
    except Exception:
        from tqdm import tqdm
        return tqdm


def _log_file_only(record_logger: logging.Logger, message: str, level: int = logging.INFO) -> None:
    """把适合进度条显示的消息写入文件日志，同时避免重复输出到控制台。"""
    root = logging.getLogger()
    record = record_logger.makeRecord(
        record_logger.name,
        level,
        __file__,
        0,
        message,
        args=(),
        exc_info=None,
    )
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and level >= handler.level:
            handler.handle(record)


def _write_progress_line(message: str) -> None:
    """
    打印状态行，同时不破坏 tqdm 进度条。

    训练和复评进度条统一写到 stdout。直接 logger.info() 可能和进度条
    挤在同一行；tqdm.write() 会打印到进度条上方，并让进度条自行重绘。
    """
    try:
        tqdm = _progress_tqdm()
        tqdm.write(message, file=sys.stdout)
    except Exception:
        print(message, flush=True)
    _log_file_only(logger, message)


def _episode_value(info: dict, episode_info: dict, key: str, default):
    """从 VecMonitor 输出读取自定义 episode 指标，并回退到顶层 info。"""
    if episode_info is not None and key in episode_info:
        return episode_info.get(key, default)
    return info.get(key, default)


def _info_stage(info: dict, episode_info: dict) -> int:
    """返回 episode 或逐步奖励行对应的阶段。"""
    if episode_info is not None and "stage" in episode_info:
        return int(episode_info.get("stage", 0))
    return int(info.get("stage", 0))


def _next_interval_step(current_step: int, interval: int) -> int:
    """Return the next positive interval boundary strictly after current_step."""
    interval = max(1, int(interval))
    current_step = max(0, int(current_step))
    return ((current_step // interval) + 1) * interval


def _format_step_compact(step: int) -> str:
    """Format large timesteps compactly for status lines."""
    step = int(step)
    if abs(step) >= 1_000_000:
        return f"{step / 1_000_000:.2f}M"
    if abs(step) >= 1_000:
        return f"{step / 1_000:.1f}k"
    return str(step)


# ─────────────────────────────────────────────────────────────────────────────
# 0. 训练阶段进度条
# ─────────────────────────────────────────────────────────────────────────────

class StageProgressCallback(BaseCallback):
    """使用自管 Rich 风格进度条显示当前课程阶段训练进度。"""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self._stage = 1
        self._label = "Stage"
        self._stage_start_step = 0
        self._stage_budget = 1
        self._displayed_steps = 0
        self._bar = None
        self._interactive = False

    def start_stage(
        self,
        *,
        stage: int,
        label: str,
        start_step: int,
        stage_budget: int,
    ) -> None:
        self._stage = int(stage)
        self._label = str(label)
        self._stage_start_step = int(start_step)
        self._stage_budget = max(1, int(stage_budget))
        self._displayed_steps = 0

    def _progress_value(self) -> int:
        elapsed = int(self.num_timesteps) - int(self._stage_start_step)
        return max(0, min(int(self._stage_budget), elapsed))

    def _on_training_start(self) -> None:
        self._displayed_steps = self._progress_value()
        self._interactive = bool(sys.stdout.isatty())
        if not self._interactive:
            if self.verbose >= 1:
                logger.info(
                    "Stage %s %s progress: 0/%s timesteps",
                    self._stage,
                    self._label,
                    f"{self._stage_budget:,}",
                )
            return

        try:
            tqdm = _progress_tqdm()

            self._bar = tqdm(
                total=self._stage_budget,
                initial=self._displayed_steps,
                desc=f"Stage {self._stage} {self._label}",
                unit="steps",
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
                mininterval=0.5,
            )
        except Exception:
            self._bar = None
            self._interactive = False
            if self.verbose >= 1:
                logger.info(
                    "Stage %s %s progress: 0/%s timesteps",
                    self._stage,
                    self._label,
                    f"{self._stage_budget:,}",
                )

    def _on_step(self) -> bool:
        current = self._progress_value()
        delta = current - self._displayed_steps
        if delta > 0:
            self._displayed_steps = current
            if self._bar is not None:
                self._bar.update(delta)
        return True

    def _on_training_end(self) -> None:
        current = self._progress_value()
        delta = current - self._displayed_steps
        if delta > 0:
            self._displayed_steps = current
            if self._bar is not None:
                self._bar.update(delta)

        if self._bar is not None:
            self._bar.refresh()
            self._bar.close()
            self._bar = None
        elif self.verbose >= 1:
            logger.info(
                "Stage %s %s progress: %s/%s timesteps",
                self._stage,
                self._label,
                f"{self._displayed_steps:,}",
                f"{self._stage_budget:,}",
            )


# ─────────────────────────────────────────────────────────────────────────────
# 1. 课程学习回调
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """
    记录手动课程学习进度。

    每一步：
      - 记录 VecEnv 报告的 episode 结果
      - 每 `check_freq` 步：将阶段指标写入 TensorBoard
      - 每 `status_interval_episodes` 个当前阶段 episode：输出课程状态短行
    """

    def __init__(self, manager: CurriculumManager, check_freq: int = 1_000,
                 status_interval_episodes: int = 200,
                 verbose: int = 1):
        super().__init__(verbose)
        self.manager = manager
        self.check_freq = max(1, int(check_freq))
        self.status_interval_episodes = max(1, int(status_interval_episodes))
        self._mixed_count = 0
        self._total_count = 0
        self._next_check_step = self.check_freq
        self._last_status_stage = None
        self._next_status_eps = self.status_interval_episodes

    def _on_training_start(self) -> None:
        self._next_check_step = _next_interval_step(self.num_timesteps, self.check_freq)
        self._last_status_stage = None
        self._next_status_eps = self.status_interval_episodes

    def _on_step(self) -> bool:
        step_delta = int(getattr(self.training_env, "num_envs", 1))
        self.manager.record_step(step_delta)

        # 从所有并行环境收集 episode 结果
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                ep_stage = _info_stage(info, ep)
                self._total_count += 1
                # 穿插训练的混合回合不记入滚动窗口（修复 Bug 4）
                if ep_stage != self.manager.current_stage:
                    self._mixed_count += 1
                    continue
                metrics = {
                    key: float(_episode_value(info, ep, key, 0.0))
                    for key in self.manager.current_episode_metric_keys()
                }
                self.manager.record_episode(
                    success=bool(_episode_value(info, ep, "success", False)),
                    reward=float(ep.get("r", 0.0)),
                    length=int(ep.get("l", 0)),
                    metrics=metrics,
                )

        # 周期性记录手动课程指标
        if self.num_timesteps >= self._next_check_step:
            self._log_to_tensorboard()
            while self._next_check_step <= self.num_timesteps:
                self._next_check_step += self.check_freq

        self._maybe_print_status()

        return True

    def _log_to_tensorboard(self) -> None:
        sr = self.manager.rolling_success_rate()
        st = self.manager.current_stage
        eps = self.manager.stage_stats[st].episodes
        self.logger.record("curriculum/stage",             float(st))
        self.logger.record("curriculum/rolling_success_rate", sr)
        self.logger.record("curriculum/rolling_avg_reward",
                           self.manager.rolling_avg_reward())
        self.logger.record("curriculum/rolling_avg_length",
                           self.manager.rolling_avg_length())
        self.logger.record("curriculum/stage_episodes",
                           float(eps))
        metric_keys = self.manager.current_episode_metric_keys()
        for key in metric_keys:
            self.logger.record(
                f"curriculum/rolling_{key}",
                self.manager.rolling_metric(key),
            )
        if self._total_count > 0:
            self.logger.record(
                "curriculum/actual_mix_ratio",
                float(self._mixed_count / self._total_count),
            )
        self._mixed_count = 0
        self._total_count = 0

    def _maybe_print_status(self) -> None:
        if self.verbose < 1:
            return

        st = self.manager.current_stage
        eps = self.manager.stage_stats[st].episodes
        if self._last_status_stage != st:
            self._last_status_stage = st
            self._next_status_eps = (
                (eps // self.status_interval_episodes) + 1
            ) * self.status_interval_episodes

        if eps < self._next_status_eps:
            return

        sr_window = self.manager.cfg.window_size
        metric_keys = self.manager.current_episode_metric_keys()
        summary_fields = [f"SR{sr_window}={self.manager.rolling_success_rate():.1%}"]
        if "train_score" in metric_keys:
            summary_fields.append(
                f"score{sr_window}="
                f"{self.manager.rolling_metric('train_score'):.1f}"
            )
        summary_fields.extend([
            f"len{sr_window}={self.manager.rolling_avg_length():.0f}",
            f"rew{sr_window}={self.manager.rolling_avg_reward():+.1f}",
        ])
        _write_progress_line(
            f"[{_format_step_compact(self.num_timesteps)}] "
            f"S{st} {STAGE_SHORT_LABELS[st]} | "
            f"eps={eps} | "
            + " | ".join(summary_fields)
        )
        while self._next_status_eps <= eps:
            self._next_status_eps += self.status_interval_episodes


# ─────────────────────────────────────────────────────────────────────────────
# 2. CSV 日志回调
# ─────────────────────────────────────────────────────────────────────────────

class CSVLoggingCallback(BaseCallback):
    """
    将训练指标导出为 CSV 文件，供 MATLAB / Python 后处理。

    在 `csv_dir/` 中创建的文件：
      episode_log.csv     — 每个 episode 一行：stage、reward、success
      reward_log_stageN.csv — 可选调试日志，每步一行：各阶段自己的奖励分项和调试指标
      training_log.csv    — 每 `flush_freq` 步一行：loss、ent_coef、lr、fps
    """

    # 列定义（用于 CSV 表头）
    BASE_EPISODE_COLS = [
        "timestep", "stage", "reward", "length", "success",
    ]
    TRAIN_COLS = [
        "timestep", "stage", "fps",
        "actor_loss", "critic_loss", "ent_coef", "learning_rate",
    ]

    def __init__(self, csv_dir: str, flush_freq: int = 2_000,
                 env_config: EnvConfig = None,
                 reward_step_csv_enabled: bool = False,
                 verbose: int = 0):
        super().__init__(verbose)
        self.csv_dir    = csv_dir
        self.flush_freq = max(1, int(flush_freq))
        self.env_config = env_config or EnvConfig()
        self.reward_step_csv_enabled = bool(reward_step_csv_enabled)

        os.makedirs(csv_dir, exist_ok=True)
        self._episode_fh = self._train_fh = None
        self._episode_w  = self._train_w  = None
        self._reward_fhs: Dict[int, object] = {}
        self._reward_ws: Dict[int, csv.DictWriter] = {}
        self._reward_cols: Dict[int, tuple] = {}
        self._reward_strategies: Dict[int, object] = {}
        self._episode_cols = [*self.BASE_EPISODE_COLS, "train_score"]
        self._has_written_header = False
        self._next_flush_step = self.flush_freq

    def _on_training_start(self) -> None:
        """打开 CSV 文件并写入表头。"""
        def _open(name: str, cols: List[str]):
            mode = "a" if self._has_written_header else "w"
            fh = open(os.path.join(self.csv_dir, name), mode, newline="", encoding="utf-8")
            w  = csv.DictWriter(fh, fieldnames=cols)
            if not self._has_written_header:
                w.writeheader()
            return fh, w

        self._episode_fh, self._episode_w = _open("episode_log.csv",  self._episode_cols)
        self._train_fh,   self._train_w   = _open("training_log.csv", self.TRAIN_COLS)
        self._has_written_header = True
        self._next_flush_step = _next_interval_step(self.num_timesteps, self.flush_freq)
        if self.reward_step_csv_enabled:
            logger.info("Reward step CSV enabled; reward_log_stageN.csv may become very large.")
        else:
            logger.info("Reward step CSV disabled; only episode/training CSV will be written.")

    def _strategy_for_stage(self, stage: int):
        stage = int(stage)
        if stage not in self._reward_strategies:
            self._reward_strategies[stage] = create_strategy(stage, self.env_config)
        return self._reward_strategies[stage]

    def _open_reward_writer(self, stage: int) -> csv.DictWriter:
        stage = int(stage)
        if stage in self._reward_ws:
            return self._reward_ws[stage]

        strategy = self._strategy_for_stage(stage)
        cols = tuple(strategy.reward_log_columns())
        path = os.path.join(self.csv_dir, f"reward_log_stage{stage}.csv")
        exists = os.path.exists(path)
        nonempty = exists and os.path.getsize(path) > 0

        if nonempty:
            with open(path, "r", newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                existing_cols = tuple(next(reader, []))
            if existing_cols != cols:
                raise RuntimeError(
                    f"Reward CSV header mismatch for stage {stage}: {path}. "
                    "Use a new --exp_name or remove the old reward CSV."
                )

        fh = open(path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if not nonempty:
            writer.writeheader()

        self._reward_fhs[stage] = fh
        self._reward_ws[stage] = writer
        self._reward_cols[stage] = cols
        return writer

    def _on_step(self) -> bool:
        ts = self.num_timesteps

        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            stage = _info_stage(info, ep)

            # 回合行
            if ep is not None and self._episode_w:
                episode_row = {
                    "timestep": ts,
                    "stage":    stage,
                    "reward":   round(ep.get("r", 0.0), 4),
                    "length":   ep.get("l", 0),
                    "success":  int(bool(_episode_value(info, ep, "success", False))),
                }
                for key in self._episode_cols:
                    if key not in episode_row:
                        episode_row[key] = _episode_value(info, ep, key, 0.0)
                self._episode_w.writerow(episode_row)
                self._episode_fh.flush()

            # 奖励分项行（每个包含奖励 info 的 step）
            if self.reward_step_csv_enabled and "reward/total" in info:
                strategy = self._strategy_for_stage(stage)
                writer = self._open_reward_writer(stage)
                row = strategy.reward_log_row(info)
                row["timestep"] = ts
                row["stage"] = stage
                writer.writerow(row)

        # 训练指标行（周期性 flush）
        if self.num_timesteps >= self._next_flush_step and self._train_w:
            lv = self.model.logger.name_to_value
            stage = 1
            try:
                stage = self.training_env.env_method("get_curriculum_stage")[0]
            except Exception:
                pass
            self._train_w.writerow({
                "timestep":     ts,
                "stage":        stage,
                "fps":          round(lv.get("time/fps",            0.0), 1),
                "actor_loss":   round(lv.get("train/actor_loss",    0.0), 6),
                "critic_loss":  round(lv.get("train/critic_loss",   0.0), 6),
                "ent_coef":     round(lv.get("train/ent_coef",      0.0), 6),
                "learning_rate":round(lv.get("train/learning_rate", 0.0), 8),
            })
            self._train_fh.flush()
            for fh in self._reward_fhs.values():
                fh.flush()
            while self._next_flush_step <= self.num_timesteps:
                self._next_flush_step += self.flush_freq

        return True

    def _on_training_end(self) -> None:
        for fh in (self._episode_fh, self._train_fh, *self._reward_fhs.values()):
            if fh:
                fh.close()
        self._episode_fh = self._train_fh = None
        self._episode_w = self._train_w = None
        self._reward_fhs.clear()
        self._reward_ws.clear()
        self._reward_cols.clear()
        logger.info(f"CSV logs saved to: {self.csv_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Checkpoint 回调
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointCallback(BaseCallback):
    """按固定步数间隔保存模型和 replay buffer。"""

    def __init__(self, save_freq: int, save_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = max(1, int(save_freq))
        self.save_dir  = save_dir
        self._next_save_step = self.save_freq
        os.makedirs(save_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        self._next_save_step = _next_interval_step(self.num_timesteps, self.save_freq)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_save_step:
            ts   = self.num_timesteps
            path = os.path.join(self.save_dir, f"ckpt_{ts:010d}")
            self.model.save(path)
            if self.verbose >= 1:
                _write_progress_line(f"Checkpoint saved -> {path}.zip")
            while self._next_save_step <= self.num_timesteps:
                self._next_save_step += self.save_freq
        return True
