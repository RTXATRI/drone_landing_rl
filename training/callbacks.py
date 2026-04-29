# 功能：定义手动课程指标、CSV 日志导出和模型 checkpoint 的 SB3 回调。
"""
自定义 Stable-Baselines3 callbacks。

训练器会组合三个 callback：

  CurriculumCallback     — 跟踪 episode 结果，并把手动课程阶段指标写入 TensorBoard。

  CSVLoggingCallback     — 将每个 episode 和每步的指标导出到 CSV，
                           供 MATLAB/Python 后处理使用。

  CheckpointCallback     — 按固定间隔保存模型 checkpoint。
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List

from stable_baselines3.common.callbacks import BaseCallback

from curriculum.curriculum_manager import CurriculumManager, STAGE_SHORT_LABELS
from curriculum.strategies import create_strategy, registered_episode_metric_keys
from configs.env_config import EnvConfig

logger = logging.getLogger(__name__)


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
    打印状态行，同时不破坏 tqdm/rich 进度条。

    learn() 运行时终端由 SB3 的进度条接管。直接 logger.info() 会写到 stdout，
    可能和进度条挤在同一行；tqdm.write() 会打印到进度条上方，并让进度条自行重绘。
    """
    try:
        from tqdm import tqdm
        tqdm.write(message)
    except Exception:
        print(message, flush=True)
    _log_file_only(logger, message)


def _episode_value(info: dict, episode_info: dict, key: str, default):
    """从 VecMonitor 输出读取自定义 episode 指标，并回退到顶层 info。"""
    if episode_info is not None and key in episode_info:
        return episode_info.get(key, default)
    return info.get(key, default)


def _episode_stage(info: dict, episode_info: dict) -> int:
    """返回 episode 或逐步奖励行对应的阶段。"""
    if episode_info is not None:
        if "episode_stage" in episode_info:
            return int(episode_info.get("episode_stage", 0))
        if "stage" in episode_info:
            return int(episode_info.get("stage", 0))
    return int(info.get("episode_stage", info.get("stage", 0)))


# ─────────────────────────────────────────────────────────────────────────────
# 1. 课程学习回调
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """
    记录手动课程学习进度。

    每一步：
      - 记录 VecEnv 报告的 episode 结果
      - 每 `check_freq` 步：将阶段指标写入 TensorBoard/控制台
    """

    def __init__(self, manager: CurriculumManager, check_freq: int = 8_000,
                 verbose: int = 1):
        super().__init__(verbose)
        self.manager = manager
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        step_delta = int(getattr(self.training_env, "num_envs", 1))
        self.manager.record_step(step_delta)

        # 从所有并行环境收集 episode 结果
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
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
        if self.n_calls % self.check_freq == 0:
            self._log_to_tensorboard()

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
        self.logger.dump(self.num_timesteps)

        if self.verbose >= 1:
            sr_window = self.manager.cfg.window_size
            stage1_score = ""
            if "stage1_train_score" in metric_keys:
                stage1_score = (
                    f" | TrainScore{sr_window}="
                    f"{self.manager.rolling_metric('stage1_train_score'):.1f}"
                )
            _write_progress_line(
                f"[{self.num_timesteps:>10,}] "
                f"Stage {st} {STAGE_SHORT_LABELS[st]} | "
                f"Eps={eps} | "
                f"SuccessRate{sr_window}={sr:.1%} | "
                f"AvgR{sr_window}={self.manager.rolling_avg_reward():+.1f} | "
                f"AvgLen{sr_window}={self.manager.rolling_avg_length():.0f}"
                f"{stage1_score}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 2. CSV 日志回调
# ─────────────────────────────────────────────────────────────────────────────

class CSVLoggingCallback(BaseCallback):
    """
    将训练指标导出为 CSV 文件，供 MATLAB / Python 后处理。

    在 `csv_dir/` 中创建的文件：
      episode_log.csv     — 每个 episode 一行：stage、reward、success
      reward_log_stageN.csv — 每步一行：各阶段自己的奖励分项和调试指标
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
                 env_config: EnvConfig = None, verbose: int = 0):
        super().__init__(verbose)
        self.csv_dir    = csv_dir
        self.flush_freq = flush_freq
        self.env_config = env_config or EnvConfig()

        os.makedirs(csv_dir, exist_ok=True)
        self._episode_fh = self._train_fh = None
        self._episode_w  = self._train_w  = None
        self._reward_fhs: Dict[int, object] = {}
        self._reward_ws: Dict[int, csv.DictWriter] = {}
        self._reward_cols: Dict[int, tuple] = {}
        self._reward_strategies: Dict[int, object] = {}
        self._episode_cols = [
            *self.BASE_EPISODE_COLS,
            *registered_episode_metric_keys(),
        ]
        self._has_written_header = False

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
            stage = _episode_stage(info, ep)

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
            if "reward/total" in info:
                strategy = self._strategy_for_stage(stage)
                writer = self._open_reward_writer(stage)
                row = strategy.reward_log_row(info)
                row["timestep"] = ts
                row["stage"] = stage
                writer.writerow(row)

        # 训练指标行（周期性 flush）
        if self.n_calls % self.flush_freq == 0 and self._train_w:
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

        return True

    def _on_training_end(self) -> None:
        for fh in (self._episode_fh, self._train_fh, *self._reward_fhs.values()):
            if fh:
                fh.close()
        logger.info(f"CSV logs saved to: {self.csv_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Checkpoint 回调
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointCallback(BaseCallback):
    """按固定步数间隔保存模型和 replay buffer。"""

    def __init__(self, save_freq: int, save_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_dir  = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            ts   = self.num_timesteps
            path = os.path.join(self.save_dir, f"ckpt_{ts:010d}")
            self.model.save(path)
            if self.verbose >= 1:
                _write_progress_line(f"Checkpoint saved -> {path}.zip")
        return True
