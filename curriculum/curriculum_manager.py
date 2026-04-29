# 功能：管理已注册课程的 episode 统计和手动阶段切换指标。
"""
课程学习管理器。

记录每个阶段的 episode 统计，用于手动课程控制。
它不持有仿真状态，是纯 Python 统计模块。

阶段推进有意交给 Trainer 中的用户决策；本模块只记录滚动指标，并在阶段变化时重置窗口。
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Optional

import numpy as np

from curriculum.strategies import (
    STAGE_LABELS,
    STAGE_SHORT_LABELS,
    get_strategy_class,
    registered_stage_ids,
)

logger = logging.getLogger(__name__)


class StageStats:
    """累积单个课程阶段的统计量。"""

    def __init__(self):
        self.episodes:                  int   = 0
        self.successes:                 int   = 0
        self.total_reward:              float = 0.0
        self.total_steps:               int   = 0
        self.metric_totals:             Dict[str, float] = {}

    def record(
        self,
        success: bool,
        reward: float,
        length: int,
        metrics: Optional[dict] = None,
    ) -> None:
        metrics = dict(metrics or {})

        self.episodes     += 1
        self.total_steps  += length
        self.total_reward += reward
        for key, value in metrics.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            self.metric_totals[key] = self.metric_totals.get(key, 0.0) + numeric_value
        if success:
            self.successes += 1

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.episodes if self.episodes else 0.0

    @property
    def avg_length(self) -> float:
        return self.total_steps / self.episodes if self.episodes else 0.0

    def avg_metric(self, key: str) -> float:
        return self.metric_totals.get(key, 0.0) / self.episodes if self.episodes else 0.0

    def to_dict(self) -> dict:
        return {
            "episodes":                 self.episodes,
            "successes":                self.successes,
            "success_rate":             self.success_rate,
            "avg_reward":               self.avg_reward,
            "avg_length":               self.avg_length,
            "avg_metrics": {
                key: self.avg_metric(key)
                for key in sorted(self.metric_totals)
            },
            "total_steps":              self.total_steps,
        }


class CurriculumManager:
    """
    记录已注册课程学习的指标。

    用法（在 callback 内）：
        mgr.record_step(step_delta)
        mgr.record_episode(success, reward, length)
        # Trainer 决定是否进入下一阶段。
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: CurriculumConfig
        """
        self.cfg           = cfg
        self.current_stage = 1
        self._total_steps  = 0
        self._stage_start  = 0

        # 用于控制台/TensorBoard 汇总的滚动窗口
        self._success_window: deque = deque(maxlen=cfg.window_size)
        self._reward_window:  deque = deque(maxlen=cfg.window_size)
        self._length_window:  deque = deque(maxlen=cfg.window_size)
        self._metric_windows: Dict[str, deque] = {}

        # 每阶段累积统计
        self.stage_stats: Dict[int, StageStats] = {
            i: StageStats()
            for i in registered_stage_ids()
        }

    # ── 公共 API ─────────────────────────────────────────────────────────────

    def record_step(self, step_delta: int = 1) -> None:
        """记录自上次 callback 以来环境推进的 timestep 数。"""
        self._total_steps += int(step_delta)

    def record_episode(
        self,
        success: bool,
        reward: float,
        length: int,
        metrics: Optional[dict] = None,
    ) -> None:
        """在 episode 结束时调用。"""
        metrics = dict(metrics or {})

        self._success_window.append(float(success))
        self._reward_window.append(float(reward))
        self._length_window.append(float(length))
        for key, value in metrics.items():
            self._metric_window(key).append(float(value))
        self.stage_stats.setdefault(self.current_stage, StageStats()).record(
            success,
            reward,
            length,
            metrics=metrics,
        )

    def current_episode_metric_keys(self) -> tuple:
        return tuple(get_strategy_class(self.current_stage).episode_metrics)

    def _metric_window(self, key: str) -> deque:
        if key not in self._metric_windows:
            self._metric_windows[key] = deque(maxlen=self.cfg.window_size)
        return self._metric_windows[key]

    def rolling_success_rate(self) -> float:
        if not self._success_window:
            return 0.0
        return float(np.mean(self._success_window))

    def rolling_avg_reward(self) -> float:
        if not self._reward_window:
            return 0.0
        return float(np.mean(self._reward_window))

    def rolling_avg_length(self) -> float:
        if not self._length_window:
            return 0.0
        return float(np.mean(self._length_window))

    def rolling_metric(self, key: str) -> float:
        values = self._metric_windows.get(key)
        if not values:
            return 0.0
        return float(np.mean(values))

    def stage_steps(self) -> int:
        return int(self._total_steps - self._stage_start)

    def set_stage(self, stage: int) -> None:
        """切换到用户选择的阶段，并重置滚动窗口。"""
        self.current_stage = int(stage)
        self._stage_start = self._total_steps
        self.clear_windows()

    def clear_windows(self) -> None:
        """在手动阶段边界清空滚动窗口。"""
        self._success_window.clear()
        self._reward_window.clear()
        self._length_window.clear()
        self._metric_windows.clear()

    def summary(self) -> dict:
        return {
            "current_stage":        self.current_stage,
            "total_steps":          self._total_steps,
            "stage_steps":          self.stage_steps(),
            "rolling_success_rate": self.rolling_success_rate(),
            "rolling_avg_reward":   self.rolling_avg_reward(),
            "rolling_avg_length":   self.rolling_avg_length(),
            "stage_stats": {
                i: s.to_dict()
                for i, s in sorted(self.stage_stats.items())
            },
        }

    def log_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("Curriculum Summary")
        logger.info(f"  Current stage : {self.current_stage} — {STAGE_LABELS[self.current_stage]}")
        logger.info(f"  Total steps   : {self._total_steps:,}")
        logger.info(f"  Stage steps   : {self.stage_steps():,}")
        logger.info(f"  Rolling SR    : {self.rolling_success_rate():.1%}")
        for stage, stats in sorted(self.stage_stats.items()):
            if stats.episodes > 0:
                logger.info(
                    f"  Stage {stage}: {stats.episodes:4d} eps | "
                    f"SR={stats.success_rate:.1%} | "
                    f"AvgR={stats.avg_reward:+.1f} | "
                    f"AvgLen={stats.avg_length:.0f}"
                )
        logger.info("=" * 60)
