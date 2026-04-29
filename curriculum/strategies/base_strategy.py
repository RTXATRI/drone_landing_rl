# 功能：定义课程策略抽象基类和内置策略可选辅助逻辑。
"""
课程策略抽象。

每个课程阶段独立负责目标点、场景设置、奖励、成功条件和日志字段。
通用 RewardCalculator 已退役；具体奖励必须放在具体策略中。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np

from configs.env_config import EnvConfig


class CurriculumStrategy(ABC):
    """单个课程阶段的基础接口。"""

    label: str = "Unknown"
    short_label: str = "Unknown"
    hover_stage: bool = False
    episode_metrics: Tuple[str, ...] = ()

    def __init__(self, env_config: EnvConfig):
        self.config = env_config
        self._hover_height = 2.0

    @abstractmethod
    def stage_id(self) -> int:
        """返回日志和 CLI 阶段选择中使用的稳定数字 id。"""

    @abstractmethod
    def setup_scene(self, env: Any, rng: np.random.Generator) -> None:
        """在后端重置模型前配置场景级阶段状态。"""

    @abstractmethod
    def get_target_pos(self, platform_pos: np.ndarray) -> np.ndarray:
        """返回世界坐标系下的阶段任务目标点。"""

    @abstractmethod
    def compute_reward(
        self,
        *,
        env: Any = None,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
        action: np.ndarray,
        prev_action: np.ndarray,
        target_pos: np.ndarray,
        hold_steps: int = 0,
        prev_hold_steps: int = 0,
    ) -> Tuple[float, dict]:
        """计算阶段特有的稠密奖励和日志项。"""

    @abstractmethod
    def check_success(
        self,
        *,
        env: Any,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
    ) -> Tuple[bool, dict]:
        """返回由阶段策略负责的成功/接触终止状态。"""

    def is_hover_stage(self) -> bool:
        return bool(self.hover_stage)

    def is_hold_stable(
        self,
        *,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
        target_pos: np.ndarray,
    ) -> bool:
        """返回当前 step 是否处于该策略定义的稳定保持区间。"""
        return False

    def terminal_bonus(
        self,
        *,
        env: Any,
        terminated: bool,
        truncated: bool,
        term_info: dict,
    ) -> Tuple[float, bool]:
        """返回 (终止奖励加成, episode_success)。"""
        return 0.0, False

    def episode_metric_keys(self) -> Tuple[str, ...]:
        return tuple(self.episode_metrics)

    def get_episode_metrics(self, env: Any) -> dict:
        """返回 episode 级策略指标。"""
        return {}

    def get_step_metrics(self, env: Any) -> dict:
        """返回用于 info 日志的逐步策略指标。"""
        return {}

    def reward_log_columns(self) -> Tuple[str, ...]:
        """返回该策略逐步 reward CSV 的列定义。"""
        return (
            "timestep", "stage", "reward_total",
            "metric_dist", "metric_horiz_err", "metric_vert_err", "metric_speed",
        )

    def reward_log_row(self, info: dict) -> dict:
        """把环境 info 映射为该策略逐步 reward CSV 的一行。"""
        return {
            "stage": int(info.get("stage", self.stage_id())),
            "reward_total": round(info.get("reward/total", 0.0), 5),
            "metric_dist": round(info.get("metric/dist", 0.0), 4),
            "metric_horiz_err": round(info.get("metric/horiz_err", 0.0), 4),
            "metric_vert_err": round(info.get("metric/vert_err", 0.0), 4),
            "metric_speed": round(info.get("metric/speed", 0.0), 4),
        }

    def get_eval_metrics(self, episode_info: dict) -> dict:
        """从 episode info 中提取策略特有的评估指标。"""
        return {
            key: episode_info.get(key, 0.0)
            for key in self.episode_metric_keys()
        }

    def set_hover_height(self, height: float) -> None:
        self._hover_height = float(height)

    def get_hover_height(self) -> float:
        return float(self._hover_height)


class HoverStrategyMixin:
    """悬停类策略的场景/目标辅助逻辑；奖励由具体阶段实现。"""

    hover_stage = True

    def _setup_hover_scene(self, env: Any, rng: np.random.Generator, motion: str) -> None:
        cfg = self.config.episode
        height = float(rng.uniform(cfg.hover_height_min, cfg.hover_height_max))
        self.set_hover_height(height)
        env._current_hover_height = height
        env._platform.set_motion(motion)

    def get_target_pos(self, platform_pos: np.ndarray) -> np.ndarray:
        return platform_pos + np.array([0.0, 0.0, self.get_hover_height()])

    def check_success(
        self,
        *,
        env: Any,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
    ) -> Tuple[bool, dict]:
        return False, {"termination": "none", "success": False}


class LandingStrategyMixin:
    """降落类策略的场景/目标/接触辅助逻辑；奖励由具体阶段实现。"""

    hover_stage = False

    def _setup_landing_scene(self, env: Any, rng: np.random.Generator, motion: str) -> None:
        env._platform.set_motion(motion)

    def get_target_pos(self, platform_pos: np.ndarray) -> np.ndarray:
        return platform_pos + np.array([0.0, 0.0, self.config.platform.half_extents[2]])

    def check_success(
        self,
        *,
        env: Any,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
    ) -> Tuple[bool, dict]:
        dp = drone_state["position"]
        dv = drone_state["velocity"]
        pp = platform_state["position"]
        cfg = self.config.platform

        horiz_dist = float(np.linalg.norm(dp[:2] - pp[:2]))
        vert_dist = abs(float(dp[2] - pp[2]))
        speed = float(np.linalg.norm(dv))

        in_horiz = horiz_dist < cfg.landing_radius
        in_vert = vert_dist < cfg.landing_height_tol
        low_speed = speed < cfg.landing_speed_max

        if in_horiz and in_vert and low_speed:
            return True, {"termination": "landed", "success": True}

        if env._check_contact():
            success = speed < cfg.landing_speed_max
            return True, {
                "termination": "landed" if success else "crashed",
                "success": success,
            }

        return False, {"termination": "none", "success": False}
