# 功能：平台静止运动策略（课程一使用）。
from typing import Tuple

import numpy as np

from configs.env_config import PlatformConfig
from .base import MotionStrategy


class StaticMotion(MotionStrategy):
    """平台保持静止不动。"""

    @property
    def name(self) -> str:
        return "static"

    def reset(self, rng: np.random.Generator, cfg: PlatformConfig) -> None:
        pass

    def current_state(self, cfg: PlatformConfig) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)

    def step(
        self, dt: float, t: float, cfg: PlatformConfig, speed: float | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
