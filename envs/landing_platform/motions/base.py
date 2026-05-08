# 功能：运动策略抽象基类。
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from configs.env_config import PlatformConfig


class MotionStrategy(ABC):
    """平台运动策略抽象接口。

    每个具体子类实现一种运动模式（静止/李萨如/巡逻/航点）。

    step() 的 speed 参数：
      - None：策略保持当前位置不推进
      - float：策略使用此标量速度推进位置并计算速度

    step() 返回 (pos_xy, vel_xy)：
      - pos_xy: 平台当前 XY 位置 (m)
      - vel_xy: 平台当前 XY 速度 (m/s)，由策略自行计算完整速度
      MovingPlatform 直接使用返回值，不再做速度缩放。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """唯一标识，用于日志和 CSV 记录。"""

    @abstractmethod
    def reset(self, rng: np.random.Generator, cfg: PlatformConfig) -> None:
        """每个 episode 开始时随机化内部参数。"""

    def current_state(self, cfg: PlatformConfig) -> Tuple[np.ndarray, np.ndarray]:
        """返回当前 (pos_xy, vel_xy)，不推进内部状态。"""
        return np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)

    @abstractmethod
    def step(
        self,
        dt: float,
        t: float,
        cfg: PlatformConfig,
        speed: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (pos_xy: (2,), vel_xy: (2,))。
        vel_xy 为平台在 XY 方向的实际速度 (m/s)，由策略全权计算。"""
