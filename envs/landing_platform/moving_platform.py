# 功能：带可配置运动模式的降落平台。
"""
带可配置运动模式的降落平台。

运动逻辑已抽取到 motions/ 策略包中，通过组合模式注入。
速度控制和边界裁剪通过可选方法按需激活（仅课程二使用）。
"""

import numpy as np
from typing import Dict

from configs.env_config import PlatformConfig
from .motions.base import MotionStrategy
from .motions.static_motion import StaticMotion
from .speed_controller import SpeedController


class MovingPlatform:
    """降落平台。

    运动模式由外部通过 set_motion_strategy() 注入 MotionStrategy 实例。
    速度控制器和边界裁剪默认关闭，各课程按需激活。
    """

    def __init__(self, cfg: PlatformConfig, dt: float):
        self.cfg = cfg
        self.dt = dt

        self._origin = np.zeros(3, dtype=np.float64)
        self.position = self._origin.copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.euler = np.zeros(3, dtype=np.float64)
        self.angular_rate = np.zeros(3, dtype=np.float64)
        self.detection_quality = 1.0
        self._t = 0.0
        self._motion_strategy: MotionStrategy = StaticMotion()
        self._speed_ctrl: SpeedController | None = None
        self._boundary_xy: float | None = None

    # ── 组合式配置 ────────────────────────────────────────────────────────────

    def set_motion_strategy(self, strategy: MotionStrategy) -> None:
        """设置运动策略。各课程在 setup_scene() 中调用。"""
        if not isinstance(strategy, MotionStrategy):
            raise TypeError(
                f"Expected MotionStrategy instance, got {type(strategy).__name__}"
            )
        self._motion_strategy = strategy

    def enable_speed_controller(self) -> None:
        """激活速度控制器。仅课程二调用。"""
        if self._speed_ctrl is None:
            self._speed_ctrl = SpeedController()

    def set_boundary(self, xy_limit: float) -> None:
        """设置平台 XY 位置裁剪边界（半边长）。仅课程二调用。"""
        self._boundary_xy = float(xy_limit)

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    def reset(self, rng: np.random.Generator) -> None:
        self._t = 0.0
        self.position = self._origin.copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.euler = np.zeros(3, dtype=np.float64)
        self.angular_rate = np.zeros(3, dtype=np.float64)
        self.detection_quality = 1.0
        self._motion_strategy.reset(rng, self.cfg)
        if self._speed_ctrl is not None:
            self._speed_ctrl.reset(rng)
        pos_xy, vel_xy = self._motion_strategy.current_state(self.cfg)
        self.position[0] = float(pos_xy[0])
        self.position[1] = float(pos_xy[1])
        self.velocity[0] = float(vel_xy[0])
        self.velocity[1] = float(vel_xy[1])
        self._apply_boundary()

    def step(self) -> Dict[str, np.ndarray]:
        self._t += self.dt

        speed = self._speed_ctrl.step(self.dt) if self._speed_ctrl else None
        pos_xy, vel_xy = self._motion_strategy.step(
            self.dt, self._t, self.cfg, speed,
        )

        self.position[0] = pos_xy[0]
        self.position[1] = pos_xy[1]
        self.velocity[0] = vel_xy[0]
        self.velocity[1] = vel_xy[1]

        self._apply_boundary()

        return self.get_state()

    def _apply_boundary(self) -> None:
        if self._boundary_xy is not None:
            self.position[0] = float(np.clip(
                self.position[0], -self._boundary_xy, self._boundary_xy
            ))
            self.position[1] = float(np.clip(
                self.position[1], -self._boundary_xy, self._boundary_xy
            ))

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "position": self.position.astype(np.float32),
            "velocity": self.velocity.astype(np.float32),
            "euler": self.euler.astype(np.float32),
            "angular_rate": self.angular_rate.astype(np.float32),
            "detection_quality": np.float32(self.detection_quality),
        }
