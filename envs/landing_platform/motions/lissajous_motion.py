# 功能：速度受控的闭合李萨如运动策略。
from typing import Tuple

import numpy as np

from configs.env_config import PlatformConfig
from .base import MotionStrategy

_FREQ_POOL = [1.0, 2.0, 3.0]
_TWO_PI = 2.0 * np.pi


class LissajousMotion(MotionStrategy):
    """闭合李萨如曲线。

    Stage 2 使用统一 SpeedController 输出的标量速度推进曲线弧长，
    因此平台速度上限由 SpeedController 控制，而不是由解析时间频率决定。
    """

    @property
    def name(self) -> str:
        return "lissajous"

    def reset(self, rng: np.random.Generator, cfg: PlatformConfig) -> None:
        self._ax = float(rng.uniform(15.0, 35.0))
        self._ay = float(rng.uniform(10.0, 30.0))
        self._fx = float(rng.choice(_FREQ_POOL))
        self._fy = float(rng.choice(_FREQ_POOL))
        self._phase = float(rng.uniform(0.0, _TWO_PI))
        self._u = float(rng.uniform(0.0, _TWO_PI))
        self._pos = self._curve(self._u)
        self._vel = np.zeros(2, dtype=np.float64)

    def current_state(self, cfg: PlatformConfig) -> Tuple[np.ndarray, np.ndarray]:
        return self._pos.copy(), self._vel.copy()

    def step(
        self, dt: float, t: float, cfg: PlatformConfig, speed: float | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if speed is None or speed <= 0.0:
            self._vel = np.zeros(2, dtype=np.float64)
            return self.current_state(cfg)

        prev = self._pos.copy()
        remaining = max(0.0, float(speed) * float(dt))
        max_du = 0.002
        for _ in range(2000):
            if remaining <= 1e-9:
                break
            tangent = self._curve_derivative(self._u)
            tangent_norm = max(float(np.linalg.norm(tangent)), 1e-9)
            du = min(remaining / tangent_norm, max_du)
            old_pos = self._curve(self._u)
            new_u = (self._u + du) % _TWO_PI
            new_pos = self._curve(new_u)
            moved = float(np.linalg.norm(new_pos - old_pos))
            self._u = new_u
            if moved <= 1e-9 and du < max_du:
                break
            remaining -= min(remaining, moved)
        self._pos = self._curve(self._u)

        delta = self._pos - prev
        self._vel = delta / max(float(dt), 1e-9)
        return self.current_state(cfg)

    def _curve(self, u: float) -> np.ndarray:
        return np.array([
            self._ax * np.sin(self._fx * u + self._phase),
            self._ay * np.sin(self._fy * u),
        ], dtype=np.float64)

    def _curve_derivative(self, u: float) -> np.ndarray:
        return np.array([
            self._ax * self._fx * np.cos(self._fx * u + self._phase),
            self._ay * self._fy * np.cos(self._fy * u),
        ], dtype=np.float64)
