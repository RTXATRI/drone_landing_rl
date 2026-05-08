# 功能：往返巡逻运动策略 — 原点 ↔ 随机目标的往返巡逻。
from typing import Tuple

import numpy as np

from configs.env_config import PlatformConfig
from .base import MotionStrategy


class PatrolMotion(MotionStrategy):
    """原点 ↔ 随机目标点往返巡逻。每程随机选择直线或正弦波轨迹。

    speed 参数用于推进沿程距离；None 时退化为位置不变。
    目标点距离由 boundary 控制（限制在 boundary * 0.8 内）。
    """

    def __init__(self, boundary: float = 50.0):
        self._boundary = float(boundary)
        self._target = np.zeros(2, dtype=np.float64)
        self._going_to_target = True
        self._leg_mode = "straight"
        self._wave_amp = 0.5
        self._wave_freq = 2.0
        self._wave_phi = 0.0
        self._dist_traveled = 0.0
        self._seg_len = 1.0
        self._pos = np.zeros(2, dtype=np.float64)
        self._vel = np.zeros(2, dtype=np.float64)

    @property
    def name(self) -> str:
        return "patrol"

    def reset(self, rng: np.random.Generator, cfg: PlatformConfig) -> None:
        dist = float(rng.uniform(15.0, 40.0))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        self._target = np.array([
            dist * np.cos(angle),
            dist * np.sin(angle),
        ], dtype=np.float64)

        self._going_to_target = bool(rng.choice([True, False]))
        self._leg_mode = str(rng.choice(["straight", "wave"]))
        self._wave_amp = float(rng.uniform(0.3, 1.2))
        self._wave_freq = float(rng.uniform(1.5, 3.0))
        self._wave_phi = float(rng.uniform(0.0, 2.0 * np.pi))
        self._dist_traveled = 0.0
        self._seg_len = float(np.linalg.norm(self._target))
        self._pos = self._position_on_current_leg()
        self._vel = np.zeros(2, dtype=np.float64)

    def current_state(self, cfg: PlatformConfig) -> Tuple[np.ndarray, np.ndarray]:
        return self._pos.copy(), self._vel.copy()

    def step(
        self, dt: float, t: float, cfg: PlatformConfig, speed: float | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        prev = self._pos.copy()
        self._advance(max(0.0, float(speed or 0.0)) * float(dt))
        self._pos = self._position_on_current_leg()
        self._vel = (self._pos - prev) / max(float(dt), 1e-9)
        return self.current_state(cfg)

    def _current_leg(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        origin = np.zeros(2, dtype=np.float64)
        if self._going_to_target:
            start, end = origin.copy(), self._target.copy()
        else:
            start, end = self._target.copy(), origin.copy()
        seg_vec = end - start
        seg_len = max(float(np.linalg.norm(seg_vec)), 1e-9)
        forward = seg_vec / seg_len
        return start, seg_vec, forward, seg_len

    def _position_on_current_leg(self) -> np.ndarray:
        start, seg_vec, forward, seg_len = self._current_leg()
        perp = np.array([-forward[1], forward[0]], dtype=np.float64)
        t_progress = float(np.clip(
            self._dist_traveled / max(seg_len, 1e-9), 0.0, 1.0
        ))

        if self._leg_mode == "straight":
            return start + t_progress * seg_vec

        pos_on_line = start + t_progress * seg_vec
        envelope = np.sin(np.pi * t_progress)
        wave_phase = 2.0 * np.pi * self._wave_freq * t_progress + self._wave_phi
        offset = self._wave_amp * envelope * np.sin(wave_phase)
        return (pos_on_line + offset * perp).astype(np.float64)

    def _advance(self, distance: float) -> None:
        remaining = float(distance)
        for _ in range(2000):
            if remaining <= 1e-9:
                break
            _, _, _, seg_len = self._current_leg()
            old_pos = self._position_on_current_leg()
            old_dist = self._dist_traveled
            if self._leg_mode == "straight":
                inc = min(remaining, max(0.0, seg_len - old_dist))
                self._dist_traveled += inc
                moved = inc
            else:
                t_progress = old_dist / max(seg_len, 1e-9)
                dq = min(0.002, remaining / max(seg_len, 1e-9))
                self._dist_traveled = min(seg_len, old_dist + dq * seg_len)
                moved = float(np.linalg.norm(
                    self._position_on_current_leg() - old_pos
                ))
                if moved > remaining:
                    self._dist_traveled = old_dist
                    hi = dq
                    lo = 0.0
                    for _ in range(12):
                        mid = 0.5 * (lo + hi)
                        self._dist_traveled = old_dist + mid * seg_len
                        mid_moved = float(np.linalg.norm(
                            self._position_on_current_leg() - old_pos
                        ))
                        if mid_moved <= remaining:
                            lo = mid
                        else:
                            hi = mid
                    self._dist_traveled = old_dist + lo * seg_len
                    moved = float(np.linalg.norm(
                        self._position_on_current_leg() - old_pos
                    ))

            remaining -= min(remaining, max(moved, 0.0))
            if self._dist_traveled >= seg_len - 1e-9:
                self._dist_traveled = 0.0
                self._going_to_target = not self._going_to_target
