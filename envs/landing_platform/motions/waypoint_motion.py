# 功能：多边形航点运动策略 — 沿随机闭合多边形边界平滑运动。
from typing import List, Tuple

import numpy as np

from configs.env_config import PlatformConfig
from .base import MotionStrategy


def _generate_valid_waypoints(
    rng: np.random.Generator, n: int, min_edge: float, xy_range: float,
) -> List[np.ndarray]:
    """在 [-xy_range, xy_range]² 内生成 n 个点，确保每条边 ≥ min_edge。"""
    for _ in range(50):
        pts = [
            np.array([
                rng.uniform(-xy_range, xy_range),
                rng.uniform(-xy_range, xy_range),
            ], dtype=np.float64)
            for _ in range(n)
        ]
        valid = True
        for i in range(n):
            j = (i + 1) % n
            if float(np.linalg.norm(pts[j] - pts[i])) < min_edge:
                valid = False
                break
        if valid:
            return pts
    # 兜底：返回随机点（放宽最小边约束）
    return [
        np.array([
            rng.uniform(-xy_range, xy_range),
            rng.uniform(-xy_range, xy_range),
        ], dtype=np.float64)
        for _ in range(n)
    ]


class WaypointMotion(MotionStrategy):
    """沿闭合多边形边界平滑运动，模拟船舶操舵。

    speed 参数用于推进沿边距离并计算速度；None 时位置不变。
    接近航点时提前 blend 到下一段方向，通过一阶低通实现平滑转向。

    参数：
        boundary: 平台运动边界半边长 (m)，航点生成范围 = boundary。
    """

    TRANSITION_ZONE = 1.0    # 提前转向距离 (m)
    HEADING_SMOOTH = 0.15    # 航向一阶低通系数

    def __init__(self, boundary: float = 50.0):
        self._boundary = float(boundary)
        self._waypoints: List[np.ndarray] = []
        self._n = 0
        self._seg_index = 0
        self._seg_progress = 0.0
        self._heading = 0.0
        self._pos = np.zeros(2, dtype=np.float64)
        self._vel = np.zeros(2, dtype=np.float64)

    @property
    def name(self) -> str:
        return "waypoint"

    def reset(self, rng: np.random.Generator, cfg: PlatformConfig) -> None:
        n = int(rng.integers(3, 7))
        xy_range = self._boundary
        self._waypoints = _generate_valid_waypoints(rng, n, 10.0, xy_range)
        self._n = len(self._waypoints)
        self._seg_index = 0
        self._seg_progress = 0.0

        # 初始化航向为第一条边的方向
        p0 = self._waypoints[0]
        p1 = self._waypoints[1 % self._n]
        to_next = p1 - p0
        self._heading = float(np.arctan2(to_next[1], to_next[0]))
        self._pos = self._position_on_segment()
        self._vel = np.zeros(2, dtype=np.float64)

    def current_state(self, cfg: PlatformConfig) -> Tuple[np.ndarray, np.ndarray]:
        return self._pos.copy(), self._vel.copy()

    def step(
        self, dt: float, t: float, cfg: PlatformConfig, speed: float | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._n == 0:
            return np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)

        current, next_wp, seg_vec, seg_len = self._current_segment()

        # 计算目标航向
        current_heading = float(np.arctan2(seg_vec[1], seg_vec[0]))

        dist_to_end = seg_len - self._seg_progress
        if dist_to_end < self.TRANSITION_ZONE and seg_len > 1e-9:
            next_idx = (self._seg_index + 1) % self._n
            next_next = self._waypoints[(next_idx + 1) % self._n]
            next_vec = next_next - self._waypoints[next_idx]
            next_heading = float(np.arctan2(next_vec[1], next_vec[0]))

            # 处理角度 wrap
            diff = next_heading - current_heading
            diff = float((diff + np.pi) % (2.0 * np.pi) - np.pi)
            blend = float(np.clip(dist_to_end / self.TRANSITION_ZONE, 0.0, 1.0))
            heading_target = current_heading + (1.0 - blend) * diff
        else:
            heading_target = current_heading

        # 一阶低通平滑
        heading_diff = heading_target - self._heading
        heading_diff = float((heading_diff + np.pi) % (2.0 * np.pi) - np.pi)
        self._heading += self.HEADING_SMOOTH * heading_diff

        vel_dir = np.array([
            np.cos(self._heading), np.sin(self._heading)
        ], dtype=np.float64)

        # 推进位置
        prev = self._pos.copy()
        if speed is not None:
            self._advance(max(0.0, float(speed)) * float(dt))
            current, next_wp, seg_vec, seg_len = self._current_segment()

        self._pos = self._position_on_segment()
        self._vel = (self._pos - prev) / max(float(dt), 1e-9)

        return self.current_state(cfg)

    def _current_segment(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        current = self._waypoints[self._seg_index].copy()
        next_wp = self._waypoints[(self._seg_index + 1) % self._n]
        seg_vec = next_wp - current
        seg_len = float(np.linalg.norm(seg_vec))
        return current, next_wp, seg_vec, seg_len

    def _position_on_segment(self) -> np.ndarray:
        current, _, seg_vec, seg_len = self._current_segment()
        return (
            current + (seg_vec / max(seg_len, 1e-9)) * self._seg_progress
        ).astype(np.float64)

    def _advance(self, distance: float) -> None:
        remaining = float(distance)
        for _ in range(2000):
            if remaining <= 1e-9:
                break
            _, _, _, seg_len = self._current_segment()
            available = max(0.0, seg_len - self._seg_progress)
            inc = min(remaining, available)
            self._seg_progress += inc
            remaining -= inc
            if self._seg_progress >= seg_len - 1e-9 and seg_len > 1e-9:
                self._seg_progress = 0.0
                self._seg_index = (self._seg_index + 1) % self._n
