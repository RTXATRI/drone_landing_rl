# 功能：定义降落平台的静止、直线、正弦和 8 字形运动模式。
"""
带可配置运动模式的降落平台。

运动类型由当前活动课程策略从外部选择。

支持的运动模式：
  'static'     — 平台静止
  'linear'     — 仅沿 X 方向正弦运动
  'sinusoidal' — X/Y 独立正弦运动（每个 episode 随机相位）
  'figure8'    — XY 平面内的 8 字形轨迹
"""

import numpy as np
from typing import Dict, Tuple


class MovingPlatform:
    SUPPORTED_MOTIONS = {"static", "linear", "sinusoidal", "figure8"}

    def __init__(self, cfg, dt: float):
        """
        参数：
            cfg: PlatformConfig
            dt:  控制 timestep (s)
        """
        self.cfg = cfg
        self.dt  = dt

        self._origin  = np.zeros(3, dtype=np.float64)
        self.position = self._origin.copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.euler = np.zeros(3, dtype=np.float64)
        self.angular_rate = np.zeros(3, dtype=np.float64)
        self.detection_quality = 1.0
        self._t       = 0.0
        self._motion  = "static"
        self._phi_x   = 0.0
        self._phi_y   = 0.0

    def set_motion(self, motion_type: str) -> None:
        """选择下一次 reset/episode 使用的平台运动模式。"""
        motion_type = str(motion_type)
        if motion_type not in self.SUPPORTED_MOTIONS:
            known = ", ".join(sorted(self.SUPPORTED_MOTIONS))
            raise ValueError(f"Unknown platform motion '{motion_type}'. Known: {known}")
        self._motion = motion_type

    def reset(self, rng: np.random.Generator) -> None:
        """为新的 episode 重置平台。"""
        self._t       = 0.0
        self.position = self._origin.copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.euler = np.zeros(3, dtype=np.float64)
        self.angular_rate = np.zeros(3, dtype=np.float64)
        self.detection_quality = 1.0

        # 随机化初始相位，让训练看到更多样的轨迹
        self._phi_x = rng.uniform(0.0, 2.0 * np.pi)
        self._phi_y = rng.uniform(0.0, 2.0 * np.pi)

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """推进平台一个 timestep，返回 (position, velocity)。"""
        self._t += self.dt
        t    = self._t
        cfg  = self.cfg
        ax   = cfg.motion_amplitude_x
        ay   = cfg.motion_amplitude_y
        ω    = 2.0 * np.pi * cfg.motion_frequency
        φx   = self._phi_x
        φy   = self._phi_y

        if self._motion == "static":
            pass  # position/velocity stay zero

        elif self._motion == "linear":
            self.position[0] = ax * np.sin(ω * t + φx)
            self.position[1] = 0.0
            self.velocity[0] = ax * ω * np.cos(ω * t + φx)
            self.velocity[1] = 0.0

        elif self._motion == "sinusoidal":
            # X/Y 使用不同频率，避免纯周期重复模式
            self.position[0] = ax * np.sin(ω       * t + φx)
            self.position[1] = ay * np.sin(ω * 0.7 * t + φy)
            self.velocity[0] = ax * ω       * np.cos(ω       * t + φx)
            self.velocity[1] = ay * ω * 0.7 * np.cos(ω * 0.7 * t + φy)

        elif self._motion == "figure8":
            # 伯努利双纽线（归一化）
            s        = ω * t + φx
            denom    = 1.0 + np.sin(s) ** 2
            px       = ax * np.cos(s) / denom
            py       = ay * np.sin(s) * np.cos(s) / denom
            # 数值导数
            ds       = ω * self.dt
            s1       = s + ds
            d1       = 1.0 + np.sin(s1) ** 2
            self.velocity[0] = (ax * np.cos(s1) / d1 - px) / self.dt
            self.velocity[1] = (ay * np.sin(s1) * np.cos(s1) / d1 - py) / self.dt
            self.position[0] = px
            self.position[1] = py

        return self.get_state()

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "position": self.position.astype(np.float32),
            "velocity": self.velocity.astype(np.float32),
            "euler": self.euler.astype(np.float32),
            "angular_rate": self.angular_rate.astype(np.float32),
            "detection_quality": np.float32(self.detection_quality),
        }
