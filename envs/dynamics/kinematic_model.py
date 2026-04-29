# 功能：实现无人机速度指令低通滤波、轻量扰动和运动学积分模型。
"""
无人机运动学模型。

对速度指令实现一阶低通滤波，避免 RL agent 利用瞬时速度跳变
（相当于模拟真实 PX4 速度控制器的带宽限制）。它还可以加入小幅水平风
漂移/阵风和速度跟踪误差，使轻量预训练仿真中的悬停不再是完美静态问题。

这里不模拟气动或电机动力学；这些由真实飞控处理。
"""

from typing import Dict, Optional

import numpy as np

from configs.env_config import DisturbanceConfig


class KinematicDroneModel:
    """
    将速度指令积分为无人机位置/姿态。

    即使神经网络连续两步输出差异很大，滤波器也能保证速度不会不连续跳变。
    这是动作空间平滑的主要机制。

    Filter: v_filtered[t] = α * v_filtered[t-1] + (1-α) * v_cmd[t]
      α=0 → 不滤波（直接跟踪）
      α=0.7 → 中等平滑（约 3-5 step 滞后）
    """

    def __init__(
        self,
        dt: float,
        filter_alpha: float = 0.7,
        disturbance_config: Optional[DisturbanceConfig] = None,
    ):
        """
        参数：
            dt:           控制 timestep（秒）。
            filter_alpha: 低通系数，范围 [0, 1)。
            disturbance_config: 水平风和速度跟踪设置。
        """
        self.dt = dt
        self.alpha = float(np.clip(filter_alpha, 0.0, 0.99))
        self.disturbance_cfg = disturbance_config or DisturbanceConfig()
        self._rng = np.random.default_rng()

        # 状态
        self.position: np.ndarray  = np.zeros(3, dtype=np.float64)
        self.velocity: np.ndarray  = np.zeros(3, dtype=np.float64)  # 实际速度
        self.euler: np.ndarray     = np.zeros(3, dtype=np.float64)  # roll, pitch, yaw
        self.yaw_rate: float       = 0.0

        # 滤波后的指令（内部状态）
        self._filt_vel: np.ndarray = np.zeros(3, dtype=np.float64)
        self._filt_yr: float       = 0.0
        self._tracked_vel: np.ndarray = np.zeros(3, dtype=np.float64)
        self._tracking_scale: np.ndarray = np.ones(3, dtype=np.float64)

        # 扰动状态（世界系；当前版本仅水平扰动）
        self._base_wind_vel: np.ndarray = np.zeros(3, dtype=np.float64)
        self._gust_vel: np.ndarray = np.zeros(3, dtype=np.float64)
        self._wind_vel: np.ndarray = np.zeros(3, dtype=np.float64)
        self._gust_peak_vel: np.ndarray = np.zeros(3, dtype=np.float64)
        self._gust_elapsed: float = 0.0
        self._gust_duration: float = 0.0
        self._gust_shape: str = "smooth"

    def reset(
        self,
        position: np.ndarray,
        euler: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.position   = np.array(position, dtype=np.float64)
        self.velocity   = np.zeros(3, dtype=np.float64)
        self.euler      = np.array(euler, dtype=np.float64)
        self.yaw_rate   = 0.0
        self._filt_vel  = np.zeros(3, dtype=np.float64)
        self._filt_yr   = 0.0
        self._tracked_vel = np.zeros(3, dtype=np.float64)
        self._tracking_scale = np.ones(3, dtype=np.float64)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._reset_disturbance()

    def apply_command(self, vel_cmd: np.ndarray) -> None:
        """
        通过低通滤波缓存速度指令。
        应在 step() 前调用。

        参数：
            vel_cmd: 世界系下的 [vx, vy, vz, omega_yaw]，使用真实单位。
        """
        α = self.alpha
        self._filt_vel = α * self._filt_vel + (1.0 - α) * vel_cmd[:3]
        self._filt_yr  = α * self._filt_yr  + (1.0 - α) * float(vel_cmd[3])

    def step(self) -> None:
        """积分加入跟踪误差和扰动后的实际速度。"""
        self._tracking_scale = self._sample_tracking_scale()
        self._tracked_vel = self._filt_vel * self._tracking_scale
        self._wind_vel = self._update_wind_velocity()

        self.velocity  = self._tracked_vel + self._wind_vel
        self.position += self.velocity * self.dt
        self.yaw_rate  = self._filt_yr
        self.euler[2] += self._filt_yr * self.dt
        # 将 yaw 包装到 [-π, π]
        self.euler[2]  = (self.euler[2] + np.pi) % (2.0 * np.pi) - np.pi
        # 假设 roll 和 pitch 被良好控制，近似为 0
        self.euler[0] = 0.0
        self.euler[1] = 0.0

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "position": self.position.astype(np.float32),
            "velocity": self.velocity.astype(np.float32),
            "euler":    self.euler.astype(np.float32),
            "yaw_rate": np.float32(self.yaw_rate),
        }

    def get_disturbance_info(self) -> Dict[str, np.ndarray]:
        """返回世界系下的当前扰动/调试值。"""
        return {
            "wind_velocity": self._wind_vel.astype(np.float32),
            "base_wind_velocity": self._base_wind_vel.astype(np.float32),
            "gust_velocity": self._gust_vel.astype(np.float32),
            "tracking_scale": self._tracking_scale.astype(np.float32),
            "tracked_velocity": self._tracked_vel.astype(np.float32),
        }

    # ── 扰动辅助函数 ────────────────────────────────────────────────────────

    def _reset_disturbance(self) -> None:
        cfg = self.disturbance_cfg
        self._base_wind_vel = np.zeros(3, dtype=np.float64)
        self._gust_vel = np.zeros(3, dtype=np.float64)
        self._wind_vel = np.zeros(3, dtype=np.float64)
        self._gust_peak_vel = np.zeros(3, dtype=np.float64)
        self._gust_elapsed = 0.0
        self._gust_duration = 0.0
        self._gust_shape = "smooth"

        if not cfg.enabled:
            return

        self._base_wind_vel = self._sample_horizontal_vector(
            cfg.base_wind_speed_min,
            cfg.base_wind_speed_max,
        )
        self._wind_vel = self._base_wind_vel.copy()

    def _sample_tracking_scale(self) -> np.ndarray:
        cfg = self.disturbance_cfg
        if not cfg.enabled or not cfg.tracking_error_enabled:
            return np.ones(3, dtype=np.float64)
        return self._rng.uniform(
            cfg.tracking_scale_min,
            cfg.tracking_scale_max,
            size=3,
        ).astype(np.float64)

    def _sample_horizontal_vector(self, speed_min: float, speed_max: float) -> np.ndarray:
        lo = max(0.0, min(float(speed_min), float(speed_max)))
        hi = max(lo, float(speed_max))
        speed = float(self._rng.uniform(lo, hi))
        angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
        return np.array([
            speed * np.cos(angle),
            speed * np.sin(angle),
            0.0,
        ], dtype=np.float64)

    def _update_wind_velocity(self) -> np.ndarray:
        cfg = self.disturbance_cfg
        if not cfg.enabled:
            self._gust_vel = np.zeros(3, dtype=np.float64)
            return np.zeros(3, dtype=np.float64)

        if self._gust_is_active():
            self._gust_vel = self._current_gust_velocity()
            self._gust_elapsed += self.dt
            if self._gust_elapsed >= self._gust_duration:
                self._clear_gust()
        else:
            self._gust_vel = np.zeros(3, dtype=np.float64)
            self._maybe_start_gust()

        return self._base_wind_vel + self._gust_vel

    def _gust_is_active(self) -> bool:
        return self._gust_duration > 0.0 and self._gust_elapsed < self._gust_duration

    def _maybe_start_gust(self) -> None:
        cfg = self.disturbance_cfg
        if cfg.gust_prob_per_second <= 0.0:
            return
        probability = 1.0 - np.exp(-float(cfg.gust_prob_per_second) * self.dt)
        if float(self._rng.random()) >= probability:
            return

        self._gust_peak_vel = self._sample_horizontal_vector(
            cfg.gust_speed_min,
            cfg.gust_speed_max,
        )
        self._gust_duration = float(
            self._rng.uniform(cfg.gust_duration_min, cfg.gust_duration_max)
        )
        self._gust_elapsed = 0.0
        self._gust_shape = (
            "smooth" if float(self._rng.random()) < cfg.gust_smooth_prob else "impulse"
        )
        self._gust_vel = self._current_gust_velocity()

    def _current_gust_velocity(self) -> np.ndarray:
        if self._gust_duration <= 0.0:
            return np.zeros(3, dtype=np.float64)
        phase = float(np.clip(self._gust_elapsed / self._gust_duration, 0.0, 1.0))
        if self._gust_shape == "smooth":
            envelope = 0.5 - 0.5 * np.cos(2.0 * np.pi * phase)
        else:
            attack = 0.15
            if phase < attack:
                envelope = phase / attack
            else:
                envelope = np.exp(-4.0 * (phase - attack) / max(1.0 - attack, 1e-6))
        return self._gust_peak_vel * float(envelope)

    def _clear_gust(self) -> None:
        self._gust_vel = np.zeros(3, dtype=np.float64)
        self._gust_peak_vel = np.zeros(3, dtype=np.float64)
        self._gust_elapsed = 0.0
        self._gust_duration = 0.0
        self._gust_shape = "smooth"
