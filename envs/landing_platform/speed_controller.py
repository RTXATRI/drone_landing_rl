# 功能：通用速度控制器 — 多正弦叠加 + 一阶低通，生成无突变速度序列。
import numpy as np


class SpeedController:
    """独立于运动模式的速度控制器。

    通过 2~3 个低频正弦叠加产生目标速度，再经一阶低通滤波，
    确保输出速度连续无突变，模拟真实船舶的加减速特性。

    用法：
        ctrl = SpeedController()
        ctrl.reset(rng)           # 每 episode 调用
        speed = ctrl.step(dt)     # 每 step 调用，返回标量速度 (m/s)
    """

    SMOOTH_BETA = 0.06    # 一阶低通系数

    def __init__(self):
        self._t = 0.0
        self._v_min = 0.3
        self._v_max = 1.2
        self._mid = 0.75
        self._amp = 0.45
        self._freqs: list[float] = []
        self._phases: list[float] = []
        self._speed = 0.75

    def reset(self, rng: np.random.Generator) -> None:
        self._t = 0.0
        self._v_min = float(rng.uniform(0.3, 1.0))
        self._v_max = float(rng.uniform(1.2, 4.5))
        self._mid = (self._v_min + self._v_max) / 2.0
        self._amp = (self._v_max - self._v_min) / 2.0

        n_modes = int(rng.integers(2, 4))
        self._freqs = [float(rng.uniform(0.03, 0.10)) for _ in range(n_modes)]
        self._phases = [float(rng.uniform(0.0, 2.0 * np.pi)) for _ in range(n_modes)]
        self._speed = self._mid

    def step(self, dt: float) -> float:
        self._t += dt
        raw = self._mid + self._amp * (
            sum(
                np.sin(2.0 * np.pi * f * self._t + p)
                for f, p in zip(self._freqs, self._phases)
            )
            / float(len(self._freqs))
        )
        raw = float(np.clip(raw, self._v_min, self._v_max))
        self._speed += self.SMOOTH_BETA * (raw - self._speed)
        return float(self._speed)
