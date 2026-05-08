# 功能：平台运动状态的 4D 卡尔曼滤波器 — 匀速模型。
import numpy as np


class PlatformKalmanFilter:
    """4 状态卡尔曼滤波器：[px, py, vx, vy] 匀速模型。

    仅观测位置（2 维测量），速度由匀速模型通过位置差/dt 推算。

    用法：
        kf = PlatformKalmanFilter(dt)
        kf.reset(init_pos, init_vel)       # 每 episode 重置
        x_est = kf.step(z)                  # 每 step predict + update
        # z = [px_meas, py_meas]           # 仅位置测量
        # x_est = [px_est, py_est, vx_est, vy_est]
    """

    STATE_DIM = 4
    MEAS_DIM = 2

    def __init__(self, dt: float):
        self._dt = float(dt)
        # 状态转移矩阵 F (匀速模型)
        self._F = np.eye(self.STATE_DIM, dtype=np.float64)
        self._F[0, 2] = self._dt
        self._F[1, 3] = self._dt

        # 测量矩阵 H (仅观测位置)
        self._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)

        # 过程噪声协方差 Q
        self._Q = np.diag([1e-4, 1e-4, 0.5, 0.5]).astype(np.float64)

        # 测量噪声协方差 R（仅位置噪声）
        self._R = np.diag([0.01, 0.01]).astype(np.float64)

        # 内部状态
        self._x = np.zeros(self.STATE_DIM, dtype=np.float64)
        self._P = np.eye(self.STATE_DIM, dtype=np.float64) * 0.1

    def reset(self, init_pos: np.ndarray, init_vel: np.ndarray) -> None:
        """用初始位置和速度初始化滤波器状态。"""
        self._x = np.array([
            float(init_pos[0]), float(init_pos[1]),
            float(init_vel[0]), float(init_vel[1]),
        ], dtype=np.float64)
        self._P = np.eye(self.STATE_DIM, dtype=np.float64) * 0.1

    def predict(self) -> None:
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

    def update(self, z: np.ndarray) -> np.ndarray:
        y = z[:self.MEAS_DIM] - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(self.STATE_DIM) - K @ self._H) @ self._P
        return self._x.copy()

    def step(self, z: np.ndarray) -> np.ndarray:
        """执行一次 predict + update，返回后验状态估计。
        z = [px_meas, py_meas] — 仅位置测量，速度由 KF 匀速模型推算。"""
        self.predict()
        return self.update(np.asarray(z, dtype=np.float64))
