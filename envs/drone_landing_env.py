# 功能：实现基于 PyBullet 的无人机降落环境后端，用于 Python 预训练仿真。
"""
无人机降落环境的 PyBullet 实现。

通过实现抽象后端方法来扩展 BaseDroneLandingEnv。无人机使用运动学控制
（mass=0 的刚体），因此 PyBullet 主要用于：
  - 渲染/可视化（GUI 模式）
  - RGB 数组生成（rgb_array 模式）
  - 几何接触检测（用解析方式实现，不依赖 PyBullet contact；
    两个 mass=0 物体之间的 contact 不够可靠）

如果要替换为 ROS/Gazebo：
  创建 `envs/gazebo_env.py` → 继承 BaseDroneLandingEnv → 只重新实现下面这些方法。
  其它逻辑（obs、reward、curriculum）保持不变。
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pybullet
import pybullet_data

from configs.env_config import EnvConfig
from curriculum.strategies.base_strategy import CurriculumStrategy
from envs.base_env import BaseDroneLandingEnv
from envs.dynamics.kinematic_model import KinematicDroneModel
from envs.landing_platform.moving_platform import MovingPlatform


class DroneLandingEnv(BaseDroneLandingEnv):
    """
    基于 PyBullet 后端的无人机降落环境。

    每个实例都会创建一个 PyBullet server。使用 SubprocVecEnv 时，
    每个子进程会自动拥有独立 server，彼此不冲突。
    """

    def __init__(
        self,
        env_config: EnvConfig,
        strategy: CurriculumStrategy,
        render_mode: Optional[str] = None,
    ):
        # 调用 super().__init__ 前先保存（后者可能调用 _init_simulation）
        self._render_mode = render_mode
        self._client: int = -1

        # PyBullet 物体 ID（在 _init_simulation 中设置）
        self._drone_id:    int = -1
        self._platform_id: int = -1
        self._plane_id:    int = -1

        # 缓存的状态字典（每步更新）
        self._drone_state:    Dict[str, np.ndarray] = {}
        self._platform_state: Dict[str, np.ndarray] = {}

        # 运动学模型
        self._drone_model = KinematicDroneModel(
            dt=env_config.episode.dt,
            filter_alpha=env_config.drone.cmd_filter_alpha,
            disturbance_config=env_config.disturbance,
        )
        self._platform = MovingPlatform(env_config.platform, env_config.episode.dt)

        super().__init__(env_config, strategy)
        self._init_simulation()

    # =========================================================================
    # 抽象方法实现
    # =========================================================================

    def _init_simulation(self) -> None:
        """启动 PyBullet server 并创建场景物体。"""
        if self._render_mode == "human":
            self._client = pybullet.connect(pybullet.GUI)
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0,
                                               physicsClientId=self._client)
            pybullet.resetDebugVisualizerCamera(
                cameraDistance=14, cameraYaw=45, cameraPitch=-35,
                cameraTargetPosition=[0, 0, 1],
                physicsClientId=self._client,
            )
        else:
            self._client = pybullet.connect(pybullet.DIRECT)

        p = self._client
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=p)
        pybullet.setGravity(0, 0, 0, physicsClientId=p)  # 运动学控制，不使用重力
        pybullet.setTimeStep(self.config.episode.dt, physicsClientId=p)

        # 地面平面（视觉参考）
        self._plane_id = pybullet.loadURDF("plane.urdf", physicsClientId=p)

        # 无人机机体（mass=0 → 静态运动学刚体，不参与动力学）
        hw = self.config.drone.half_extents
        drone_col = pybullet.createCollisionShape(
            pybullet.GEOM_BOX, halfExtents=list(hw), physicsClientId=p
        )
        drone_vis = pybullet.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=list(hw),
            rgbaColor=[0.20, 0.45, 0.85, 0.95], physicsClientId=p
        )
        self._drone_id = pybullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=drone_col,
            baseVisualShapeIndex=drone_vis,
            basePosition=[5.0, 4.0, 3.0],
            physicsClientId=p,
        )

        # 降落平台
        ph = self.config.platform.half_extents
        plat_col = pybullet.createCollisionShape(
            pybullet.GEOM_BOX, halfExtents=list(ph), physicsClientId=p
        )
        plat_vis = pybullet.createVisualShape(
            pybullet.GEOM_BOX, halfExtents=list(ph),
            rgbaColor=[0.85, 0.30, 0.20, 0.95], physicsClientId=p
        )
        self._platform_id = pybullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=plat_col,
            baseVisualShapeIndex=plat_vis,
            basePosition=[0.0, 0.0, 0.0],
            physicsClientId=p,
        )

    def _reset_simulation(self, rng: np.random.Generator) -> None:
        """随机化无人机起始位置，并重置所有模型。"""
        cfg = self.config.episode

        # 让当前课程策略配置场景级状态。
        self.strategy.setup_scene(self, rng)

        # 围绕平台的圆柱随机出生点
        radius = rng.uniform(cfg.init_spawn_radius_min, cfg.init_spawn_radius_max)
        angle  = rng.uniform(0, 2 * np.pi)
        height = rng.uniform(cfg.init_spawn_height_min, cfg.init_spawn_height_max)
        drone_pos = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            height,
        ])

        # 随机初始 yaw
        init_euler = np.array([0.0, 0.0, rng.uniform(-np.pi, np.pi)])

        # 重置模型
        self._drone_model.reset(drone_pos, init_euler, rng=rng)
        self._platform.reset(rng=rng)

        # 在 _sync_pybullet 前填充状态缓存
        self._drone_state    = self._drone_model.get_state()
        self._drone_state.update(self._drone_model.get_disturbance_info())
        self._platform_state = self._platform.get_state()

        self._sync_bodies()

    def _apply_action(self, vel_cmd: np.ndarray) -> None:
        self._drone_model.apply_command(vel_cmd)

    def _step_simulation(self) -> None:
        # 推进运动学模型
        self._drone_model.step()
        self._platform.step()

        # 更新缓存
        self._drone_state    = self._drone_model.get_state()
        self._drone_state.update(self._drone_model.get_disturbance_info())
        self._platform_state = self._platform.get_state()

        # 同步 PyBullet 可视化位置
        self._sync_bodies()

        # 推进物理仿真（渲染和调试绘制需要）
        pybullet.stepSimulation(physicsClientId=self._client)

    def _get_drone_state(self) -> Dict[str, np.ndarray]:
        return self._drone_state

    def _get_platform_state(self) -> Dict[str, np.ndarray]:
        return self._platform_state

    def _check_contact(self) -> bool:
        """
        几何 AABB 接触检测。

        这比两个 mass=0 运动学刚体之间的 PyBullet contacts 更可靠。
        """
        dp = self._drone_state["position"]
        pp = self._platform_state["position"]
        ph = self.config.platform.half_extents
        dh = self.config.drone.half_extents

        # 穿透测试使用的合并半长
        tol_xy = ph[0] + dh[0]
        tol_z  = ph[2] + dh[2]

        return (
            abs(dp[0] - pp[0]) < tol_xy and
            abs(dp[1] - pp[1]) < tol_xy and
            abs(dp[2] - pp[2]) < tol_z
        )

    def render(self) -> Optional[np.ndarray]:
        if self._render_mode == "rgb_array":
            W, H = 640, 480
            view = pybullet.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=14, yaw=45, pitch=-35, roll=0, upAxisIndex=2,
                physicsClientId=self._client,
            )
            proj = pybullet.computeProjectionMatrixFOV(
                fov=55, aspect=W / H, nearVal=0.1, farVal=200.0,
                physicsClientId=self._client,
            )
            _, _, rgba, _, _ = pybullet.getCameraImage(
                W, H, view, proj, physicsClientId=self._client
            )
            return np.array(rgba, dtype=np.uint8)[:, :, :3]
        return None

    def close(self) -> None:
        if self._client >= 0:
            try:
                pybullet.disconnect(physicsClientId=self._client)
            except Exception:
                pass
            self._client = -1

    # =========================================================================
    # 私有辅助函数
    # =========================================================================

    def _sync_bodies(self) -> None:
        """更新 PyBullet 可视化位置，使其匹配运动学状态。"""
        if self._client < 0:
            return
        p = self._client

        # 无人机
        dp = self._drone_state.get("position", np.array([5.0, 4.0, 3.0]))
        de = self._drone_state.get("euler",    np.zeros(3))
        dq = pybullet.getQuaternionFromEuler(de.tolist())
        pybullet.resetBasePositionAndOrientation(
            self._drone_id, dp.tolist(), dq, physicsClientId=p
        )

        # 平台
        pp = self._platform_state.get("position", np.zeros(3))
        pe = self._platform_state.get("euler", np.zeros(3))
        pq = pybullet.getQuaternionFromEuler(pe.tolist())
        pybullet.resetBasePositionAndOrientation(
            self._platform_id, pp.tolist(), pq, physicsClientId=p
        )
