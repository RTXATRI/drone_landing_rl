# 功能：定义后端无关的无人机降落 Gymnasium 环境基类和观测/动作处理逻辑。
"""
无人机降落任务的抽象基类环境。

设计原则：
  这个类负责所有与具体仿真后端无关的逻辑：
    - 观测构造与归一化（机体系）
    - 动作缩放（机体系）和坐标变换
    - 奖励计算调度（具体奖励由活动课程策略决定）
    - Episode 终止逻辑
    - 课程策略管理
    - 策略指标跟踪

  子类只需要实现触碰物理后端的方法（当前是 PyBullet，未来可替换为 ROS/Gazebo）：
    _init_simulation(), _reset_simulation(), _apply_action(),
    _step_simulation(), _get_drone_state(), _get_platform_state(),
    _check_contact(), render(), close()

迁移指南（PyBullet → ROS/Gazebo）：
  1. 创建继承 BaseDroneLandingEnv 的 `envs/ros_gazebo_env.py`。
  2. 使用 ROS topic/service 实现这些抽象方法。
  3. base_env、configs、curriculum 和 training 无需修改。
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from configs.env_config import EnvConfig
from curriculum.strategies.base_strategy import CurriculumStrategy


class BaseDroneLandingEnv(gym.Env, ABC):
    """
    面向移动平台无人机降落任务的抽象 Gymnasium 环境。

    观测向量（34 维，yaw-FLU 机体系，已归一化并裁剪）：
      [0:3]   task_target_body   当前策略目标 - 无人机；x/y /20m，z /10m。
                                  具体目标由活动课程策略独立定义。
      [3:6]   rel_vel_body       平台 - 无人机速度；x/y /20m/s，z /5m/s
      [6]     rel_yaw            wrap(platform_yaw - drone_yaw) / π
      [7]     rel_yaw_rate       platform_yaw_rate - drone_yaw_rate
      [8]     range              ||target - drone|| /20m
      [9:12]  drone_vel_body     x/y /20m/s, z /5m/s
      [12:15] drone_euler        roll, pitch, yaw / π
      [15]    drone_yaw_rate
      [16]    local_height       AGL/海面/局部高度抽象 /30m。
                                  当前平台/地面基准为 z=0，因此 PyBullet 使用
                                  drone_z - platform_z。
      [17:20] platform_euler     roll, pitch, yaw / π
      [20:23] platform_rates     roll_rate, pitch_rate, yaw_rate
      [23:26] platform_vel_body  x/y /20m/s, z /5m/s
      [26]    detection_quality  [0, 1]
      [27:30] capabilities       v_xy/20, v_z_down/5, v_z_up/5
      [30:34] prev_action        上一时刻动作，范围 [-1, 1]

    动作向量（4 维连续，Box[-1, 1]，机体系）：
      [0] vx_body → 前向速度，按当前 episode 的 v_xy 限制映射
      [1] vy_body → 左向速度，按当前 episode 的 v_xy 限制映射
      [2] vz      → 下降/上升速度，按当前 episode 的垂直限制映射
      [3] yaw_rate → 按配置的 max_yaw_rate 映射
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    OBS_DIM = 34
    ACT_DIM = 4

    def __init__(self, env_config: EnvConfig, strategy: CurriculumStrategy):
        super().__init__()
        self.config = env_config
        if strategy is None:
            raise TypeError("BaseDroneLandingEnv requires a CurriculumStrategy instance.")
        strategy.config = env_config
        self.strategy = strategy
        self.curriculum_stage = int(strategy.stage_id())

        obs_low, obs_high = self._observation_bounds()
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high,
            shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.ACT_DIM,), dtype=np.float32
        )

        # 回合状态
        self._step_count: int = 0
        self._prev_action: np.ndarray = np.zeros(self.ACT_DIM, dtype=np.float32)
        self._episode_reward: float = 0.0
        self._episode_success: bool = False

        # 策略可选的连续保持步数，供具体奖励函数使用。
        self._hover_hold_steps: int = 0
        self._hover_total_hold_steps: int = 0
        self._current_hover_height: float = 2.0
        self._stage1_r_hold_total: float = 0.0
        self._stage1_hold_refund_steps: int = 0
        self._stage1_is_refunding_hold: bool = False
        self._success_mode: str = "train"

        # 每个 episode 随机化的控制能力
        dc = self.config.drone
        self._current_v_xy_max: float = max(dc.max_vx, dc.max_vy)
        self._current_vz_up_max: float = dc.max_vz_up
        self._current_vz_down_max: float = dc.max_vz_down

    def _observation_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回与分组观测裁剪相匹配的有限 Box 边界。"""
        oc = self.config.observation
        low = np.full(self.OBS_DIM, -oc.clip_signed, dtype=np.float32)
        high = np.full(self.OBS_DIM, oc.clip_signed, dtype=np.float32)

        low[6] = -oc.clip_angle
        high[6] = oc.clip_angle
        low[7] = -oc.clip_rate
        high[7] = oc.clip_rate
        low[8] = 0.0
        high[8] = oc.clip_nonnegative

        low[12:15] = -oc.clip_angle
        high[12:15] = oc.clip_angle
        low[15] = -oc.clip_rate
        high[15] = oc.clip_rate
        low[16] = 0.0
        high[16] = oc.clip_nonnegative

        low[17:20] = -oc.clip_angle
        high[17:20] = oc.clip_angle
        low[20:23] = -oc.clip_rate
        high[20:23] = oc.clip_rate

        low[26] = 0.0
        high[26] = 1.0
        low[27:30] = 0.0
        high[27:30] = oc.clip_capability
        low[30:34] = -1.0
        high[30:34] = 1.0
        return low, high

    # =========================================================================
    # 抽象后端接口：每个具体子类需要实现这些方法
    # =========================================================================

    @abstractmethod
    def _init_simulation(self) -> None:
        """物理后端的一次性初始化。"""

    @abstractmethod
    def _reset_simulation(self, rng: np.random.Generator) -> None:
        """
        将无人机和平台重置到随机初始状态。

        参数：
            rng: 带种子的随机数生成器，用于可复现随机化。
        """

    @abstractmethod
    def _apply_action(self, vel_cmd: np.ndarray) -> None:
        """
        向无人机发送速度指令 [vx, vy, vz, omega_yaw]。
        单位：WORLD 坐标系下的 m/s 和 rad/s（已从机体系转换）。
        """

    @abstractmethod
    def _step_simulation(self) -> None:
        """推进一个控制 timestep（config.episode.dt）。"""

    @abstractmethod
    def _get_drone_state(self) -> Dict[str, np.ndarray]:
        """
        返回当前无人机状态字典：
          'position':  np.ndarray (3,)  世界系 XYZ (m)
          'velocity':  np.ndarray (3,)  世界系实际线速度 (m/s)
          'euler':     np.ndarray (3,)  [roll, pitch, yaw] (rad)
          'yaw_rate':  float            yaw 角速度 (rad/s)
        """

    @abstractmethod
    def _get_platform_state(self) -> Dict[str, np.ndarray]:
        """
        返回当前平台状态字典：
          'position':  np.ndarray (3,)  世界系 XYZ (m)
          'velocity':  np.ndarray (3,)  世界系线速度 (m/s)
          'euler':     np.ndarray (3,)  [roll, pitch, yaw] (rad)
          'angular_rate': np.ndarray (3,) [roll_rate, pitch_rate, yaw_rate] (rad/s)
          'detection_quality': float      传感器置信度，范围 [0, 1]
        """

    @abstractmethod
    def _check_contact(self) -> bool:
        """如果无人机与平台发生物理接触，返回 True。"""

    @abstractmethod
    def render(self) -> Optional[np.ndarray]:
        """渲染当前帧，返回 RGB 数组或 None。"""

    @abstractmethod
    def close(self) -> None:
        """释放所有仿真资源。"""

    # =========================================================================
    # 课程管理（Trainer 手动切换阶段时调用）
    # =========================================================================

    def set_curriculum_stage(self, stage: int) -> None:
        """兼容旧训练工具的包装方法。"""
        from curriculum.strategies import create_strategy

        self.set_strategy(create_strategy(stage, self.config))

    def get_curriculum_stage(self) -> int:
        return int(self.strategy.stage_id())

    def set_strategy(self, strategy: CurriculumStrategy) -> None:
        """切换下一次 reset 使用的活动课程策略。"""
        if strategy is None:
            raise TypeError("set_strategy() requires a CurriculumStrategy instance.")
        strategy.config = self.config
        self.strategy = strategy
        self.curriculum_stage = int(strategy.stage_id())
        self._current_hover_height = float(strategy.get_hover_height())

    def set_hover_height(self, height: float) -> None:
        """设置支持悬停高度的活动策略目标高度。"""
        self._current_hover_height = float(height)
        self.strategy.set_hover_height(height)

    def get_hover_height(self) -> float:
        return float(self.strategy.get_hover_height())

    def set_success_mode(self, mode: str) -> None:
        """设置策略终局 success 口径：训练或评估。"""
        mode = str(mode).strip().lower()
        if mode not in {"train", "eval"}:
            raise ValueError("success mode must be 'train' or 'eval'")
        self._success_mode = mode

    def get_success_mode(self) -> str:
        return str(self._success_mode)

    # =========================================================================
    # Gymnasium API：所有后端共享的实现
    # =========================================================================

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        self._step_count = 0
        self._prev_action = np.zeros(self.ACT_DIM, dtype=np.float32)
        self._episode_reward = 0.0
        self._episode_success = False
        self._hover_hold_steps = 0
        self._hover_total_hold_steps = 0
        self._current_hover_height = 2.0
        self._stage1_r_hold_total = 0.0
        self._stage1_hold_refund_steps = 0
        self._stage1_is_refunding_hold = False
        self._randomize_capabilities(rng)

        self._reset_simulation(rng)

        drone_state = self._get_drone_state()
        platform_state = self._get_platform_state()
        self._reset_hold_tracking()
        self.strategy.reset_episode_metrics(self, drone_state, platform_state)
        obs = self._build_observation(drone_state, platform_state)
        return obs, {}

    def step(self, action: np.ndarray):
        self._step_count += 1

        # 1. 缩放动作（机体系）并转换到世界系
        vel_cmd_body = self._scale_action(action)
        current_yaw = float(self._get_drone_state()["euler"][2])
        cy, sy = np.cos(current_yaw), np.sin(current_yaw)
        vel_cmd_world = np.array([
            cy * vel_cmd_body[0] - sy * vel_cmd_body[1],   # vx_world
            sy * vel_cmd_body[0] + cy * vel_cmd_body[1],   # vy_world
            vel_cmd_body[2],                                 # vz 不变
            vel_cmd_body[3],                                 # yaw_rate 不变
        ], dtype=np.float32)
        self._apply_action(vel_cmd_world)
        self._step_simulation()

        # 2. 观测新状态
        drone_state = self._get_drone_state()
        platform_state = self._get_platform_state()

        # 3. 计算观测
        obs = self._build_observation(drone_state, platform_state)

        # 4. 当活动策略使用悬停评分时，跟踪悬停保持步数
        prev_hold_steps = self._hover_hold_steps
        if self.strategy.is_hover_stage():
            target = self._get_target_pos(platform_state["position"])
            if self.strategy.is_hold_stable(
                drone_state=drone_state,
                platform_state=platform_state,
                target_pos=target,
            ):
                self._hover_hold_steps += 1
                self._hover_total_hold_steps += 1
            else:
                self._hover_hold_steps = 0
        else:
            self._hover_hold_steps = 0

        self.strategy.update_step_metrics(
            self,
            drone_state,
            platform_state,
            self._get_target_pos(platform_state["position"]),
        )

        # 5. 计算稠密奖励
        target_pos = self._get_target_pos(platform_state["position"])
        reward, reward_info = self.strategy.compute_reward(
            env=self,
            drone_state=drone_state,
            platform_state=platform_state,
            action=action,
            prev_action=self._prev_action,
            target_pos=target_pos,
            hold_steps=self._hover_hold_steps,
            prev_hold_steps=prev_hold_steps,
        )

        # 6. 检查终止条件
        terminated, term_info = self._check_termination(drone_state, platform_state)

        truncated = self._step_count >= self.config.episode.max_steps

        # 7. 应用策略负责的终止奖励塑形
        episode_done = terminated or truncated
        if episode_done:
            bonus, success = self.strategy.terminal_bonus(
                env=self,
                terminated=terminated,
                truncated=truncated,
                term_info=term_info,
            )
            reward += bonus
            self._episode_success = bool(success)

        self._episode_reward += reward
        self._prev_action = action.copy()

        # 8. 构建 info 字典
        info = {**reward_info, **term_info, "stage": self.curriculum_stage}
        wind_velocity = drone_state.get("wind_velocity", np.zeros(3, dtype=np.float32))
        base_wind_velocity = drone_state.get("base_wind_velocity", np.zeros(3, dtype=np.float32))
        gust_velocity = drone_state.get("gust_velocity", np.zeros(3, dtype=np.float32))
        tracking_scale = drone_state.get("tracking_scale", np.ones(3, dtype=np.float32))
        info.update({
            "metric/wind_speed": float(np.linalg.norm(wind_velocity)),
            "metric/base_wind_speed": float(np.linalg.norm(base_wind_velocity)),
            "metric/gust_speed": float(np.linalg.norm(gust_velocity)),
            "metric/tracking_scale_mean": float(np.mean(tracking_scale)),
        })
        info.update(self.strategy.get_step_metrics(self))

        if episode_done:
            episode_metrics = self.strategy.get_episode_metrics(self)
            info.update({
                "success": bool(self._episode_success),
                "episode_stage": int(self.curriculum_stage),
            })
            info.update(episode_metrics)
            info["episode"] = {
                "r": float(self._episode_reward),
                "l": int(self._step_count),
                "success": bool(self._episode_success),
                "stage": self.curriculum_stage,
                "episode_stage": int(self.curriculum_stage),
            }
            info["episode"].update(episode_metrics)

        return obs, float(reward), terminated, truncated, info

    # =========================================================================
    # 内部辅助函数：所有后端共享
    # =========================================================================

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """将 [-1,1] 中的归一化动作映射为机体系速度指令。"""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        z_action = float(action[2])
        vz = z_action * (self._current_vz_up_max if z_action >= 0.0
                         else self._current_vz_down_max)
        return np.array([
            action[0] * self._current_v_xy_max,    # 前向
            action[1] * self._current_v_xy_max,    # 左向
            vz,                                      # 垂直
            action[3] * self.config.drone.max_yaw_rate,
        ], dtype=np.float32)

    def _randomize_capabilities(self, rng: np.random.Generator) -> None:
        """采样每个 episode 的无人机速度上限，用于动作和观测。"""
        oc = self.config.observation
        self._current_v_xy_max = float(rng.uniform(oc.v_xy_min, oc.v_xy_max))
        self._current_vz_up_max = float(rng.uniform(oc.v_z_up_min, oc.v_z_up_max))
        down_hi = min(float(oc.v_z_down_max), self._current_vz_up_max)
        down_lo = min(float(oc.v_z_down_min), down_hi)
        self._current_vz_down_max = float(rng.uniform(down_lo, down_hi))

    def _reset_hold_tracking(self) -> None:
        """重置连续保持步数状态。"""
        self._hover_hold_steps = 0
        self._hover_total_hold_steps = 0
        self._stage1_r_hold_total = 0.0
        self._stage1_hold_refund_steps = 0
        self._stage1_is_refunding_hold = False

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    @staticmethod
    def _norm_signed_z(value: float, up_norm: float, down_norm: float) -> float:
        denom = up_norm if value >= 0.0 else down_norm
        return float(value) / (denom + 1e-6)

    def _get_target_pos(self, platform_pos: np.ndarray) -> np.ndarray:
        """
        计算当前活动课程策略的目标位置。
        """
        return self.strategy.get_target_pos(platform_pos)

    def _build_observation(
        self,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        组装并归一化 34 维观测向量。
        平移量使用 yaw-FLU：x 向前，y 向左，z 向上。XY 由无人机 yaw 旋转；
        z 保持世界系竖直方向。裁剪只影响模型输入，不影响奖励或终止计算。
        """
        oc = self.config.observation
        drone_pos  = drone_state["position"]
        drone_vel  = drone_state["velocity"]
        drone_eul  = drone_state["euler"]
        yaw_rate   = drone_state["yaw_rate"]
        plat_pos   = platform_state["position"]
        plat_vel   = platform_state["velocity"]
        plat_eul   = platform_state.get("euler", np.zeros(3, dtype=np.float32))
        plat_rates = platform_state.get("angular_rate", np.zeros(3, dtype=np.float32))
        quality    = float(platform_state.get("detection_quality", 1.0))

        target_pos = self._get_target_pos(plat_pos)
        rel_pos    = target_pos - drone_pos
        rel_vel    = plat_vel - drone_vel

        # ── 世界系 → 机体系旋转（仅 XY，Z 不变）────────────────────────────
        yaw = float(drone_eul[2])
        cy, sy = np.cos(yaw), np.sin(yaw)

        def _rotate_xy(vec: np.ndarray) -> np.ndarray:
            """将 XY 分量从世界系旋转到机体系。"""
            return np.array([
                cy * vec[0] + sy * vec[1],
               -sy * vec[0] + cy * vec[1],
                vec[2],
            ])

        rel_pos_body  = _rotate_xy(rel_pos)
        rel_vel_body  = _rotate_xy(rel_vel)
        drone_vel_body = _rotate_xy(drone_vel)
        plat_vel_body  = _rotate_xy(plat_vel)

        def _norm_vel(vec: np.ndarray) -> np.ndarray:
            return np.array([
                vec[0] / (oc.v_xy_norm + 1e-6),
                vec[1] / (oc.v_xy_norm + 1e-6),
                self._norm_signed_z(float(vec[2]), oc.v_z_up_norm, oc.v_z_down_norm),
            ], dtype=np.float32)

        rel_yaw = self._wrap_pi(float(plat_eul[2]) - float(drone_eul[2]))
        rel_yaw_rate = float(plat_rates[2]) - float(yaw_rate)
        range_norm = float(np.linalg.norm(rel_pos)) / (oc.range_norm + 1e-6)
        # 面向未来真实飞行部署的 Local/AGL 高度抽象。
        # 当前 PyBullet 设置中，platform_z 也作为局部地面/海面基准。
        height_norm = float(drone_pos[2] - plat_pos[2]) / (oc.height_norm + 1e-6)

        obs = np.concatenate([
            np.array([
                rel_pos_body[0] / (oc.rel_xy_norm + 1e-6),
                rel_pos_body[1] / (oc.rel_xy_norm + 1e-6),
                rel_pos_body[2] / (oc.rel_z_norm + 1e-6),
            ], dtype=np.float32),                                      # [0:3]
            _norm_vel(rel_vel_body),                                   # [3:6]
            np.array([
                rel_yaw / np.pi,
                rel_yaw_rate / (oc.yaw_rate_norm + 1e-6),
                range_norm,
            ], dtype=np.float32),                                      # [6:9]
            _norm_vel(drone_vel_body),                                 # [9:12]
            drone_eul / np.pi,                                         # [12:15]
            np.array([float(yaw_rate) / (oc.yaw_rate_norm + 1e-6),
                      height_norm], dtype=np.float32),                 # [15:17]
            plat_eul / np.pi,                                          # [17:20]
            np.array([
                float(plat_rates[0]) / (oc.roll_rate_norm + 1e-6),
                float(plat_rates[1]) / (oc.pitch_rate_norm + 1e-6),
                float(plat_rates[2]) / (oc.yaw_rate_norm + 1e-6),
            ], dtype=np.float32),                                      # [20:23]
            _norm_vel(plat_vel_body),                                  # [23:26]
            np.array([
                quality,
                self._current_v_xy_max / (oc.v_xy_norm + 1e-6),
                self._current_vz_down_max / (oc.v_z_down_norm + 1e-6),
                self._current_vz_up_max / (oc.v_z_up_norm + 1e-6),
            ], dtype=np.float32),                                      # [26:30]
            self._prev_action,                                         # [30:34]
        ]).astype(np.float32)

        return self._clip_observation(obs)

    def _clip_observation(self, obs: np.ndarray) -> np.ndarray:
        """对模型输入应用按分组的归一化后裁剪。"""
        oc = self.config.observation
        out = obs.copy()
        out[0:3]   = np.clip(out[0:3],   -oc.clip_signed, oc.clip_signed)
        out[3:6]   = np.clip(out[3:6],   -oc.clip_signed, oc.clip_signed)
        out[6]     = np.clip(out[6],     -oc.clip_angle,  oc.clip_angle)
        out[7]     = np.clip(out[7],     -oc.clip_rate,   oc.clip_rate)
        out[8]     = np.clip(out[8],      0.0,            oc.clip_nonnegative)
        out[9:12]  = np.clip(out[9:12],  -oc.clip_signed, oc.clip_signed)
        out[12:15] = np.clip(out[12:15], -oc.clip_angle,  oc.clip_angle)
        out[15]    = np.clip(out[15],    -oc.clip_rate,   oc.clip_rate)
        out[16]    = np.clip(out[16],     0.0,            oc.clip_nonnegative)
        out[17:20] = np.clip(out[17:20], -oc.clip_angle,  oc.clip_angle)
        out[20:23] = np.clip(out[20:23], -oc.clip_rate,   oc.clip_rate)
        out[23:26] = np.clip(out[23:26], -oc.clip_signed, oc.clip_signed)
        out[26]    = np.clip(out[26],     0.0,            1.0)
        out[27:30] = np.clip(out[27:30],  0.0,            oc.clip_capability)
        out[30:34] = np.clip(out[30:34], -1.0,            1.0)
        return out.astype(np.float32)

    def _check_termination(
        self,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
    ) -> Tuple[bool, dict]:
        """检查所有 episode 终止条件。"""
        dp  = drone_state["position"]
        pp  = platform_state["position"]

        horiz_dist = np.linalg.norm(dp[:2] - pp[:2])

        # 越界
        if horiz_dist > self.config.episode.max_horiz_dist:
            return True, {"termination": "oob", "success": False}

        # 低于地面
        if dp[2] < self.config.episode.min_height:
            return True, {"termination": "below_ground", "success": False}

        # 由策略负责的成功/接触终止。
        return self.strategy.check_success(
            env=self,
            drone_state=drone_state,
            platform_state=platform_state,
        )
