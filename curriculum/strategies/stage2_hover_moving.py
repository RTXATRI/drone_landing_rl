# 功能：课程二——移动平台上方悬停的策略和奖励函数。
#
# 核心差异 vs 课程一：
#   - 相对速度替代绝对速度
#   - 移动平台 + 速度控制器 + 轻量观测延迟/误差
#   - 3 种运动模式池随机选择
#
# 奖励设计：
#   - 三层位置函数：反二次（长尾）+ 高斯（中距精度）+ 窄高斯（厘米级梯度）
#   - 远距相对闭合速度奖励：鼓励相对平台朝目标移动，抑制初始外逃
#   - 速度匹配惩罚：三轴误差修正速度模型，允许位置误差收敛速度
#   - yaw / yaw_rate / action 平滑惩罚保留

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from curriculum.strategies.base_strategy import CurriculumStrategy, HoverStrategyMixin
from envs.landing_platform.motions import (
    LissajousMotion,
    PatrolMotion,
    WaypointMotion,
)

MOTION_CLASSES = {
    "patrol":     PatrolMotion,
    "lissajous":  LissajousMotion,
    "waypoint":   WaypointMotion,
}
MOTION_POOL = list(MOTION_CLASSES.keys())

PLATFORM_BOUNDARY_XY = 50.0       # 100m × 100m


class Stage2HoverMovingStrategy(HoverStrategyMixin, CurriculumStrategy):
    label = "Hover  | Moving Platform"
    short_label = "Hover-Moving"
    episode_metrics = ("train_score", "eval_score")

    # ── 传感器噪声 / 延迟（多方法共享）─────────────────────────────────────
    SENSOR_NOISE_POS = 0.03     # 位置观测噪声 σ (m)
    SENSOR_NOISE_YAW = 0.03     # 偏航观测噪声 σ (rad)
    SENSOR_DROP_PROB  = 0.05    # 检测丢帧概率
    VELOCITY_DELAY_STEPS = 2    # 速度观测延迟步数
    VELOCITY_SCALE_MIN = 0.98   # 速度比例误差下限
    VELOCITY_SCALE_MAX = 1.02   # 速度比例误差上限

    # ── OOB / 训练/评估分数空间（多方法共享）────────────────────────────────
    OOB_HORIZ_DIST = 65.0           # 越界水平距离 (m)，即 130m×130m

    TRAIN_SCORE_RADIUS = 0.25       # train_score 水平半径 (m)
    TRAIN_SCORE_VERT_TOL = 0.15     # train_score 垂直容差 (m)

    EVAL_SCORE_RADIUS = 0.10        # eval_score 水平半径 (m)
    EVAL_SCORE_VERT_TOL = 0.05      # eval_score 垂直容差 (m)

    EVAL_SCORE_MISS_RESET_STEPS = 10  # eval 连续不稳定达到该步数后重置

    SCORE_SPEED_REL_TOL = 0.15      # train/eval 水平速度大小相对误差上限
    SCORE_SPEED_ANGLE_MAX_DEG = 15.0  # train/eval 水平速度方向误差上限(角度值)
    SCORE_VERT_SPEED_MAX = 0.05     # train/eval 最大垂直速度 (m/s)
    
    SCORE_SPEED_EPS = 1e-6          # 速度判定零速阈值，避免除零

    SUCCESS_SCORE_THRESHOLD = 60.0  # 成功分数阈值 (0-100)
    FAILURE_TERMINAL_PENALTY = -500.0  # 越界/坠地/坠毁失败终止惩罚

    def __init__(self, env_config, motion=None):
        super().__init__(env_config)
        self._motion_override = motion                # None = 使用随机池
        self._motion_active = "lissajous"              # 当前 ep 的运动名（日志用）
        self._velocity_history: list[np.ndarray] = []
        self._velocity_scale_xy = np.ones(2, dtype=np.float32)

    def stage_id(self) -> int:
        return 2

    # ── 策略 hooks ────────────────────────────────────────────────────────────

    def get_max_horiz_dist(self) -> float | None:
        return self.OOB_HORIZ_DIST

    def get_map_bounds(self) -> tuple | None:
        """绝对地图边界：|x|,|y| ≤ 100m, z ≥ 0。"""
        return (100.0, 0.0, 200.0)

    def get_spawn_position(
        self, platform_pos: np.ndarray, rng: np.random.Generator,
    ) -> np.ndarray | None:
        """无人机出生在 [-80,80]² 外环（排除 [-50,50]² 平台活动区）。"""
        height = float(rng.uniform(
            self.config.episode.init_spawn_height_min,
            self.config.episode.init_spawn_height_max,
        ))
        for _ in range(50):
            x = float(rng.uniform(-80.0, 80.0))
            y = float(rng.uniform(-80.0, 80.0))
            if abs(x) > 50.0 or abs(y) > 50.0:
                return np.array([x, y, height], dtype=np.float64)
        return np.array([55.0, 55.0, height], dtype=np.float64)

    def process_platform_state(
        self, raw_state: dict, rng: np.random.Generator | None = None,
    ) -> dict:
        if rng is None:
            return raw_state

        observed = self._inject_observation_noise(raw_state, rng)
        raw_vel = np.asarray(raw_state["velocity"], dtype=np.float32).copy()
        self._velocity_history.append(raw_vel)
        max_len = int(self.VELOCITY_DELAY_STEPS) + 1
        if len(self._velocity_history) > max_len:
            self._velocity_history = self._velocity_history[-max_len:]
        delayed_vel = self._velocity_history[0].copy()
        delayed_vel[:2] *= self._velocity_scale_xy
        observed["velocity"] = delayed_vel.astype(np.float32)
        return observed

    # ── 场景设置 ──────────────────────────────────────────────────────────────

    def setup_scene(self, env, rng: np.random.Generator) -> None:
        self._apply_hover_height(env, self._sample_hover_height(rng))

        # 选择运动策略
        if self._motion_override is not None:
            motion_name = self._motion_override
        else:
            motion_name = MOTION_POOL[rng.integers(0, len(MOTION_POOL))]
        self._motion_active = str(motion_name)

        motion_cls = MOTION_CLASSES[motion_name]
        if motion_name in ("patrol", "waypoint"):
            motion = motion_cls(boundary=PLATFORM_BOUNDARY_XY)
        else:
            motion = motion_cls()
        env._platform.set_motion_strategy(motion)

        # 激活课程二专属功能
        env._platform.enable_speed_controller()
        env._platform.set_boundary(PLATFORM_BOUNDARY_XY)

        self._velocity_history = []
        self._velocity_scale_xy = rng.uniform(
            self.VELOCITY_SCALE_MIN,
            self.VELOCITY_SCALE_MAX,
            size=2,
        ).astype(np.float32)

    # ── 传感器噪声注入 ───────────────────────────────────────────────────────

    def _inject_observation_noise(
        self, raw_state: dict, rng: np.random.Generator,
    ) -> dict:
        noisy = {
            "position": raw_state["position"].copy(),
            "velocity": raw_state["velocity"].copy(),
            "euler": raw_state["euler"].copy(),
            "angular_rate": np.asarray(raw_state["angular_rate"], dtype=np.float32).copy(),
            "detection_quality": np.float32(raw_state["detection_quality"]),
        }
        noisy["position"][:2] += rng.normal(
            0.0, self.SENSOR_NOISE_POS, 2
        ).astype(np.float32)
        noisy["euler"][2] += np.float32(rng.normal(0.0, self.SENSOR_NOISE_YAW))

        if rng.random() < self.SENSOR_DROP_PROB:
            noisy["detection_quality"] = np.float32(0.0)

        return noisy

    # ── Episode 指标初始化 ───────────────────────────────────────────────────

    def reset_episode_metrics(
        self,
        env: Any,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
    ) -> None:
        # episode 指标
        env._stage2_train_hit_steps = 0
        env._stage2_eval_hold_steps = 0
        env._stage2_eval_max_hold_steps = 0
        env._stage2_eval_miss_steps = 0
        env._stage2_eval_hold_active = False
        env._stage2_train_possible_steps = self._possible_steps(
            env, drone_state, platform_state,
            horiz_tol=self.TRAIN_SCORE_RADIUS,
            vert_tol=self.TRAIN_SCORE_VERT_TOL,
        )
        env._stage2_eval_possible_steps = self._possible_steps(
            env, drone_state, platform_state,
            horiz_tol=self.EVAL_SCORE_RADIUS,
            vert_tol=self.EVAL_SCORE_VERT_TOL,
        )
        # 近距连续梯度由 peak 层提供。

    # ── 保持判定 / 成功辅助 ───────────────────────────────────────────────────

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _horizontal_velocity_match(
        self,
        *,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
    ) -> bool:
        drone_vel_xy = np.asarray(drone_state["velocity"][:2], dtype=np.float64)
        platform_vel_xy = np.asarray(platform_state["velocity"][:2], dtype=np.float64)
        drone_speed = float(np.linalg.norm(drone_vel_xy))
        platform_speed = float(np.linalg.norm(platform_vel_xy))
        eps = float(self.SCORE_SPEED_EPS)

        if platform_speed <= eps:
            return bool(drone_speed <= eps)
        if drone_speed <= eps:
            return False

        speed_rel_err = abs(drone_speed - platform_speed) / platform_speed
        if speed_rel_err > self.SCORE_SPEED_REL_TOL:
            return False

        cos_angle = float(np.dot(drone_vel_xy, platform_vel_xy)) / (
            drone_speed * platform_speed
        )
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        return bool(angle_deg <= self.SCORE_SPEED_ANGLE_MAX_DEG)

    def _vertical_velocity_stable(
        self,
        *,
        drone_state: Dict[str, np.ndarray],
    ) -> bool:
        return abs(float(drone_state["velocity"][2])) < self.SCORE_VERT_SPEED_MAX

    def is_hold_stable(
        self,
        *,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
        target_pos: np.ndarray,
    ) -> bool:
        target_vec = target_pos - drone_state["position"]
        horiz_err = float(np.linalg.norm(target_vec[:2]))
        vert_err = abs(float(target_vec[2]))
        return (
            horiz_err < self.EVAL_SCORE_RADIUS
            and vert_err < self.EVAL_SCORE_VERT_TOL
            and self._horizontal_velocity_match(
                drone_state=drone_state,
                platform_state=platform_state,
            )
            and self._vertical_velocity_stable(drone_state=drone_state)
        )

    def _state_in_box(
        self,
        *,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
        target_pos: np.ndarray,
        horiz_tol: float,
        vert_tol: float,
    ) -> bool:
        target_vec = target_pos - drone_state["position"]
        horiz_err = float(np.linalg.norm(target_vec[:2]))
        vert_err = abs(float(target_vec[2]))
        return (
            horiz_err < horiz_tol
            and vert_err < vert_tol
            and self._horizontal_velocity_match(
                drone_state=drone_state,
                platform_state=platform_state,
            )
            and self._vertical_velocity_stable(drone_state=drone_state)
        )

    def _possible_steps(
        self,
        env: Any,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
        *,
        horiz_tol: float,
        vert_tol: float,
    ) -> int:
        target = self.get_target_pos(platform_state["position"])
        rel = target - drone_state["position"]
        horiz_dist = float(np.linalg.norm(rel[:2]))
        vert_dist = abs(float(rel[2]))

        xy_cap = max(float(env._current_v_xy_max), 1e-6)
        z_cap = env._current_vz_up_max if rel[2] >= 0.0 else env._current_vz_down_max
        z_cap = max(float(z_cap), 1e-6)

        entry_xy = max(0.0, horiz_dist - float(horiz_tol))
        entry_z = max(0.0, vert_dist - float(vert_tol))
        t_arrive = max(entry_xy / xy_cap, entry_z / z_cap)
        arrive_steps = int(np.ceil(t_arrive / (self.config.episode.dt + 1e-9)))
        return max(1, int(self.config.episode.max_steps) - arrive_steps)

    # ── 评分 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _score(numer_steps: int, possible_steps: int) -> float:
        denom = max(int(possible_steps), 1)
        score = 100.0 * float(numer_steps) / float(denom)
        return float(np.clip(score, 0.0, 100.0))

    def _train_score(self, env: Any) -> float:
        return self._score(
            int(getattr(env, "_stage2_train_hit_steps", 0)),
            int(getattr(env, "_stage2_train_possible_steps", 1)),
        )

    def _eval_score(self, env: Any) -> float:
        return self._score(
            int(getattr(env, "_stage2_eval_max_hold_steps", 0)),
            int(getattr(env, "_stage2_eval_possible_steps", 1)),
        )

    # ── Episode 指标 ──────────────────────────────────────────────────────────

    def update_step_metrics(
        self,
        env: Any,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
        target_pos: np.ndarray,
    ) -> None:
        if self._state_in_box(
            drone_state=drone_state,
            platform_state=platform_state,
            target_pos=target_pos,
            horiz_tol=self.TRAIN_SCORE_RADIUS,
            vert_tol=self.TRAIN_SCORE_VERT_TOL,
        ):
            env._stage2_train_hit_steps = int(
                getattr(env, "_stage2_train_hit_steps", 0)
            ) + 1

        eval_stable = self.is_hold_stable(
            drone_state=drone_state,
            platform_state=platform_state,
            target_pos=target_pos,
        )
        eval_active = bool(getattr(env, "_stage2_eval_hold_active", False))
        eval_hold_steps = int(getattr(env, "_stage2_eval_hold_steps", 0))
        eval_miss_steps = int(getattr(env, "_stage2_eval_miss_steps", 0))

        if eval_stable:
            env._stage2_eval_hold_active = True
            env._stage2_eval_miss_steps = 0
            env._stage2_eval_hold_steps = eval_hold_steps + 1
        elif eval_active:
            eval_miss_steps += 1
            if eval_miss_steps < self.EVAL_SCORE_MISS_RESET_STEPS:
                env._stage2_eval_miss_steps = eval_miss_steps
                env._stage2_eval_hold_steps = eval_hold_steps + 1
            else:
                env._stage2_eval_hold_active = False
                env._stage2_eval_miss_steps = 0
                env._stage2_eval_hold_steps = 0
        else:
            env._stage2_eval_hold_active = False
            env._stage2_eval_miss_steps = 0
            env._stage2_eval_hold_steps = 0
        env._stage2_eval_max_hold_steps = max(
            int(getattr(env, "_stage2_eval_max_hold_steps", 0)),
            int(getattr(env, "_stage2_eval_hold_steps", 0)),
        )

    def get_episode_metrics(self, env: Any) -> dict:
        return {
            "train_score": float(self._train_score(env)),
            "eval_score": float(self._eval_score(env)),
        }

    def get_step_metrics(self, env: Any) -> dict:
        return {"metric/motion_type": self._motion_active}

    # ── 奖励函数 ───────────────────────────────────────────────────
    #
    # 设计思路：
    #   1. 三层位置函数：反二次（长尾）+ 高斯（中距精度）+ 窄高斯（厘米级梯度）
    #   2. 远距相对闭合速度奖励：鼓励相对平台朝目标移动
    #   3. 速度匹配惩罚：三轴误差修正速度模型，避免近距速度匹配压制位置收敛
    #   4. yaw / yaw_rate / action 平滑惩罚保留

    def compute_reward(
        self,
        *,
        env: Any = None,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
        action: np.ndarray,
        prev_action: np.ndarray,
        target_pos: np.ndarray,
        hold_steps: int = 0,
        prev_hold_steps: int = 0,
    ) -> Tuple[float, dict]:
        drone_pos = drone_state["position"]
        drone_vel = drone_state["velocity"]
        plat_vel = platform_state["velocity"]

        target_vec = target_pos - drone_pos
        horiz_err = float(np.linalg.norm(target_vec[:2]))
        vert_err = abs(float(target_vec[2]))
        dist = float(np.linalg.norm(target_vec))
        speed = float(np.linalg.norm(drone_vel))
        rel_vel = drone_vel - plat_vel
        rel_speed = float(np.linalg.norm(rel_vel))
        plat_speed = float(np.linalg.norm(plat_vel))

        # ── 位置奖励 r_pos ：四层叠加 ──
        # r_approach：米级远距接近梯度，保证远离目标时仍有方向性。
        POS_APPROACH_WEIGHT = 3.0
        POS_APPROACH_SIGMA_XY = 5.0     # 米级水平半衰尺度 (m)
        POS_APPROACH_SIGMA_Z = 3.0      # 米级垂直半衰尺度 (m)

        r_approach = POS_APPROACH_WEIGHT / ( 1.0 + (horiz_err / POS_APPROACH_SIGMA_XY) ** 2 + (vert_err / POS_APPROACH_SIGMA_Z) ** 2)

        # r_mid_precision：约 1m 尺度的中距精度梯度。
        POS_MID_PRECISION_WEIGHT = 1.5
        POS_MID_PRECISION_SIGMA_XY = 0.5
        POS_MID_PRECISION_SIGMA_Z = 0.5

        r_mid_precision = POS_MID_PRECISION_WEIGHT * float(np.exp( -(horiz_err ** 2) / (2.0 * POS_MID_PRECISION_SIGMA_XY ** 2) - (vert_err ** 2) / (2.0 * POS_MID_PRECISION_SIGMA_Z ** 2)))

        # r_near_peak：约 0.20~0.30m 尺度的近距峰值，提供分米级收敛梯度。
        POS_NEAR_PEAK_WEIGHT = 1.5
        POS_NEAR_PEAK_SIGMA_XY = 0.20
        POS_NEAR_PEAK_SIGMA_Z = 0.20

        r_near_peak = POS_NEAR_PEAK_WEIGHT * float(np.exp( -(horiz_err ** 2) / (2.0 * POS_NEAR_PEAK_SIGMA_XY ** 2) - (vert_err ** 2) / (2.0 * POS_NEAR_PEAK_SIGMA_Z ** 2)))

        # r_centimeter_refine：约 0.10m 尺度的厘米级精修峰值。
        POS_CENTIMETER_REFINE_WEIGHT = 1.0
        POS_CENTIMETER_REFINE_SIGMA_XY = 0.05
        POS_CENTIMETER_REFINE_SIGMA_Z = 0.05

        r_centimeter_refine = POS_CENTIMETER_REFINE_WEIGHT * float(np.exp( -(horiz_err ** 2) / (2.0 * POS_CENTIMETER_REFINE_SIGMA_XY ** 2) - (vert_err ** 2) / (2.0 * POS_CENTIMETER_REFINE_SIGMA_Z ** 2)))

        # 计算位置奖励总和，中心峰值为 7.0。
        r_pos = r_approach + r_mid_precision + r_near_peak + r_centimeter_refine

        # ── 远距相对闭合速度奖励 ──
        CLOSING_WEIGHT = 0.50
        CLOSING_V_REF = 2.0              # m/s, tanh 限幅参考速度
        CLOSING_DIST_INNER = 0.75        # 近距关闭，避免干扰速度匹配
        CLOSING_DIST_OUTER = 3.00        # 远距全开
        target_dir = target_vec / (dist + 1e-6)
        closing_speed = float(np.dot(rel_vel, target_dir))
        gate_far = self._smoothstep(
            (dist - CLOSING_DIST_INNER)
            / (CLOSING_DIST_OUTER - CLOSING_DIST_INNER)
        )
        r_closing = CLOSING_WEIGHT * gate_far * float(np.tanh(
            closing_speed / CLOSING_V_REF
        ))

        # ── 速度匹配惩罚 r_vel_match ：误差修正型速度匹配 ──
        # 该项要求 v_drone ~= v_platform - Kp * (p_drone - p_target)
        # 这样位置误差存在时允许相对平台产生修正速度，避免速度匹配惩罚
        # 压制位置误差收敛；当 rel_pos -> 0 时，自然退化为平台速度匹配。

        VEL_MATCH_GATE_INNER = 0.10       # XY 速度惩罚全开水平误差阈值 (m)
        VEL_MATCH_GATE_OUTER = 0.50       # XY 速度惩罚全关水平误差阈值 (m)
        VEL_MATCH_Z_GATE_INNER = 0.10     # Z 速度惩罚全开垂直误差阈值 (m)
        VEL_MATCH_Z_GATE_OUTER = 0.30     # Z 速度惩罚全关垂直误差阈值 (m)
        VEL_MATCH_EPS = 1e-6              # 避免归一化除零
        VEL_MATCH_XY_WEIGHT = 0.20        # XY 速度惩罚权重
        VEL_MATCH_Z_WEIGHT = 0.25         # Z 速度惩罚权重
        VEL_CORR_GAIN_X = 0.7             # X 轴位置误差到期望相对速度的比例增益
        VEL_CORR_GAIN_Y = 0.7             # Y 轴位置误差到期望相对速度的比例增益
        VEL_CORR_GAIN_Z = 0.6             # Z 轴位置误差到期望相对速度的比例增益
        VEL_CORR_MAX_X = 0.25             # X 轴最大期望修正速度 (m/s)
        VEL_CORR_MAX_Y = 0.25             # Y 轴最大期望修正速度 (m/s)
        VEL_CORR_MAX_Z = 0.10             # Z 轴最大期望修正速度 (m/s)
        VEL_ERR_SCALE_X = max(VEL_CORR_MAX_X, VEL_MATCH_EPS)  # X 轴速度误差归一化尺度
        VEL_ERR_SCALE_Y = max(VEL_CORR_MAX_Y, VEL_MATCH_EPS)  # Y 轴速度误差归一化尺度
        VEL_ERR_SCALE_Z = max(VEL_CORR_MAX_Z, VEL_MATCH_EPS)  # Z 轴速度误差归一化尺度

        gate_xy = self._smoothstep(
            (VEL_MATCH_GATE_OUTER - horiz_err)
            / (VEL_MATCH_GATE_OUTER - VEL_MATCH_GATE_INNER)
        )
        gate_z = self._smoothstep(
            (VEL_MATCH_Z_GATE_OUTER - vert_err)
            / (VEL_MATCH_Z_GATE_OUTER - VEL_MATCH_Z_GATE_INNER)
        )

        # gate_near_3d 表示“水平和垂直都接近目标”的稳定区门控。
        # 数值可复用给 Z 速度、yaw、yaw_rate，但变量名按奖励项拆开，
        # 避免把垂直速度惩罚误读成由 yaw 条件控制。
        gate_near_3d = float(gate_xy * gate_z)
        gate_vel_z = gate_near_3d                   # 垂直速度惩罚要求 3D 近距
        gate_yaw = gate_near_3d                     # 偏航角只在 3D 近距时约束
        gate_yaw_rate = gate_near_3d                # 偏航角速度同偏航角门控

        rel_pos = drone_pos - target_pos
        rel_pos_xy = rel_pos[:2]
        rel_dist_xy = float(np.linalg.norm(rel_pos_xy))
        rel_vel_xy = rel_vel[:2]
        rel_speed_xy = float(np.linalg.norm(rel_vel_xy))
        rel_speed_z = abs(float(rel_vel[2]))

        def _smooth_l1(norm_error: float) -> float:
            norm_error = float(abs(norm_error))
            if norm_error < 1.0:
                return 0.5 * norm_error ** 2
            return norm_error - 0.5

        vel_rel_des_x = float(np.clip( -VEL_CORR_GAIN_X * float(rel_pos[0]), -VEL_CORR_MAX_X, VEL_CORR_MAX_X,))
        vel_rel_des_y = float(np.clip( -VEL_CORR_GAIN_Y * float(rel_pos[1]), -VEL_CORR_MAX_Y, VEL_CORR_MAX_Y,))
        vel_rel_des_z = float(np.clip( -VEL_CORR_GAIN_Z * float(rel_pos[2]), -VEL_CORR_MAX_Z, VEL_CORR_MAX_Z,))

        vel_err_x = float(rel_vel[0]) - vel_rel_des_x
        vel_err_y = float(rel_vel[1]) - vel_rel_des_y
        vel_err_z = float(rel_vel[2]) - vel_rel_des_z

        vel_err_norm_x = abs(vel_err_x) / VEL_ERR_SCALE_X
        vel_err_norm_y = abs(vel_err_y) / VEL_ERR_SCALE_Y
        vel_err_norm_z = abs(vel_err_z) / VEL_ERR_SCALE_Z

        loss_xy = _smooth_l1(vel_err_norm_x) + _smooth_l1(vel_err_norm_y)
        loss_z = _smooth_l1(vel_err_norm_z)

        r_vel_xy = -VEL_MATCH_XY_WEIGHT * gate_xy * loss_xy
        r_vel_z = -VEL_MATCH_Z_WEIGHT * gate_vel_z * loss_z

        # 计算最终速度惩罚
        r_vel_match = r_vel_xy + r_vel_z

        # ── 偏航角惩罚（水平/垂直都靠近时才关心偏航）──
        YAW_WEIGHT = 0.20
        yaw = float(drone_state["euler"][2])
        r_yaw = -YAW_WEIGHT * gate_yaw * abs(yaw)

        # ── 偏航角速度惩罚（水平/垂直都靠近时才关心角速度）──
        YAW_RATE_WEIGHT = 0.05
        yaw_rate = float(drone_state["yaw_rate"])
        r_yaw_rate = -YAW_RATE_WEIGHT * gate_yaw_rate * (yaw_rate ** 2)

        # ── 动作平滑和幅值惩罚 ──
        ACTION_SMOOTH_WEIGHT = 0.10   # 动作变化惩罚权重
        ACTION_MAG_WEIGHT = 0.01      # 动作幅值惩罚权重
        action_delta = action - prev_action
        r_action = ( -ACTION_SMOOTH_WEIGHT * float(np.sum(action_delta ** 2)) -ACTION_MAG_WEIGHT * float(np.sum(action ** 2)))

        # 计算总奖励
        total = r_pos + r_closing + r_vel_match + r_yaw + r_yaw_rate + r_action

        # 速度修正诊断：
        #   vel_rel_des_* 表示由位置误差给出的期望相对速度。
        #   vel_err_* 表示实际相对速度相对该期望值的残差。
        #   vel_err_norm_* 表示按各轴速度误差尺度归一化后的残差。
        info = {
            "reward/pos": r_pos,
            "reward/near_peak": r_near_peak,
            "reward/cm_refine": r_centimeter_refine,
            "reward/closing": r_closing,
            "reward/vel_match": r_vel_match,
            "reward/vel_match_xy": r_vel_xy,
            "reward/vel_match_z": r_vel_z,
            "reward/yaw": r_yaw,
            "reward/yaw_rate": r_yaw_rate,
            "reward/action": r_action,
            "reward/total": total,
            "metric/dist": dist,
            "metric/horiz_err": horiz_err,
            "metric/vert_err": vert_err,
            "metric/speed": speed,
            "metric/rel_speed": rel_speed,
            "metric/plat_speed": plat_speed,
            "metric/closing_speed": closing_speed,
            "metric/rel_dist_xy": rel_dist_xy,
            "metric/rel_speed_xy": rel_speed_xy,
            "metric/rel_speed_z": rel_speed_z,
            "metric/vel_rel_des_x": vel_rel_des_x,
            "metric/vel_rel_des_y": vel_rel_des_y,
            "metric/vel_rel_des_z": vel_rel_des_z,
            "metric/vel_err_x": vel_err_x,
            "metric/vel_err_y": vel_err_y,
            "metric/vel_err_z": vel_err_z,
            "metric/vel_err_norm_x": vel_err_norm_x,
            "metric/vel_err_norm_y": vel_err_norm_y,
            "metric/vel_err_norm_z": vel_err_norm_z,
            "metric/gate_far": gate_far,
            "metric/gate_xy": gate_xy,
            "metric/gate_z": gate_z,
            "metric/gate_near_3d": gate_near_3d,
            "metric/gate_vel_z": gate_vel_z,
            "metric/gate_yaw": gate_yaw,
            "metric/gate_yaw_rate": gate_yaw_rate,
        }

        return float(total), info

    def terminal_bonus(
        self,
        *,
        env: Any,
        terminated: bool,
        truncated: bool,
        term_info: dict,
    ) -> Tuple[float, bool]:
        if term_info.get("termination") in {"oob", "below_ground", "crashed"}:
            return self.FAILURE_TERMINAL_PENALTY, False
        if truncated:
            mode = env.get_success_mode() if env is not None else "train"
            score = self._eval_score(env) if mode == "eval" else self._train_score(env)
            return 0.0, bool(score > self.SUCCESS_SCORE_THRESHOLD)
        return 0.0, False

    # ── CSV 日志 ──────────────────────────────────────────────────────────────

    def reward_log_columns(self) -> Tuple[str, ...]:
        return (
            "timestep", "stage", "reward_total",
            "reward_pos", "reward_near_peak", "reward_cm_refine",
            "reward_vel_match",
            "reward_yaw", "reward_yaw_rate", "reward_action",
            "metric_dist", "metric_horiz_err", "metric_vert_err",
            "metric_speed", "metric_rel_speed", "metric_plat_speed",
            "metric_gate_xy",
            "metric_motion_type",
        )

    def reward_log_row(self, info: dict) -> dict:
        return {
            "stage": int(info.get("stage", self.stage_id())),
            "reward_total": round(info.get("reward/total", 0.0), 5),
            "reward_pos": round(info.get("reward/pos", 0.0), 5),
            "reward_near_peak": round(info.get("reward/near_peak", 0.0), 5),
            "reward_cm_refine": round(info.get("reward/cm_refine", 0.0), 5),
            "reward_vel_match": round(info.get("reward/vel_match", 0.0), 5),
            "reward_yaw": round(info.get("reward/yaw", 0.0), 5),
            "reward_yaw_rate": round(info.get("reward/yaw_rate", 0.0), 5),
            "reward_action": round(info.get("reward/action", 0.0), 5),
            "metric_dist": round(info.get("metric/dist", 0.0), 4),
            "metric_horiz_err": round(info.get("metric/horiz_err", 0.0), 4),
            "metric_vert_err": round(info.get("metric/vert_err", 0.0), 4),
            "metric_speed": round(info.get("metric/speed", 0.0), 4),
            "metric_rel_speed": round(info.get("metric/rel_speed", 0.0), 4),
            "metric_plat_speed": round(info.get("metric/plat_speed", 0.0), 4),
            "metric_gate_xy": round(info.get("metric/gate_xy", 0.0), 4),
            "metric_motion_type": str(info.get("metric/motion_type", "")),
        }
