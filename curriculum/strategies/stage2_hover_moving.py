# 功能：课程二——移动平台上方悬停的策略和奖励函数。
#
# 核心差异 vs 课程一：
#   - 相对速度替代绝对速度
#   - 移动平台 + 速度控制器 + 轻量观测延迟/误差
#   - 3 种运动模式池随机选择
#
# 奖励设计：
#   - 三层位置函数：反二次（长尾）+ 高斯（中距精度）+ 宽高斯（近距连续梯度）
#   - 速度匹配惩罚（0.15m-0.50m smoothstep 门控）：惩罚 |v_drone - v_platform|²
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
    TRAIN_SCORE_VERT_TOL = 0.20     # train_score 垂直容差 (m)
    TRAIN_SCORE_REL_SPEED_MAX = 0.18  # train_score 最大相对速度 (m/s)

    EVAL_SCORE_RADIUS = 0.10        # eval_score 水平半径 (m)
    EVAL_SCORE_VERT_TOL = 0.05      # eval_score 垂直容差 (m)
    EVAL_SCORE_REL_SPEED_MAX = 0.10  # eval_score 最大相对速度 (m/s)

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
        cfg = self.config.episode
        height = float(rng.uniform(cfg.hover_height_min, cfg.hover_height_max))
        self.set_hover_height(height)
        env._current_hover_height = height

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
        # hold/refund 已移除——由宽 sigma peak 层提供近距连续梯度

    # ── 保持判定 / 成功辅助 ───────────────────────────────────────────────────

    @staticmethod
    def _smoothstep(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

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
        rel_vel = drone_state["velocity"] - platform_state["velocity"]
        rel_speed = float(np.linalg.norm(rel_vel))
        return (
            horiz_err < self.EVAL_SCORE_RADIUS
            and vert_err < self.EVAL_SCORE_VERT_TOL
            and rel_speed < self.EVAL_SCORE_REL_SPEED_MAX
        )

    @staticmethod
    def _state_in_box(
        *,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
        target_pos: np.ndarray,
        horiz_tol: float,
        vert_tol: float,
        rel_speed_max: float,
    ) -> bool:
        target_vec = target_pos - drone_state["position"]
        horiz_err = float(np.linalg.norm(target_vec[:2]))
        vert_err = abs(float(target_vec[2]))
        rel_vel = drone_state["velocity"] - platform_state["velocity"]
        rel_speed = float(np.linalg.norm(rel_vel))
        return horiz_err < horiz_tol and vert_err < vert_tol and rel_speed < rel_speed_max

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
            rel_speed_max=self.TRAIN_SCORE_REL_SPEED_MAX,
        ):
            env._stage2_train_hit_steps = int(
                getattr(env, "_stage2_train_hit_steps", 0)
            ) + 1

        if self.is_hold_stable(
            drone_state=drone_state,
            platform_state=platform_state,
            target_pos=target_pos,
        ):
            env._stage2_eval_hold_steps = int(
                getattr(env, "_stage2_eval_hold_steps", 0)
            ) + 1
        else:
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
    #   1. 三层位置函数：反二次（长尾）+ 高斯（中距精度）+ 宽高斯（近距连续梯度）
    #   2. 速度匹配惩罚：0.15m-0.50m smoothstep 门控，惩罚 |v_drone - v_platform|²
    #   3. yaw / yaw_rate / action 平滑惩罚保留

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

        # ── 位置奖励：三层叠加 ──
        # 反二次分量：长尾，提供远距离梯度
        POS_APPROACH_WEIGHT = 3.0
        POS_APPROACH_SIGMA_XY = 8.0     # 远距离水平 σ (m)
        POS_APPROACH_SIGMA_Z = 5.0      # 远距离垂直 σ (m)
        r_approach = POS_APPROACH_WEIGHT / (
            1.0
            + (horiz_err / POS_APPROACH_SIGMA_XY) ** 2
            + (vert_err / POS_APPROACH_SIGMA_Z) ** 2
        )

        # 高斯分量：中距精度
        POS_PRECISE_WEIGHT = 3.0        # 近距精度权重
        POS_PRECISE_SIGMA_XY = 1.0      # 近距水平 σ (m)
        POS_PRECISE_SIGMA_Z = 0.5       # 近距垂直 σ (m)
        r_precise = POS_PRECISE_WEIGHT * float(np.exp(
            -(horiz_err ** 2) / (2.0 * POS_PRECISE_SIGMA_XY ** 2)
            - (vert_err ** 2) / (2.0 * POS_PRECISE_SIGMA_Z ** 2)
        ))

        # 高斯分量：窄近距峰值——提供 <0.15m 精确梯度
        POS_PEAK_WEIGHT = 2.0           # 峰值权重
        POS_PEAK_SIGMA_XY = 0.15        # 峰值水平 σ (m)
        POS_PEAK_SIGMA_Z = 0.10         # 峰值垂直 σ (m)
        r_peak = POS_PEAK_WEIGHT * float(np.exp(
            -(horiz_err ** 2) / (2.0 * POS_PEAK_SIGMA_XY ** 2)
            - (vert_err ** 2) / (2.0 * POS_PEAK_SIGMA_Z ** 2)
        ))

        # 计算位置奖励总和
        r_pos = r_approach + r_precise + r_peak

        # ── 速度匹配惩罚：近距生效 ──
        # gate: 0.15m 全开 → 0.50m 全关（反推值）
        VEL_MATCH_GATE_INNER = 0.15      # 全开水平误差阈值 (m)
        VEL_MATCH_GATE_OUTER = 0.50      # 全关水平误差阈值 (m)
        VEL_MATCH_WEIGHT = 1.0           # 速度匹配惩罚权重

        gate_vm = self._smoothstep(
            (VEL_MATCH_GATE_OUTER - horiz_err)
            / (VEL_MATCH_GATE_OUTER - VEL_MATCH_GATE_INNER)
        )

        r_vel_match = -VEL_MATCH_WEIGHT * gate_vm * (rel_speed ** 2)

        # ── 偏航角惩罚（gate_vm 门控：靠近时才关心偏航）──
        YAW_WEIGHT = 0.20
        yaw = float(drone_state["euler"][2])
        r_yaw = -YAW_WEIGHT * gate_vm * abs(yaw)

        # ── 偏航角速度惩罚（gate_vm 门控：靠近时才关心角速度）──
        YAW_RATE_WEIGHT = 0.05
        yaw_rate = float(drone_state["yaw_rate"])
        r_yaw_rate = -YAW_RATE_WEIGHT * gate_vm * (yaw_rate ** 2)

        # ── 动作平滑和幅值惩罚 ──
        ACTION_SMOOTH_WEIGHT = 0.10   # 动作变化惩罚权重
        ACTION_MAG_WEIGHT = 0.01      # 动作幅值惩罚权重
        action_delta = action - prev_action
        r_action = (
            -ACTION_SMOOTH_WEIGHT * float(np.sum(action_delta ** 2))
            -ACTION_MAG_WEIGHT * float(np.sum(action ** 2))
        )

        # 计算总奖励
        total = (
            r_pos + r_vel_match + r_yaw + r_yaw_rate
            + r_action
        )

        info = {
            "reward/pos": r_pos,
            "reward/peak": r_peak,
            "reward/vel_match": r_vel_match,
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
            "metric/gate_vm": gate_vm,
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
            "reward_pos", "reward_peak", "reward_vel_match",
            "reward_yaw", "reward_yaw_rate", "reward_action",
            "metric_dist", "metric_horiz_err", "metric_vert_err",
            "metric_speed", "metric_rel_speed", "metric_plat_speed",
            "metric_gate_vm",
            "metric_motion_type",
        )

    def reward_log_row(self, info: dict) -> dict:
        return {
            "stage": int(info.get("stage", self.stage_id())),
            "reward_total": round(info.get("reward/total", 0.0), 5),
            "reward_pos": round(info.get("reward/pos", 0.0), 5),
            "reward_peak": round(info.get("reward/peak", 0.0), 5),
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
            "metric_gate_vm": round(info.get("metric/gate_vm", 0.0), 4),
            "metric_motion_type": str(info.get("metric/motion_type", "")),
        }
