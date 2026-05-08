# 功能：阶段一静态平台悬停策略和奖励函数。

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from curriculum.strategies.base_strategy import CurriculumStrategy, HoverStrategyMixin


class Stage1HoverStaticStrategy(HoverStrategyMixin, CurriculumStrategy):
    label = "Hover  | Static Platform"
    short_label = "Hover-Static"
    episode_metrics = ("train_score", "eval_score")

    # ── 多方法共享参数 ──────────────────────────────────────────────────────
    HOLD_RADIUS = 0.10              # 保持判定水平半径 (m)
    HOLD_VERT_TOL = 0.08            # 保持判定垂直容差 (m)
    HOLD_SPEED_MAX = 0.10           # 保持判定最大速度 (m/s)
    TRAIN_SUCCESS_RADIUS = 0.20     # 训练成功水平半径 (m)
    TRAIN_SUCCESS_VERT_TOL = 0.15   # 训练成功垂直容差 (m)
    TRAIN_SUCCESS_SPEED_MAX = 0.15  # 训练成功最大速度 (m/s)
    SUCCESS_SCORE_THRESHOLD = 60.0  # 成功分数阈值 (0-100)
    OOB_PENALTY = -100.0            # 越界终止惩罚

    def stage_id(self) -> int:
        return 1

    def setup_scene(self, env, rng: np.random.Generator) -> None:
        from envs.landing_platform.motions.static_motion import StaticMotion
        self._setup_hover_scene(env, rng, motion_strategy=StaticMotion())

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
        speed = float(np.linalg.norm(drone_state["velocity"]))
        return (
            horiz_err < self.HOLD_RADIUS
            and vert_err < self.HOLD_VERT_TOL
            and speed < self.HOLD_SPEED_MAX
        )

    @staticmethod
    def _state_in_box(
        *,
        drone_state: Dict[str, np.ndarray],
        target_pos: np.ndarray,
        horiz_tol: float,
        vert_tol: float,
        speed_max: float,
    ) -> bool:
        target_vec = target_pos - drone_state["position"]
        horiz_err = float(np.linalg.norm(target_vec[:2]))
        vert_err = abs(float(target_vec[2]))
        speed = float(np.linalg.norm(drone_state["velocity"]))
        return horiz_err < horiz_tol and vert_err < vert_tol and speed < speed_max

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

    @staticmethod
    def _score(numer_steps: int, possible_steps: int) -> float:
        denom = max(int(possible_steps), 1)
        score = 100.0 * float(numer_steps) / float(denom)
        return float(np.clip(score, 0.0, 100.0))

    def reset_episode_metrics(
        self,
        env: Any,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
    ) -> None:
        env._stage1_train_hit_steps = 0
        env._stage1_eval_hold_steps = 0
        env._stage1_eval_max_hold_steps = 0
        env._stage1_train_possible_steps = self._possible_steps(
            env,
            drone_state,
            platform_state,
            horiz_tol=self.TRAIN_SUCCESS_RADIUS,
            vert_tol=self.TRAIN_SUCCESS_VERT_TOL,
        )
        env._stage1_eval_possible_steps = self._possible_steps(
            env,
            drone_state,
            platform_state,
            horiz_tol=self.HOLD_RADIUS,
            vert_tol=self.HOLD_VERT_TOL,
        )

    def update_step_metrics(
        self,
        env: Any,
        drone_state: Dict[str, np.ndarray],
        platform_state: Dict[str, np.ndarray],
        target_pos: np.ndarray,
    ) -> None:
        if self._state_in_box(
            drone_state=drone_state,
            target_pos=target_pos,
            horiz_tol=self.TRAIN_SUCCESS_RADIUS,
            vert_tol=self.TRAIN_SUCCESS_VERT_TOL,
            speed_max=self.TRAIN_SUCCESS_SPEED_MAX,
        ):
            env._stage1_train_hit_steps = int(
                getattr(env, "_stage1_train_hit_steps", 0)
            ) + 1

        if self.is_hold_stable(
            drone_state=drone_state,
            platform_state=platform_state,
            target_pos=target_pos,
        ):
            env._stage1_eval_hold_steps = int(
                getattr(env, "_stage1_eval_hold_steps", 0)
            ) + 1
        else:
            env._stage1_eval_hold_steps = 0
        env._stage1_eval_max_hold_steps = max(
            int(getattr(env, "_stage1_eval_max_hold_steps", 0)),
            int(getattr(env, "_stage1_eval_hold_steps", 0)),
        )

    def _train_score(self, env: Any) -> float:
        return self._score(
            int(getattr(env, "_stage1_train_hit_steps", 0)),
            int(getattr(env, "_stage1_train_possible_steps", 1)),
        )

    def _eval_score(self, env: Any) -> float:
        return self._score(
            int(getattr(env, "_stage1_eval_max_hold_steps", 0)),
            int(getattr(env, "_stage1_eval_possible_steps", 1)),
        )

    def get_episode_metrics(self, env: Any) -> dict:
        return {
            "train_score": float(self._train_score(env)),
            "eval_score": float(self._eval_score(env)),
        }

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
        drone_vel = drone_state["velocity"]  # 实际速度，包含扰动和跟踪误差。

        target_vec = target_pos - drone_pos
        horiz_err = float(np.linalg.norm(target_vec[:2]))
        vert_err = abs(float(target_vec[2]))
        dist = float(np.linalg.norm(target_vec))
        speed = float(np.linalg.norm(drone_vel))

        # ── 位置奖励：高斯接近奖励 ──
        # σ 越小奖励越集中，驱动更精确的定位
        POS_XY_WEIGHT = 3.0    # XY 位置奖励权重
        POS_XY_SIGMA = 0.25    # XY 高斯 σ (m)，0.25m 处降至 ~14%
        POS_Z_WEIGHT = 3.5     # Z  位置奖励权重
        POS_Z_SIGMA = 0.20     # Z  高斯 σ (m)
        r_pos = (
            POS_XY_WEIGHT
            * float(np.exp(-(horiz_err ** 2) / (2.0 * POS_XY_SIGMA ** 2)))
            + POS_Z_WEIGHT
            * float(np.exp(-(vert_err ** 2) / (2.0 * POS_Z_SIGMA ** 2)))
        )

        # ── 速度惩罚：vel_gate 控制强度，远距=0 近距=1 ──
        VEL_WEIGHT = 0.20
        VEL_GATE_XY_INNER = 0.25   # 全惩罚内阈值 (m)
        VEL_GATE_XY_OUTER = 0.75   # 零惩罚外阈值 (m)
        VEL_GATE_Z_INNER = 0.20
        VEL_GATE_Z_OUTER = 0.60
        g_xy = self._smoothstep(
            (VEL_GATE_XY_OUTER - horiz_err)
            / (VEL_GATE_XY_OUTER - VEL_GATE_XY_INNER)
        )
        g_z = self._smoothstep(
            (VEL_GATE_Z_OUTER - vert_err)
            / (VEL_GATE_Z_OUTER - VEL_GATE_Z_INNER)
        )
        vel_gate = float(g_xy * g_z)
        r_vel = -VEL_WEIGHT * vel_gate * float(np.dot(drone_vel, drone_vel))

        # ── 偏航角惩罚 ──
        YAW_WEIGHT = 0.20
        yaw = float(drone_state["euler"][2])
        r_yaw = -YAW_WEIGHT * abs(yaw)

        # ── 接近奖励：速度方向对准目标方向 ──
        VELOCITY_TOWARD_WEIGHT = 1.0           # 权重
        VELOCITY_TOWARD_DISTANCE_SCALE = 5.0   # 距离衰减半衰点 (m)
        VELOCITY_TOWARD_MIN_SPEED = 0.03       # 最低有效速度 (m/s)
        VELOCITY_TOWARD_FULL_SPEED = 0.10      # 全权重速度 (m/s)
        VELOCITY_TOWARD_MIN_DIST = 0.05        # 最低有效距离 (m)
        r_velocity_toward = 0.0
        velocity_toward_cos = 0.0
        velocity_toward_weight = 0.0
        if dist >= VELOCITY_TOWARD_MIN_DIST and speed >= VELOCITY_TOWARD_MIN_SPEED:
            velocity_toward_cos = float(np.clip(
                np.dot(drone_vel, target_vec) / (speed * dist + 1e-9),
                -1.0,
                1.0,
            ))
            distance_weight = 1.0 / (
                1.0 + (dist / VELOCITY_TOWARD_DISTANCE_SCALE) ** 2
            )
            speed_den = max(
                VELOCITY_TOWARD_FULL_SPEED - VELOCITY_TOWARD_MIN_SPEED,
                1e-6,
            )
            speed_weight = self._smoothstep(
                (speed - VELOCITY_TOWARD_MIN_SPEED) / speed_den
            )
            velocity_toward_weight = float(distance_weight * speed_weight)
            r_velocity_toward = (
                VELOCITY_TOWARD_WEIGHT
                * velocity_toward_weight
                * velocity_toward_cos
            )

        # ── 偏航角速度惩罚 ──
        YAW_RATE_WEIGHT = 0.05
        yaw_inner = 0.10
        yaw_outer = 0.50
        yaw_gate = self._smoothstep((yaw_outer - abs(yaw)) / (yaw_outer - yaw_inner))
        yaw_rate_gate = float(vel_gate * yaw_gate)
        yaw_rate = float(drone_state["yaw_rate"])
        r_yaw_rate = -YAW_RATE_WEIGHT * yaw_rate_gate * yaw_rate ** 2

        # ── 动作平滑和幅值惩罚 ──
        ACTION_SMOOTH_WEIGHT = 0.10   # 动作变化惩罚权重
        ACTION_MAG_WEIGHT = 0.01      # 动作幅值惩罚权重
        action_delta = action - prev_action
        r_action = (
            -ACTION_SMOOTH_WEIGHT * float(np.sum(action_delta ** 2))
            -ACTION_MAG_WEIGHT * float(np.sum(action ** 2))
        )

        # ── 保持奖励 + 退款机制 ──
        HOLD_BASE_REWARD = 0.05   # 每步保持奖励基数
        HOLD_CAP_STEPS = 500      # 累积奖励上限步数
        r_hold = 0.0
        r_hold_break = 0.0
        if env is not None:
            env._stage1_r_hold_total = float(
                getattr(env, "_stage1_r_hold_total", 0.0)
            )
            env._stage1_hold_refund_steps = int(
                getattr(env, "_stage1_hold_refund_steps", 0)
            )
            env._stage1_is_refunding_hold = bool(
                getattr(env, "_stage1_is_refunding_hold", False)
            )

        if hold_steps > 0:
            r_hold = HOLD_BASE_REWARD * min(int(hold_steps), HOLD_CAP_STEPS)
            if env is not None:
                env._stage1_r_hold_total += float(r_hold)
                env._stage1_hold_refund_steps = 0
                env._stage1_is_refunding_hold = False
        elif env is not None:
            if prev_hold_steps > 0:
                env._stage1_is_refunding_hold = True
                env._stage1_hold_refund_steps = 0
                if prev_hold_steps >= HOLD_CAP_STEPS:
                    r_hold_break -= 10.0

            if env._stage1_is_refunding_hold and env._stage1_r_hold_total > 0.0:
                env._stage1_hold_refund_steps += 1
                refund_rate = HOLD_BASE_REWARD * min(
                    int(env._stage1_hold_refund_steps),
                    HOLD_CAP_STEPS,
                )
                refund = min(float(env._stage1_r_hold_total), float(refund_rate))
                r_hold_break -= refund
                env._stage1_r_hold_total -= refund

                if env._stage1_r_hold_total <= 1e-9:
                    env._stage1_r_hold_total = 0.0
                    env._stage1_hold_refund_steps = 0
                    env._stage1_is_refunding_hold = False

        total = (
            r_pos + r_vel + r_yaw + r_velocity_toward
            + r_yaw_rate + r_action + r_hold + r_hold_break
        )

        info = {
            "reward/pos": r_pos,
            "reward/vel": r_vel,
            "reward/yaw": r_yaw,
            "reward/velocity_toward": r_velocity_toward,
            "reward/yaw_rate": r_yaw_rate,
            "reward/action": r_action,
            "reward/hold": r_hold,
            "reward/hold_break": r_hold_break,
            "reward/total": total,
            "metric/dist": dist,
            "metric/horiz_err": horiz_err,
            "metric/vert_err": vert_err,
            "metric/speed": speed,
            "metric/r_hold_total": float(getattr(env, "_stage1_r_hold_total", 0.0)),
            "metric/velocity_toward_cos": velocity_toward_cos,
            "metric/velocity_toward_weight": velocity_toward_weight,
            "metric/vel_gate": vel_gate,
            "metric/yaw_rate_gate": yaw_rate_gate,
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
        if term_info.get("termination") in {"oob", "below_ground"}:
            return self.OOB_PENALTY, False
        if truncated:
            mode = env.get_success_mode() if env is not None else "train"
            score = self._eval_score(env) if mode == "eval" else self._train_score(env)
            return 0.0, bool(score > self.SUCCESS_SCORE_THRESHOLD)
        return 0.0, False

    def reward_log_columns(self) -> Tuple[str, ...]:
        return (
            "timestep", "stage", "reward_total",
            "reward_pos", "reward_vel", "reward_velocity_toward",
            "reward_hold", "reward_hold_break",
            "metric_dist", "metric_horiz_err", "metric_vert_err", "metric_speed",
        )

    def reward_log_row(self, info: dict) -> dict:
        return {
            "stage": int(info.get("stage", self.stage_id())),
            "reward_total": round(info.get("reward/total", 0.0), 5),
            "reward_pos": round(info.get("reward/pos", 0.0), 5),
            "reward_vel": round(info.get("reward/vel", 0.0), 5),
            "reward_velocity_toward": round(
                info.get("reward/velocity_toward", 0.0), 5
            ),
            "reward_hold": round(info.get("reward/hold", 0.0), 5),
            "reward_hold_break": round(info.get("reward/hold_break", 0.0), 5),
            "metric_dist": round(info.get("metric/dist", 0.0), 4),
            "metric_horiz_err": round(info.get("metric/horiz_err", 0.0), 4),
            "metric_vert_err": round(info.get("metric/vert_err", 0.0), 4),
            "metric_speed": round(info.get("metric/speed", 0.0), 4),
        }
