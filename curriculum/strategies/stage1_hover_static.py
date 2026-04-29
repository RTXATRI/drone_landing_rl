# 功能：阶段一静态平台悬停策略和奖励函数。

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from curriculum.strategies.base_strategy import CurriculumStrategy, HoverStrategyMixin


class Stage1HoverStaticStrategy(HoverStrategyMixin, CurriculumStrategy):
    label = "Hover  | Static Platform"
    short_label = "Hover-Static"

    # 位置奖励：水平和高度分离的高斯接近奖励。
    POS_XY_WEIGHT = 3.0
    POS_XY_SIGMA = 0.25
    POS_Z_WEIGHT = 3.5
    POS_Z_SIGMA = 0.20

    # 速度惩罚：只在目标附近逐渐打开，避免远距离接近阶段被压制。
    VEL_WEIGHT = 0.20
    VEL_GATE_XY_INNER = 0.25
    VEL_GATE_XY_OUTER = 0.75
    VEL_GATE_Z_INNER = 0.20
    VEL_GATE_Z_OUTER = 0.60

    # 姿态和动作平滑。
    YAW_WEIGHT = 0.20
    YAW_RATE_WEIGHT = 0.05
    ACTION_SMOOTH_WEIGHT = 0.10
    ACTION_MAG_WEIGHT = 0.01

    # 实际速度朝向目标的 shaping。
    VELOCITY_TOWARD_WEIGHT = 1.0
    VELOCITY_TOWARD_DISTANCE_SCALE = 5.0
    VELOCITY_TOWARD_MIN_SPEED = 0.03
    VELOCITY_TOWARD_FULL_SPEED = 0.10
    VELOCITY_TOWARD_MIN_DIST = 0.05

    # 稳定悬停保持奖励和保持区间。
    HOLD_BASE_REWARD = 0.05
    HOLD_CAP_STEPS = 500
    HOLD_RADIUS = 0.10
    HOLD_VERT_TOL = 0.08
    HOLD_SPEED_MAX = 0.10

    # 终止塑形。max_steps 截断不再代表成功。
    OOB_PENALTY = -100.0

    def stage_id(self) -> int:
        return 1

    def setup_scene(self, env, rng: np.random.Generator) -> None:
        self._setup_hover_scene(env, rng, motion="static")

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

        r_pos = (
            self.POS_XY_WEIGHT
            * float(np.exp(-(horiz_err ** 2) / (2.0 * self.POS_XY_SIGMA ** 2)))
            + self.POS_Z_WEIGHT
            * float(np.exp(-(vert_err ** 2) / (2.0 * self.POS_Z_SIGMA ** 2)))
        )

        g_xy = self._smoothstep(
            (self.VEL_GATE_XY_OUTER - horiz_err)
            / (self.VEL_GATE_XY_OUTER - self.VEL_GATE_XY_INNER)
        )
        g_z = self._smoothstep(
            (self.VEL_GATE_Z_OUTER - vert_err)
            / (self.VEL_GATE_Z_OUTER - self.VEL_GATE_Z_INNER)
        )
        vel_gate = float(g_xy * g_z)
        r_vel = -self.VEL_WEIGHT * vel_gate * float(np.dot(drone_vel, drone_vel))

        yaw = float(drone_state["euler"][2])
        r_yaw = -self.YAW_WEIGHT * abs(yaw)

        r_velocity_toward = 0.0
        velocity_toward_cos = 0.0
        velocity_toward_weight = 0.0
        if dist >= self.VELOCITY_TOWARD_MIN_DIST and speed >= self.VELOCITY_TOWARD_MIN_SPEED:
            velocity_toward_cos = float(np.clip(
                np.dot(drone_vel, target_vec) / (speed * dist + 1e-9),
                -1.0,
                1.0,
            ))
            distance_weight = 1.0 / (
                1.0 + (dist / self.VELOCITY_TOWARD_DISTANCE_SCALE) ** 2
            )
            speed_den = max(
                self.VELOCITY_TOWARD_FULL_SPEED - self.VELOCITY_TOWARD_MIN_SPEED,
                1e-6,
            )
            speed_weight = self._smoothstep(
                (speed - self.VELOCITY_TOWARD_MIN_SPEED) / speed_den
            )
            velocity_toward_weight = float(distance_weight * speed_weight)
            r_velocity_toward = (
                self.VELOCITY_TOWARD_WEIGHT
                * velocity_toward_weight
                * velocity_toward_cos
            )

        yaw_inner = 0.10
        yaw_outer = 0.50
        yaw_gate = self._smoothstep((yaw_outer - abs(yaw)) / (yaw_outer - yaw_inner))
        yaw_rate_gate = float(vel_gate * yaw_gate)
        yaw_rate = float(drone_state["yaw_rate"])
        r_yaw_rate = -self.YAW_RATE_WEIGHT * yaw_rate_gate * yaw_rate ** 2

        action_delta = action - prev_action
        r_action = (
            -self.ACTION_SMOOTH_WEIGHT * float(np.sum(action_delta ** 2))
            -self.ACTION_MAG_WEIGHT * float(np.sum(action ** 2))
        )

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
            r_hold = self.HOLD_BASE_REWARD * min(int(hold_steps), self.HOLD_CAP_STEPS)
            if env is not None:
                env._stage1_r_hold_total += float(r_hold)
                env._stage1_hold_refund_steps = 0
                env._stage1_is_refunding_hold = False
        elif env is not None:
            if prev_hold_steps > 0:
                env._stage1_is_refunding_hold = True
                env._stage1_hold_refund_steps = 0
                if prev_hold_steps >= self.HOLD_CAP_STEPS:
                    r_hold_break -= 10.0

            if env._stage1_is_refunding_hold and env._stage1_r_hold_total > 0.0:
                env._stage1_hold_refund_steps += 1
                refund_rate = self.HOLD_BASE_REWARD * min(
                    int(env._stage1_hold_refund_steps),
                    self.HOLD_CAP_STEPS,
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
