#!/usr/bin/env python3
# 功能：运行环境 API、观测、奖励、终止条件和性能的基础自检。
"""
环境 sanity check 和单元测试。

训练前运行，用于验证：
  1. 观测/动作空间 shape 正确
  2. 满足 Gymnasium API 约定（reset/step 返回正确 tuple）
  3. 奖励分项均为有限值
  4. 课程阶段切换在 4 个阶段均可工作
  5. 成功和 OOB 时 episode 能正确终止
  6. 运动学模型滤波器能产生平滑输出

用法：
    python scripts/test_env.py
    python scripts/test_env.py --verbose  # 打印逐步 obs
    python scripts/test_env.py --render   # 显示 PyBullet GUI
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.env_config import EnvConfig
from curriculum.curriculum_manager import CurriculumManager
from curriculum.strategies import create_strategy
from envs.drone_landing_env import DroneLandingEnv
from envs.landing_platform.moving_platform import MovingPlatform
from envs.landing_platform.motions import LissajousMotion, PatrolMotion, WaypointMotion
from scripts.evaluate import apply_eval_control_config
from training.trainer import EPISODE_INFO_KEYWORDS

PASS = "  [PASS]"
FAIL = "  [FAIL]"
OBS_DIM = 34
STAGE1_EPISODE_METRIC_KEYS = ("train_score", "eval_score")
STAGE2_SPEED_LIMIT_TOL = 5.0


def _make_env(cfg: EnvConfig, stage: int = 1, render_mode=None) -> DroneLandingEnv:
    return DroneLandingEnv(
        cfg,
        strategy=create_strategy(stage, cfg),
        render_mode=render_mode,
    )


def _header(title: str) -> None:
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def _check(cond: bool, msg: str) -> bool:
    status = PASS if cond else FAIL
    print(f"{status}  {msg}")
    return cond


# ─────────────────────────────────────────────────────────────────────────────

def test_spaces(env: DroneLandingEnv) -> bool:
    _header("Test 1: Space Shapes & Types")
    ok = True
    obs, info = env.reset(seed=0)

    ok &= _check(env.observation_space.shape == (OBS_DIM,),
                 f"obs_space.shape == ({OBS_DIM},)  [got {env.observation_space.shape}]")
    ok &= _check(env.action_space.shape == (4,),
                 f"act_space.shape == (4,)   [got {env.action_space.shape}]")
    ok &= _check(obs.shape == (OBS_DIM,),
                 f"reset() obs.shape == ({OBS_DIM},)  [got {obs.shape}]")
    ok &= _check(obs.dtype == np.float32,
                 f"obs.dtype == float32  [got {obs.dtype}]")
    ok &= _check(isinstance(info, dict),
                 f"reset() info is dict")

    # 使用零动作 step
    zero = np.zeros(4, dtype=np.float32)
    obs2, r, term, trunc, info2 = env.step(zero)
    ok &= _check(obs2.shape == (OBS_DIM,),     f"step() obs.shape == ({OBS_DIM},)")
    ok &= _check(np.isfinite(r),          f"step() reward is finite  [got {r:.4f}]")
    ok &= _check(isinstance(term, bool),  "step() terminated is bool")
    ok &= _check(isinstance(trunc, bool), "step() truncated is bool")
    ok &= _check(isinstance(info2, dict), "step() info is dict")
    return ok


def test_observation_normalization(env: DroneLandingEnv) -> bool:
    _header("Test 2: Observation Normalization")
    ok = True
    env.reset(seed=1)
    for _ in range(50):
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            env.reset(seed=1)

    # 预热后，大多数 obs 值应位于 [-2, 2]
    clip_ratio = np.mean(np.abs(obs) < 2.0)
    ok &= _check(clip_ratio >= 0.80,
                 f"≥80% of obs dims in (-2, 2)  [got {clip_ratio:.0%}]")
    ok &= _check(np.all(np.isfinite(obs)),
                 "All obs values are finite")
    return ok


def test_reward_components(env: DroneLandingEnv) -> bool:
    _header("Test 3: Reward Components")
    ok = True
    env.reset(seed=2)

    rewards = []
    for _ in range(100):
        action = env.action_space.sample()
        _, r, term, trunc, info = env.step(action)
        rewards.append(r)
        if term or trunc:
            env.reset(seed=2)

    ok &= _check(all(np.isfinite(rewards)), "All rewards are finite")
    ok &= _check(len(rewards) == 100,       "100 steps collected")
    ok &= _check(np.mean(rewards) > -1000,  f"Mean reward sanity  [got {np.mean(rewards):.2f}]")
    ok &= _check("reward/total" in info,    "Info contains reward/total")
    ok &= _check("metric/dist"  in info,    "Info contains metric/dist")
    return ok


def test_curriculum_stages(env_config: EnvConfig, render_mode) -> bool:
    _header("Test 4: Curriculum Stage Switching")
    ok = True

    for stage in [1, 2, 3, 4]:
        try:
            env = _make_env(env_config, stage=stage, render_mode=render_mode)
            obs, _ = env.reset(seed=stage)
            ok &= _check(obs.shape == (OBS_DIM,), f"Stage {stage}: valid obs shape")
            ok &= _check(env.get_curriculum_stage() == stage,
                         f"Stage {stage}: get_curriculum_stage() returns {stage}")

            if stage <= 2:
                for _ in range(10):
                    action = env.action_space.sample()
                    env.step(action)
                ok &= _check(True, f"Stage {stage}: reward implemented")
            else:
                try:
                    env.step(np.zeros(4, dtype=np.float32))
                    ok &= _check(False, f"Stage {stage}: reward unexpectedly implemented")
                except NotImplementedError:
                    ok &= _check(True, f"Stage {stage}: reward is explicitly pending")

            # 在 episode 中途切换阶段
            env.set_curriculum_stage(min(stage + 1, 4))
            ok &= _check(True, f"Stage {stage}: set_curriculum_stage() OK")
            env.close()
        except Exception as e:
            _check(False, f"Stage {stage}: EXCEPTION — {e}")
            traceback.print_exc()
            ok = False

    return ok


def test_termination_conditions(env: DroneLandingEnv, env_config: EnvConfig) -> bool:
    _header("Test 5: Termination Conditions")
    ok = True

    # ── OOB：把无人机放到很远处 ────────────────────────────────────────────
    env.reset(seed=5)
    # 手动将无人机位置设得很远
    far_pos = np.array([999.0, 0.0, 3.0])
    env._drone_model.position = far_pos
    env._drone_state = env._drone_model.get_state()
    _, _, term, trunc, info = env.step(np.zeros(4, np.float32))
    ok &= _check(term or trunc, "OOB: episode terminates")

    # ── 低于地面 ───────────────────────────────────────────────────────────
    env.reset(seed=5)
    env._drone_model.position = np.array([0.0, 0.0, -2.0])
    env._drone_state = env._drone_model.get_state()
    _, _, term, trunc, info = env.step(np.zeros(4, np.float32))
    ok &= _check(term or trunc, "Below ground: episode terminates")

    # ── 内置降落策略成功条件：轻柔降落到平台上（使用 Stage 4 的 LandingStrategy） ──
    landing_env = _make_env(env_config, stage=4)
    landing_env.reset(seed=5)
    # 将无人机放在平台附近，速度为零
    plat_pos = landing_env._platform_state["position"]
    landing_env._drone_model.position = plat_pos + np.array([0.0, 0.0, 0.1])
    landing_env._drone_model.velocity  = np.zeros(3)
    landing_env._drone_model._filt_vel = np.zeros(3)
    landing_env._drone_state = landing_env._drone_model.get_state()

    term, info = landing_env._check_termination(
        landing_env._get_drone_state(),
        landing_env._get_platform_state(),
    )
    ok &= _check(term and info.get("success", False),
                 "Landing strategy: gentle landing termination check succeeds")
    landing_env.close()

    return ok


def test_action_smoothing(env: DroneLandingEnv) -> bool:
    _header("Test 6: Action Smoothing (Low-Pass Filter)")
    ok = True

    env.reset(seed=6)
    # 交替施加最大/最小动作，滤波后的输出应更平滑
    outputs = []
    for i in range(40):
        action = np.array([1.0, 1.0, 1.0, 1.0] if i % 2 == 0
                          else [-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        env.step(action)
        outputs.append(env._drone_model.velocity.copy())

    vx_series = [v[0] for v in outputs]
    # 平滑效果：最大范围应明显小于原始 v_xy limit * 2
    raw_range = 2 * env._current_v_xy_max
    actual_range = max(vx_series) - min(vx_series)
    ok &= _check(actual_range < raw_range,
                 f"Filter reduces velocity range: {actual_range:.3f} < {raw_range:.3f}")
    return ok


def test_stage1_platform_defaults(env: DroneLandingEnv) -> bool:
    _header("Test 7: Stage-1 Platform Defaults")
    ok = True
    env.set_curriculum_stage(1)
    obs, _ = env.reset(seed=8)
    ps = env._get_platform_state()

    ok &= _check(np.allclose(ps["velocity"], np.zeros(3)), "Platform velocity is zero")
    ok &= _check(np.allclose(ps["euler"], np.zeros(3)), "Platform attitude is zero")
    ok &= _check(np.allclose(ps["angular_rate"], np.zeros(3)), "Platform angular rate is zero")
    ok &= _check(float(ps["detection_quality"]) == 1.0, "Detection quality defaults to 1")
    ok &= _check(np.allclose(obs[17:23], np.zeros(6), atol=1e-6),
                 "Ship attitude/rates occupy zeroed obs slots")
    ok &= _check(abs(float(obs[26]) - 1.0) < 1e-6, "Detection quality obs slot is 1")
    return ok


def test_capability_randomization(env: DroneLandingEnv) -> bool:
    _header("Test 8: Capability Randomization")
    ok = True
    oc = env.config.observation
    caps = []

    for seed in range(20, 40):
        obs, _ = env.reset(seed=seed)
        caps.append((env._current_v_xy_max,
                     env._current_vz_down_max,
                     env._current_vz_up_max))
        ok &= _check(oc.v_xy_min <= env._current_v_xy_max <= oc.v_xy_max,
                     f"v_xy cap in range [{oc.v_xy_min}, {oc.v_xy_max}]")
        ok &= _check(oc.v_z_up_min <= env._current_vz_up_max <= oc.v_z_up_max,
                     f"v_z_up cap in range [{oc.v_z_up_min}, {oc.v_z_up_max}]")
        ok &= _check(oc.v_z_down_min <= env._current_vz_down_max <= env._current_vz_up_max,
                     "v_z_down cap <= v_z_up cap")
        expected = np.array([
            env._current_v_xy_max / oc.v_xy_norm,
            env._current_vz_down_max / oc.v_z_down_norm,
            env._current_vz_up_max / oc.v_z_up_norm,
        ], dtype=np.float32)
        ok &= _check(np.allclose(obs[27:30], expected, atol=1e-5),
                     "Capability obs slots match current env caps")

    unique_xy = len({round(c[0], 3) for c in caps})
    ok &= _check(unique_xy > 1, "Capabilities vary across resets")
    return ok


def test_observation_clipping(env: DroneLandingEnv) -> bool:
    _header("Test 9: Observation Clipping")
    ok = True
    env.reset(seed=9)

    drone_state = {
        "position": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "velocity": np.array([-100.0, 100.0, -100.0], dtype=np.float32),
        "euler": np.array([4.0 * np.pi, -4.0 * np.pi, 3.0 * np.pi], dtype=np.float32),
        "yaw_rate": np.float32(5.0),
    }
    platform_state = {
        "position": np.array([100.0, -100.0, 100.0], dtype=np.float32),
        "velocity": np.array([100.0, -100.0, 100.0], dtype=np.float32),
        "euler": np.array([4.0 * np.pi, -4.0 * np.pi, 3.0 * np.pi], dtype=np.float32),
        "angular_rate": np.array([5.0, -5.0, 5.0], dtype=np.float32),
        "detection_quality": np.float32(2.0),
    }
    env._current_v_xy_max = 50.0
    env._current_vz_down_max = 10.0
    env._current_vz_up_max = 10.0
    env._prev_action = np.array([2.0, -2.0, 2.0, -2.0], dtype=np.float32)

    obs = env._build_observation(drone_state, platform_state)
    oc = env.config.observation
    ok &= _check(np.all(obs[0:3] <= oc.clip_signed) and np.all(obs[0:3] >= -oc.clip_signed),
                 "Relative position clipped")
    ok &= _check(np.all(obs[3:6] <= oc.clip_signed) and np.all(obs[3:6] >= -oc.clip_signed),
                 "Relative velocity clipped")
    ok &= _check(0.0 <= obs[8] <= oc.clip_nonnegative, "Range clipped")
    ok &= _check(0.0 <= obs[16] <= oc.clip_nonnegative, "Height clipped")
    ok &= _check(0.0 <= obs[26] <= 1.0, "Detection quality clipped")
    ok &= _check(np.all(obs[27:30] <= oc.clip_capability), "Capabilities clipped")
    ok &= _check(np.all(obs[30:34] <= 1.0) and np.all(obs[30:34] >= -1.0),
                 "Prev action clipped")
    return ok


def test_action_scaling(env: DroneLandingEnv) -> bool:
    _header("Test 10: Action Scaling With Randomized Caps")
    ok = True
    env.reset(seed=10)
    env._current_v_xy_max = 12.0
    env._current_vz_up_max = 4.0
    env._current_vz_down_max = 2.5

    up = env._scale_action(np.array([1.0, -1.0, 1.0, 0.5], dtype=np.float32))
    down = env._scale_action(np.array([0.5, 0.25, -1.0, -0.5], dtype=np.float32))

    ok &= _check(np.isclose(up[0], 12.0) and np.isclose(up[1], -12.0),
                 "Horizontal action uses current v_xy cap")
    ok &= _check(np.isclose(up[2], 4.0), "Positive z action uses v_z_up cap")
    ok &= _check(np.isclose(down[2], -2.5), "Negative z action uses v_z_down cap")
    ok &= _check(np.isclose(up[3], 0.5 * env.config.drone.max_yaw_rate),
                 "Yaw action uses configured yaw-rate cap")
    return ok


def test_spawn_range_and_oob_margin(env: DroneLandingEnv) -> bool:
    _header("Test 11: Spawn Range & OOB Margin")
    ok = True
    cfg = env.config.episode
    env.set_curriculum_stage(1)

    for seed in range(30, 45):
        env.reset(seed=seed)
        ds = env._get_drone_state()
        ps = env._get_platform_state()
        horiz = float(np.linalg.norm(ds["position"][:2] - ps["position"][:2]))
        height = float(ds["position"][2])

        ok &= _check(cfg.init_spawn_radius_min <= horiz <= cfg.init_spawn_radius_max,
                     f"spawn radius in [{cfg.init_spawn_radius_min}, {cfg.init_spawn_radius_max}]")
        ok &= _check(cfg.init_spawn_height_min <= height <= cfg.init_spawn_height_max,
                     f"spawn height in [{cfg.init_spawn_height_min}, {cfg.init_spawn_height_max}]")

        _, _, term, _, info = env.step(np.zeros(4, np.float32))
        ok &= _check(not (term and info.get("termination") == "oob"),
                     "spawn does not immediately trigger OOB")

    return ok


def _reset_strategy_episode_metrics(env: DroneLandingEnv) -> None:
    env.strategy.reset_episode_metrics(
        env,
        env._get_drone_state(),
        env._get_platform_state(),
    )


def _place_drone_at_hover_target(env: DroneLandingEnv, offset=(0.0, 0.0, 0.0)) -> None:
    """将运动学无人机放到当前悬停目标点附近，并重算策略 episode 指标。"""
    platform_state = env._get_platform_state()
    target = env._get_target_pos(platform_state["position"])
    env._drone_model.position = target + np.array(offset, dtype=np.float32)
    env._drone_model.velocity = np.zeros(3)
    env._drone_model._filt_vel = np.zeros(3)
    env._drone_state = env._drone_model.get_state()
    _reset_strategy_episode_metrics(env)


def _place_drone_away_from_hover_target(env: DroneLandingEnv, offset=(1.0, 0.0, 0.0)) -> None:
    """将无人机放到悬停稳定范围之外，并重算策略 episode 指标。"""
    _place_drone_at_hover_target(env, offset=offset)


def _run_zero_action_episode(env: DroneLandingEnv) -> tuple:
    info = {}
    reward = 0.0
    term = trunc = False
    for _ in range(int(env.config.episode.max_steps)):
        _, reward, term, trunc, info = env.step(np.zeros(4, np.float32))
        if term or trunc:
            break
    return reward, term, trunc, info


def test_stage1_terminal_behavior() -> bool:
    _header("Test 12: Stage-1 Terminal Behavior")
    ok = True

    train_cfg = EnvConfig()
    train_cfg.disturbance.enabled = False
    train_cfg.episode.max_steps = 3
    train_env = _make_env(train_cfg, stage=1)
    train_env.reset(seed=48)
    _place_drone_at_hover_target(train_env, offset=(0.15, 0.0, 0.0))
    _, _, _, train_info = _run_zero_action_episode(train_env)
    train_ep = train_info.get("episode", {})
    ok &= _check(bool(train_ep.get("success", False)),
                 "train mode: wide Stage 1 success space marks success")
    ok &= _check(float(train_ep.get("train_score", 0.0)) > 60.0,
                 f"train mode: train_score > 60  [got {train_ep.get('train_score', 0.0):.1f}]")
    ok &= _check("hover_score" not in train_ep,
                 "Stage 1 episode info still does not export old hover_score")
    train_env.close()

    eval_fail_cfg = EnvConfig()
    eval_fail_cfg.disturbance.enabled = False
    eval_fail_cfg.episode.max_steps = 3
    eval_fail_env = _make_env(eval_fail_cfg, stage=1)
    eval_fail_env.set_success_mode("eval")
    eval_fail_env.reset(seed=48)
    _place_drone_at_hover_target(eval_fail_env, offset=(0.15, 0.0, 0.0))
    _, _, _, eval_fail_info = _run_zero_action_episode(eval_fail_env)
    eval_fail_ep = eval_fail_info.get("episode", {})
    ok &= _check(not bool(eval_fail_ep.get("success", False)),
                 "eval mode: same wide-only state is not a success")
    eval_fail_env.close()

    eval_success_cfg = EnvConfig()
    eval_success_cfg.disturbance.enabled = False
    eval_success_cfg.episode.max_steps = 3
    eval_success_env = _make_env(eval_success_cfg, stage=1)
    eval_success_env.set_success_mode("eval")
    eval_success_env.reset(seed=49)
    _place_drone_at_hover_target(eval_success_env)
    _, _, _, eval_success_info = _run_zero_action_episode(eval_success_env)
    eval_success_ep = eval_success_info.get("episode", {})
    ok &= _check(bool(eval_success_ep.get("success", False)),
                 "eval mode: stable hold space marks success")
    eval_success_env.close()

    oob_cfg = EnvConfig()
    oob_cfg.disturbance.enabled = False
    oob_cfg.episode.max_steps = 3
    oob_env = _make_env(oob_cfg, stage=1)
    oob_env.reset(seed=49)
    _place_drone_at_hover_target(oob_env)
    oob_env._drone_model.position = np.array([999.0, 0.0, 3.0])
    oob_env._drone_state = oob_env._drone_model.get_state()
    _, reward, term, _, oob_info = oob_env.step(np.zeros(4, np.float32))
    ok &= _check(term and oob_info.get("termination") == "oob",
                 "Stage 1 OOB terminates episode")
    ok &= _check(reward < -50.0, f"Stage 1 OOB keeps terminal penalty  [got {reward:.2f}]")
    ok &= _check(not bool(oob_info.get("episode", {}).get("success", False)),
                 "Stage 1 OOB overrides success metrics and marks failure")
    oob_env.close()

    low_cfg = EnvConfig()
    low_cfg.disturbance.enabled = False
    low_cfg.episode.max_steps = 3
    low_env = _make_env(low_cfg, stage=1)
    low_env.reset(seed=50)
    _place_drone_at_hover_target(low_env)
    low_env._drone_model.position = np.array([0.0, 0.0, -2.0])
    low_env._drone_state = low_env._drone_model.get_state()
    _, reward, term, _, low_info = low_env.step(np.zeros(4, np.float32))
    ok &= _check(term and low_info.get("termination") == "below_ground",
                 "Stage 1 below-ground terminates episode")
    ok &= _check(reward < -50.0,
                 f"Stage 1 below-ground keeps terminal penalty  [got {reward:.2f}]")
    ok &= _check(not bool(low_info.get("episode", {}).get("success", False)),
                 "Stage 1 below-ground overrides success metrics and marks failure")
    low_env.close()
    return ok


def test_vecmonitor_preserves_episode_core_fields() -> bool:
    _header("Test 13: VecMonitor Episode Core Fields")
    ok = True
    cfg = EnvConfig()
    cfg.disturbance.enabled = False
    cfg.episode.max_steps = 3

    vec_env = VecMonitor(
        DummyVecEnv([lambda: _make_env(cfg, stage=1)]),
        info_keywords=EPISODE_INFO_KEYWORDS,
    )
    vec_env.reset()
    raw_env = vec_env.venv.envs[0]
    _place_drone_at_hover_target(raw_env)

    info = {}
    done = np.array([False])
    for _ in range(3):
        _, _, done, infos = vec_env.step(np.zeros((1, 4), dtype=np.float32))
        info = infos[0]
    ep = info.get("episode", {})
    vec_env.close()

    ok &= _check(bool(done[0]), "VecMonitor receives terminal episode")
    ok &= _check(all(key in ep for key in ("r", "l", "t")),
                 "VecMonitor keeps SB3 native r/l/t episode fields")
    ok &= _check(bool(ep.get("success", False)),
                 "VecMonitor preserves custom success field")
    ok &= _check(ep.get("episode_stage", 0) == 1,
                 "VecMonitor preserves custom episode_stage field")
    ok &= _check(all(key in ep for key in STAGE1_EPISODE_METRIC_KEYS),
                 "VecMonitor preserves Stage 1 episode metric keys")
    ok &= _check(float(ep.get("train_score", 0.0)) > 60.0,
                 "VecMonitor preserves train_score")
    ok &= _check("hover_score" not in ep, "VecMonitor episode has no hover_score")
    return ok


def test_manual_curriculum_api() -> bool:
    _header("Test 15: Manual Curriculum API")
    ok = True
    ok &= _check(not hasattr(CurriculumManager, "should_advance"),
                 "CurriculumManager has no automatic should_advance API")
    ok &= _check(not hasattr(CurriculumManager, "advance_stage"),
                 "CurriculumManager has no automatic advance_stage API")
    ok &= _check(hasattr(CurriculumManager, "set_stage"),
                 "CurriculumManager exposes manual set_stage API")
    return ok


def test_disturbance_reproducibility() -> bool:
    _header("Test 16: Disturbance Reproducibility")
    ok = True

    def _collect(seed: int):
        cfg = EnvConfig()
        cfg.disturbance.gust_prob_per_second = 1000.0
        cfg.disturbance.gust_duration_min = 0.5
        cfg.disturbance.gust_duration_max = 0.5
        env = _make_env(cfg, stage=1)
        env.reset(seed=seed)
        base_wind = env._drone_model.get_disturbance_info()["base_wind_velocity"].copy()
        gust_speeds = []
        gust_z = []
        tracking_scales = []
        action = np.array([0.25, -0.15, 0.20, 0.0], dtype=np.float32)
        for _ in range(8):
            _, _, _, _, info = env.step(action)
            disturbance = env._drone_model.get_disturbance_info()
            gust_speeds.append(float(info["metric/gust_speed"]))
            gust_z.append(float(disturbance["gust_velocity"][2]))
            tracking_scales.append(disturbance["tracking_scale"].copy())
        env.close()
        return base_wind, np.array(gust_speeds), np.array(gust_z), np.array(tracking_scales)

    wind_a, gust_a, gust_z_a, scale_a = _collect(seed=101)
    wind_b, gust_b, gust_z_b, scale_b = _collect(seed=101)
    wind_c, _, _, _ = _collect(seed=102)

    ok &= _check(np.allclose(wind_a, wind_b),
                 "base wind is reproducible with fixed seed")
    ok &= _check(not np.allclose(wind_a, wind_c),
                 "base wind changes with different seeds")
    ok &= _check(abs(float(wind_a[2])) < 1e-9, "base wind has no vertical component")
    ok &= _check(np.allclose(gust_a, gust_b),
                 "gust event sequence is reproducible with fixed seed")
    ok &= _check(max(gust_a) > 0.0, "gust sequence contains a nonzero gust")
    ok &= _check(np.allclose(gust_z_a, 0.0) and np.allclose(gust_z_b, 0.0),
                 "gust wind has no vertical component")
    ok &= _check(np.allclose(scale_a, scale_b),
                 "tracking scale sequence is reproducible with fixed seed")
    return ok


def test_zero_action_wind_drift() -> bool:
    _header("Test 17: Zero Action Wind Drift")
    ok = True
    cfg = EnvConfig()
    cfg.disturbance.base_wind_speed_min = 0.05
    cfg.disturbance.base_wind_speed_max = 0.05
    cfg.disturbance.gust_prob_per_second = 0.0
    cfg.disturbance.tracking_error_enabled = False

    env = _make_env(cfg, stage=1)
    env.reset(seed=111)
    pos0 = env._get_drone_state()["position"].copy()
    for _ in range(20):
        env.step(np.zeros(4, np.float32))
    state = env._get_drone_state()
    drift_xy = float(np.linalg.norm((state["position"] - pos0)[:2]))
    env.close()

    ok &= _check(drift_xy > 0.02, f"zero action drifts horizontally  [got {drift_xy:.3f}m]")
    ok &= _check(abs(float(state["wind_velocity"][2])) < 1e-9,
                 "wind disturbance remains horizontal")
    return ok


def test_gust_activation_and_decay() -> bool:
    _header("Test 18: Gust Activation & Decay")
    ok = True
    cfg = EnvConfig()
    cfg.disturbance.base_wind_speed_min = 0.0
    cfg.disturbance.base_wind_speed_max = 0.0
    cfg.disturbance.gust_prob_per_second = 1000.0
    cfg.disturbance.gust_speed_min = 0.20
    cfg.disturbance.gust_speed_max = 0.20
    cfg.disturbance.gust_duration_min = 0.5
    cfg.disturbance.gust_duration_max = 0.5
    cfg.disturbance.gust_smooth_prob = 0.0
    cfg.disturbance.tracking_error_enabled = False

    env = _make_env(cfg, stage=1)
    env.reset(seed=121)
    speeds = []
    for _ in range(8):
        _, _, _, _, info = env.step(np.zeros(4, np.float32))
        speeds.append(float(info["metric/gust_speed"]))

    cfg.disturbance.gust_prob_per_second = 0.0
    for _ in range(20):
        _, _, _, _, info = env.step(np.zeros(4, np.float32))
    final_speed = float(info["metric/gust_speed"])
    env.close()

    ok &= _check(max(speeds) > 0.0, "gust becomes nonzero after activation")
    ok &= _check(final_speed == 0.0, "gust returns to zero after duration")
    return ok


def test_tracking_error_actual_velocity() -> bool:
    _header("Test 19: Tracking Error Actual Velocity")
    ok = True
    cfg = EnvConfig()
    cfg.disturbance.base_wind_speed_min = 0.0
    cfg.disturbance.base_wind_speed_max = 0.0
    cfg.disturbance.gust_prob_per_second = 0.0
    cfg.disturbance.tracking_error_enabled = True

    env = _make_env(cfg, stage=1)
    env.reset(seed=131)
    env.step(np.array([0.6, -0.4, 0.5, 0.0], dtype=np.float32))
    model = env._drone_model
    expected = model._filt_vel * model._tracking_scale + model._wind_vel
    env.close()

    ok &= _check(np.all(model._tracking_scale >= cfg.disturbance.tracking_scale_min),
                 "tracking scales stay above minimum")
    ok &= _check(np.all(model._tracking_scale <= cfg.disturbance.tracking_scale_max),
                 "tracking scales stay below maximum")
    ok &= _check(np.allclose(model.velocity, expected),
                 "drone velocity equals tracked command plus wind")
    return ok


def test_disturbance_disabled_zero_drift() -> bool:
    _header("Test 20: Disturbance Disabled")
    ok = True
    cfg = EnvConfig()
    cfg.disturbance.enabled = False

    env = _make_env(cfg, stage=1)
    env.reset(seed=141)
    pos0 = env._get_drone_state()["position"].copy()
    for _ in range(20):
        env.step(np.zeros(4, np.float32))
    pos1 = env._get_drone_state()["position"].copy()
    env.close()

    ok &= _check(np.allclose(pos0, pos1), "zero action does not drift when disturbance is disabled")
    return ok


def test_stage1_strategy_reward_shape() -> bool:
    _header("Test 21: Stage-1 Strategy Reward Shape")
    ok = True
    cfg = EnvConfig()
    strategy = create_strategy(1, cfg)
    target = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    platform = {
        "position": np.zeros(3, dtype=np.float32),
        "velocity": np.zeros(3, dtype=np.float32),
        "euler": np.zeros(3, dtype=np.float32),
        "angular_rate": np.zeros(3, dtype=np.float32),
    }

    def _info_at(pos, velocity=(0.0, 0.0, 0.0), action=None, prev_action=None):
        drone = {
            "position": np.array(pos, dtype=np.float32),
            "velocity": np.array(velocity, dtype=np.float32),
            "euler": np.zeros(3, dtype=np.float32),
            "yaw_rate": np.float32(0.0),
        }
        action = np.zeros(4, dtype=np.float32) if action is None else np.array(action, dtype=np.float32)
        prev_action = (
            np.zeros(4, dtype=np.float32)
            if prev_action is None
            else np.array(prev_action, dtype=np.float32)
        )
        reward, info = strategy.compute_reward(
            env=None,
            drone_state=drone,
            platform_state=platform,
            action=action,
            prev_action=prev_action,
            target_pos=target,
            hold_steps=0,
            prev_hold_steps=0,
        )
        return float(reward), info

    center_reward, center = _info_at([0.0, 0.0, 2.0])
    offset_reward, offset = _info_at([0.26, 0.0, 2.0])
    far_reward, far = _info_at([1.0, 0.0, 2.0])
    toward_reward, toward = _info_at([1.0, 0.0, 2.0], velocity=(-0.2, 0.0, 0.0))
    away_reward, away = _info_at([1.0, 0.0, 2.0], velocity=(0.2, 0.0, 0.0))
    smooth_reward, smooth = _info_at(
        [1.0, 0.0, 2.0],
        velocity=(-0.2, 0.0, 0.0),
        action=(1.0, 0.0, 0.0, 0.0),
        prev_action=(0.0, 0.0, 0.0, 0.0),
    )

    expected_keys = {
        "reward/total",
        "reward/pos",
        "reward/vel",
        "reward/velocity_toward",
        "reward/hold",
        "reward/hold_break",
        "metric/dist",
        "metric/horiz_err",
        "metric/vert_err",
        "metric/speed",
    }
    ok &= _check(np.isfinite(center_reward), "Stage 1 reward is finite")
    ok &= _check(center["reward/pos"] > offset["reward/pos"] > far["reward/pos"],
                 "position reward decays away from target")
    ok &= _check(toward["reward/velocity_toward"] > 0.0,
                 "velocity toward target gives positive reward")
    ok &= _check(away["reward/velocity_toward"] < 0.0,
                 "velocity away from target gives negative reward")
    ok &= _check(smooth_reward < toward_reward,
                 "action penalty lowers otherwise similar reward")
    ok &= _check(expected_keys.issubset(center.keys()),
                 "Stage 1 reward info exposes compact core keys")
    return ok


def test_stage2_strategy_reward_shape() -> bool:
    _header("Test 22: Stage-2 Strategy Reward Shape (Simplified)")
    ok = True
    cfg = EnvConfig()
    strategy = create_strategy(2, cfg)
    target = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    platform = {
        "position": np.zeros(3, dtype=np.float32),
        "velocity": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "euler": np.zeros(3, dtype=np.float32),
        "angular_rate": np.zeros(3, dtype=np.float32),
    }

    def _info_at(pos, velocity=(0.0, 0.0, 0.0)):
        drone = {
            "position": np.array(pos, dtype=np.float32),
            "velocity": np.array(velocity, dtype=np.float32),
            "euler": np.zeros(3, dtype=np.float32),
            "yaw_rate": np.float32(0.0),
        }
        reward, info = strategy.compute_reward(
            env=None,
            drone_state=drone,
            platform_state=platform,
            action=np.zeros(4, dtype=np.float32),
            prev_action=np.zeros(4, dtype=np.float32),
            target_pos=target,
            hold_steps=0,
            prev_hold_steps=0,
        )
        return float(reward), info

    _, center = _info_at([0.0, 0.0, 5.0])
    _, near = _info_at([0.3, 0.0, 5.0])
    _, mid = _info_at([5.0, 0.0, 5.0])
    _, far = _info_at([30.0, 0.0, 5.0])
    _, vert_far = _info_at([0.0, 0.0, 0.0])
    # 速度匹配：靠近目标 + 有相对速度时产生惩罚
    _, vm_near = _info_at([0.1, 0.0, 5.0], velocity=(0.5, 0.0, 0.0))
    _, vm_far = _info_at([3.0, 0.0, 5.0], velocity=(0.5, 0.0, 0.0))

    # 位置奖励在目标中心处最大
    ok &= _check(center["reward/pos"] >= 7.0,
                 f"Stage 2 target-center position reward is strong  [got {center['reward/pos']:.4f}]")
    # 位置奖励随距离单调衰减
    ok &= _check(center["reward/pos"] > near["reward/pos"] > mid["reward/pos"],
                 "Stage 2 position reward decays with distance from target")
    # 远距离仍有梯度
    ok &= _check(far["reward/pos"] > 0.01,
                 f"Stage 2 far-range position gradient is present  [got {far['reward/pos']:.4f}]")
    # 垂直偏离也导致衰减
    ok &= _check(vert_far["reward/pos"] < center["reward/pos"],
                 "Stage 2 vertical error reduces position reward")

    # 速度匹配：远离目标时不开门
    ok &= _check(vm_far["metric/gate_vm"] < 1e-6,
                 f"Stage 2 velocity-match gate is off when far  [got {vm_far['metric/gate_vm']:.4f}]")
    # 速度匹配：靠近时产生有意义的惩罚
    ok &= _check(vm_near["reward/vel_match"] < -0.01,
                 f"Stage 2 velocity-match penalizes rel-speed when close  [got {vm_near['reward/vel_match']:.4f}]")

    # peak 层在 <0.15m 提供非零梯度
    ok &= _check(center["reward/peak"] > near["reward/peak"],
                 f"Stage 2 peak reward provides gradient at close range")

    # 必需 info 键存在
    expected_keys = {
        "reward/pos", "reward/peak", "reward/vel_match", "reward/yaw",
        "reward/yaw_rate", "reward/action", "reward/total",
        "metric/dist", "metric/horiz_err", "metric/vert_err",
        "metric/speed", "metric/rel_speed", "metric/plat_speed",
        "metric/gate_vm",
    }
    ok &= _check(expected_keys.issubset(center.keys()),
                 "Stage 2 reward info exposes all required simplified keys")
    # 不应包含 hold/refund 键
    excluded_keys = {"reward/hold", "reward/hold_break", "metric/r_hold_total"}
    ok &= _check(excluded_keys.isdisjoint(center.keys()),
                 "Stage 2 reward info does not contain hold/refund keys")
    return ok


def test_hold_reward_space_independence() -> bool:
    _header("Test 22b: Hold Reward Space Independence")
    ok = True
    cfg = EnvConfig()

    class DummyEnv:
        pass

    def _env():
        env = DummyEnv()
        env._current_v_xy_max = 1.0
        env._current_vz_up_max = 1.0
        env._current_vz_down_max = 1.0
        return env

    def _reward_info(strategy, env, drone, platform, target, hold_steps=0):
        _, info = strategy.compute_reward(
            env=env,
            drone_state=drone,
            platform_state=platform,
            action=np.zeros(4, dtype=np.float32),
            prev_action=np.zeros(4, dtype=np.float32),
            target_pos=target,
            hold_steps=hold_steps,
            prev_hold_steps=hold_steps,
        )
        return info

    # Stage 1: reward hold space is independent from env-level hold_steps.
    s1 = create_strategy(1, cfg)
    env1 = _env()
    target1 = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    platform1 = {
        "position": np.zeros(3, dtype=np.float32),
        "velocity": np.zeros(3, dtype=np.float32),
        "euler": np.zeros(3, dtype=np.float32),
        "angular_rate": np.zeros(3, dtype=np.float32),
    }
    stable1 = {
        "position": target1.copy(),
        "velocity": np.zeros(3, dtype=np.float32),
        "euler": np.zeros(3, dtype=np.float32),
        "yaw_rate": np.float32(0.0),
    }
    train_only1 = {
        "position": np.array([0.15, 0.0, 2.0], dtype=np.float32),
        "velocity": np.array([0.12, 0.0, 0.0], dtype=np.float32),
        "euler": np.zeros(3, dtype=np.float32),
        "yaw_rate": np.float32(0.0),
    }
    s1.reset_episode_metrics(env1, stable1, platform1)
    s1_stable = _reward_info(s1, env1, stable1, platform1, target1, hold_steps=0)
    ok &= _check(np.isclose(s1_stable["reward/hold"], 0.05),
                 "Stage 1 reward hold triggers from reward space, not input hold_steps")

    s1.reset_episode_metrics(env1, stable1, platform1)
    s1_outside = _reward_info(s1, env1, train_only1, platform1, target1, hold_steps=99)
    ok &= _check(np.isclose(s1_outside["reward/hold"], 0.0),
                 "Stage 1 input hold_steps cannot force reward hold outside reward space")
    s1.update_step_metrics(env1, train_only1, platform1, target1)
    ok &= _check(env1._stage1_train_hit_steps == 1 and env1._stage1_eval_hold_steps == 0,
                 "Stage 1 train_score space can differ from eval/reward hold space")

    # Stage 2: peak 层在 <0.15m 提供非零连续梯度（替代旧 hold/refund）
    s2 = create_strategy(2, cfg)
    env2 = _env()
    target2 = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    platform2 = {
        "position": np.zeros(3, dtype=np.float32),
        "velocity": np.zeros(3, dtype=np.float32),
        "euler": np.zeros(3, dtype=np.float32),
        "angular_rate": np.zeros(3, dtype=np.float32),
    }
    at_center = {
        "position": target2.copy(),
        "velocity": np.zeros(3, dtype=np.float32),
        "euler": np.zeros(3, dtype=np.float32),
        "yaw_rate": np.float32(0.0),
    }
    at_0d1m = {
        "position": np.array([0.1, 0.0, 5.0], dtype=np.float32),
        "velocity": np.zeros(3, dtype=np.float32),
        "euler": np.zeros(3, dtype=np.float32),
        "yaw_rate": np.float32(0.0),
    }
    at_0d3m = {
        "position": np.array([0.3, 0.0, 5.0], dtype=np.float32),
        "velocity": np.zeros(3, dtype=np.float32),
        "euler": np.zeros(3, dtype=np.float32),
        "yaw_rate": np.float32(0.0),
    }
    s2.reset_episode_metrics(env2, at_center, platform2)
    s2_center = _reward_info(s2, env2, at_center, platform2, target2, hold_steps=0)
    s2_0d1m = _reward_info(s2, env2, at_0d1m, platform2, target2, hold_steps=0)
    s2_0d3m = _reward_info(s2, env2, at_0d3m, platform2, target2, hold_steps=0)
    ok &= _check(s2_center["reward/peak"] > s2_0d1m["reward/peak"] > s2_0d3m["reward/peak"],
                 "Stage 2 peak reward provides continuous gradient at close range")
    ok &= _check(s2_center["reward/peak"] > 0.5,
                 f"Stage 2 peak reward at center is significant  [got {s2_center['reward/peak']:.4f}]")
    # 不应存在 hold/refund 键
    ok &= _check("reward/hold" not in s2_center and "reward/hold_break" not in s2_center,
                 "Stage 2 reward info does not contain hold/refund keys")
    s2.update_step_metrics(env2, at_0d1m, platform2, target2)
    ok &= _check(env2._stage2_train_hit_steps == 1 and env2._stage2_eval_hold_steps == 0,
                 "Stage 2 train_score space can differ from eval hold space")
    return ok


def test_success_mode_score_consistency() -> bool:
    _header("Test 22c: Success Mode Score Consistency")
    ok = True
    cfg = EnvConfig()

    class DummyEnv:
        def __init__(self, mode="train"):
            self._mode = mode
            self._current_v_xy_max = 1.0
            self._current_vz_up_max = 1.0
            self._current_vz_down_max = 1.0

        def get_success_mode(self):
            return self._mode

    cases = (
        (
            1,
            np.array([0.0, 0.0, 2.0], dtype=np.float32),
            "_stage1_train_hit_steps",
            "_stage1_train_possible_steps",
            "_stage1_eval_max_hold_steps",
        ),
        (
            2,
            np.array([0.0, 0.0, 5.0], dtype=np.float32),
            "_stage2_train_hit_steps",
            "_stage2_train_possible_steps",
            "_stage2_eval_max_hold_steps",
        ),
    )

    platform = {
        "position": np.zeros(3, dtype=np.float32),
        "velocity": np.zeros(3, dtype=np.float32),
        "euler": np.zeros(3, dtype=np.float32),
        "angular_rate": np.zeros(3, dtype=np.float32),
    }

    for stage, target, train_hits_key, train_possible_key, eval_hits_key in cases:
        strategy = create_strategy(stage, cfg)
        env = DummyEnv()
        drone = {
            "position": target.copy(),
            "velocity": np.zeros(3, dtype=np.float32),
            "euler": np.zeros(3, dtype=np.float32),
            "yaw_rate": np.float32(0.0),
        }
        strategy.reset_episode_metrics(env, drone, platform)
        setattr(env, train_hits_key, int(getattr(env, train_possible_key)))
        setattr(env, eval_hits_key, 0)

        env._mode = "train"
        train_mode_metrics = strategy.get_episode_metrics(env)
        _, train_success = strategy.terminal_bonus(
            env=env,
            terminated=False,
            truncated=True,
            term_info={},
        )

        env._mode = "eval"
        eval_mode_metrics = strategy.get_episode_metrics(env)
        _, eval_success = strategy.terminal_bonus(
            env=env,
            terminated=False,
            truncated=True,
            term_info={},
        )

        ok &= _check(train_mode_metrics == eval_mode_metrics,
                     f"Stage {stage}: train/eval modes compute the same score values")
        ok &= _check(train_success and not eval_success,
                     f"Stage {stage}: success_mode only selects train_score or eval_score")

    return ok


def test_eval_fixed_controls_config() -> bool:
    _header("Test 23: Eval Fixed Controls")
    ok = True
    cfg = EnvConfig()
    apply_eval_control_config(
        cfg,
        hover_height=5.0,
        eval_v_xy_max=10.0,
        eval_v_z_up_max=3.0,
        eval_v_z_down_max=2.0,
    )

    env = _make_env(cfg, stage=1)
    obs, _ = env.reset(seed=151)
    ps = env._get_platform_state()
    target = env._get_target_pos(ps["position"])
    hover_height = float(target[2] - ps["position"][2])

    ok &= _check(np.isclose(hover_height, 5.0),
                 f"Stage-1 eval hover height is fixed at 5m  [got {hover_height:.2f}]")
    ok &= _check(np.isclose(env._current_v_xy_max, 10.0),
                 "eval horizontal speed cap is fixed at 10m/s")
    ok &= _check(np.isclose(env._current_vz_up_max, 3.0),
                 "eval upward speed cap is fixed at 3m/s")
    ok &= _check(np.isclose(env._current_vz_down_max, 2.0),
                 "eval downward speed cap is fixed at 2m/s")
    ok &= _check(np.allclose(obs[27:30], np.array([0.5, 0.4, 0.6], dtype=np.float32), atol=1e-5),
                 "capability observation slots match fixed eval caps")
    env.close()

    try:
        apply_eval_control_config(
            EnvConfig(),
            hover_height=5.0,
            eval_v_xy_max=10.0,
            eval_v_z_up_max=2.0,
            eval_v_z_down_max=3.0,
        )
        invalid_rejected = False
    except ValueError:
        invalid_rejected = True
    ok &= _check(invalid_rejected, "eval downward speed cannot exceed upward speed")
    return ok


def test_stage2_platform_motion_continuity() -> bool:
    _header("Test 24: Stage-2 Platform Motion Continuity")
    ok = True
    cfg = EnvConfig()
    dt = float(cfg.episode.dt)
    motion_factories = {
        "lissajous": lambda: LissajousMotion(),
        "patrol": lambda: PatrolMotion(boundary=50.0),
        "waypoint": lambda: WaypointMotion(boundary=50.0),
    }

    for name, factory in motion_factories.items():
        max_diff_speed = 0.0
        max_reported_speed = 0.0
        max_velocity_mismatch = 0.0
        for seed in range(12):
            rng = np.random.default_rng(10_000 + seed)
            platform = MovingPlatform(cfg.platform, cfg.episode.dt)
            platform.set_motion_strategy(factory())
            platform.enable_speed_controller()
            platform.set_boundary(50.0)
            platform.reset(rng)
            prev = platform.position.copy()
            for _ in range(800):
                state = platform.step()
                diff_vel = (platform.position[:2] - prev[:2]) / dt
                reported_vel = np.asarray(state["velocity"][:2], dtype=np.float64)
                diff_speed = float(np.linalg.norm(diff_vel))
                reported_speed = float(np.linalg.norm(reported_vel))
                max_diff_speed = max(max_diff_speed, diff_speed)
                max_reported_speed = max(max_reported_speed, reported_speed)
                max_velocity_mismatch = max(
                    max_velocity_mismatch,
                    float(np.linalg.norm(diff_vel - reported_vel)),
                )
                prev = platform.position.copy()

        ok &= _check(
            max_diff_speed <= STAGE2_SPEED_LIMIT_TOL,
            f"{name}: position-diff speed <= {STAGE2_SPEED_LIMIT_TOL:.1f}m/s  [max {max_diff_speed:.3f}]",
        )
        ok &= _check(
            max_reported_speed <= STAGE2_SPEED_LIMIT_TOL,
            f"{name}: reported speed <= {STAGE2_SPEED_LIMIT_TOL:.1f}m/s  [max {max_reported_speed:.3f}]",
        )
        ok &= _check(
            max_velocity_mismatch < 1e-4,
            f"{name}: reported velocity matches position diff  [max err {max_velocity_mismatch:.6f}]",
        )
    return ok


def test_stage2_platform_velocity_observation() -> bool:
    _header("Test 25: Stage-2 Platform Velocity Observation")
    ok = True
    cfg = EnvConfig()
    env = _make_env(cfg, stage=2)
    obs, _ = env.reset(seed=202)
    strategy = env.strategy
    history = list(getattr(strategy, "_velocity_history", []))
    scale = np.asarray(getattr(strategy, "_velocity_scale_xy", np.ones(2)), dtype=np.float32)
    ps = getattr(env, "_platform_state_filtered", {})
    raw = env._get_platform_state()

    ok &= _check(len(history) == 1, "Stage 2 stores initial true platform velocity history")
    ok &= _check(np.all(scale >= 0.98) and np.all(scale <= 1.02),
                 f"Stage 2 velocity scale is within [0.98,1.02]  [got {scale}]")
    expected_initial = history[0].copy()
    expected_initial[:2] *= scale
    ok &= _check(np.allclose(ps["velocity"], expected_initial, atol=1e-5),
                 "initial observed platform velocity is delayed true velocity with scale")

    raw_vels = [raw["velocity"].copy()]
    for _ in range(4):
        obs, _, term, trunc, _ = env.step(np.zeros(4, dtype=np.float32))
        raw_vels.append(env._get_platform_state()["velocity"].copy())
        if term or trunc:
            break

    ps = env._platform_state_filtered
    history = list(getattr(strategy, "_velocity_history", []))
    expected = history[0].copy()
    expected[:2] *= scale
    ok &= _check(len(history) == strategy.VELOCITY_DELAY_STEPS + 1,
                 "Stage 2 velocity history is capped to delay_steps + 1")
    ok &= _check(np.allclose(ps["velocity"], expected, atol=1e-5),
                 "observed platform velocity uses delayed true velocity with scale")
    env.close()
    return ok


def test_performance(env: DroneLandingEnv) -> bool:
    _header("Test 27: Step Performance")
    env.reset(seed=7)
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        action = env.action_space.sample()
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            env.reset(seed=7)
    dt_ms = (time.perf_counter() - t0) / N * 1000
    fps   = 1000.0 / dt_ms

    ok = _check(fps > 500, f"Single-env FPS: {fps:,.0f}  (must be >500 for 16-env viability)")
    print(f"  [INFO] {dt_ms:.3f} ms/step -> {fps:,.0f} FPS")
    return ok


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Environment sanity tests")
    parser.add_argument("--render",  action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    env_config  = EnvConfig()

    print("=" * 60)
    print("  Drone Landing RL — Environment Tests")
    print("=" * 60)

    env = _make_env(env_config, stage=1, render_mode=render_mode)

    results = {
        "Space shapes":          test_spaces(env),
        "Obs normalization":     test_observation_normalization(env),
        "Reward components":     test_reward_components(env),
        "Curriculum stages":     test_curriculum_stages(env_config, render_mode=None),
        "Termination conds":     test_termination_conditions(env, env_config),
        "Action smoothing":      test_action_smoothing(env),
        "Stage-1 defaults":      test_stage1_platform_defaults(env),
        "Capability random":     test_capability_randomization(env),
        "Obs clipping":          test_observation_clipping(env),
        "Action scaling":        test_action_scaling(env),
        "Spawn range":           test_spawn_range_and_oob_margin(env),
        "Stage-1 terminal":      test_stage1_terminal_behavior(),
        "VecMonitor metrics":    test_vecmonitor_preserves_episode_core_fields(),
        "Manual curriculum":     test_manual_curriculum_api(),
        "Disturbance reproducibility": test_disturbance_reproducibility(),
        "Wind drift":            test_zero_action_wind_drift(),
        "Gust activation":       test_gust_activation_and_decay(),
        "Tracking error":        test_tracking_error_actual_velocity(),
        "Disturbance disabled":  test_disturbance_disabled_zero_drift(),
        "Stage-1 reward shape":  test_stage1_strategy_reward_shape(),
        "Stage-2 reward shape":  test_stage2_strategy_reward_shape(),
        "Hold reward independence": test_hold_reward_space_independence(),
        "Success mode score consistency": test_success_mode_score_consistency(),
        "Stage-2 hold refund":   True,  # removed: hold/refund mechanism replaced by wide-sigma peak layer
        "Eval fixed controls":    test_eval_fixed_controls_config(),
        "Stage-2 platform motion": test_stage2_platform_motion_continuity(),
        "Stage-2 velocity obs":   test_stage2_platform_velocity_observation(),
        "Step performance":      test_performance(env),
    }

    env.close()

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results.items():
        sym = PASS if passed else FAIL
        print(f"{sym}  {name}")
        all_pass = all_pass and passed

    print(f"{'='*60}")
    if all_pass:
        print("  All tests passed - ready to train.")
    else:
        print("  Some tests FAILED - fix before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
