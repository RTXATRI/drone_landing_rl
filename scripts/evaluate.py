#!/usr/bin/env python3
# 功能：加载已训练 SAC 模型并执行阶段评估、渲染、轨迹采集和绘图。
"""
评估入口。

加载训练好的模型并运行确定性 rollout。
打印每个 episode 的结果和最终汇总统计。

示例：
    # 评估指定课程的最终模型
    python scripts/evaluate.py --model output/models/drone_landing/model_final --stage 4

    # 使用 GUI 渲染
    python scripts/evaluate.py --model output/models/drone_landing/model_final --stage 4 --render

    # 依次评估全部四个阶段
    python scripts/evaluate.py --model output/models/drone_landing/model_final --all_stages

    # 使用固定条件评估某个课程
    python scripts/evaluate.py --model output/models/stage1_hover/model_final --stage 1 \
        --hover_height 5 --eval_v_xy_max 10 --eval_v_z_up_max 3 --eval_v_z_down_max 2

    # 采集轨迹并生成离线图
    python scripts/evaluate.py --model output/models/drone_landing/model_final \
        --traj_enable --traj_plot
"""

import argparse
import ctypes
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import SAC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.env_config import EnvConfig
from curriculum.strategies import create_strategy, is_hover_stage, registered_stage_ids
from envs.drone_landing_env import DroneLandingEnv

RENDER_SPEED_DEFAULT = 1.0
RENDER_SPEED_MIN = 1.0
RENDER_SPEED_MAX = 10.0
RENDER_SPEED_UNCAPPED = 0.0
EVAL_HOVER_HEIGHT_DEFAULT = 5.0
EVAL_V_XY_MAX_DEFAULT = 10.0
EVAL_V_Z_UP_MAX_DEFAULT = 3.0
EVAL_V_Z_DOWN_MAX_DEFAULT = 2.0
AVG_MINDIST_ENTRY_HORIZ = 0.50
AVG_MINDIST_ENTRY_VERT = 0.50
AVG_MINDIST_ZERO_DEFAULT = 0.20
DIST_SCORE_FULL = 0.005
MINDIST_SCORE_ZERO = 0.050
AVG_MINDIST_SCORE_ZEROS = (0.20, 0.15, 0.10, 0.08)

TRAJ_ENABLE_DEFAULT = False
TRAJ_PLOT_DEFAULT = False
TRAJ_PLOT_REALTIME_DEFAULT = False
TRAJ_PLOT_PER_EPISODE_DEFAULT = True
TRAJ_PLOT_COMBINED_DEFAULT = True
TRAJ_OUT_DIR_DEFAULT = "output/data/eval_traj"
TRAJ_STRIDE_DEFAULT = 1
TRAJ_REALTIME_REFRESH_DEFAULT = 10

TRAJ_COLS = [
    "sim_time",
    "episode",
    "step",
    "stage",
    "drone_x",
    "drone_y",
    "drone_z",
    "drone_vx",
    "drone_vy",
    "drone_vz",
    "drone_roll",
    "drone_pitch",
    "drone_yaw",
    "drone_yaw_rate",
    "platform_x",
    "platform_y",
    "platform_z",
    "platform_vx",
    "platform_vy",
    "platform_vz",
    "platform_fx",
    "platform_fy",
    "platform_fz",
    "platform_fvx",
    "platform_fvy",
    "platform_fvz",
    "target_x",
    "target_y",
    "target_z",
    "target_fx",
    "target_fy",
    "target_fz",
    "dist_3d",
    "dist_xy",
    "speed_norm",
    "action_0",
    "action_1",
    "action_2",
    "action_3",
    "platform_hx",
    "platform_hy",
    "platform_hz",
    "done",
    "success",
]

TRAJ_EP_SUMMARY_COLS = [
    "episode",
    "stage",
    "success",
    "reward",
    "length",
    "train_score",
    "eval_score",
    "min_dist",
    "avg_min_dist",
    "avg_min_dist_valid",
    "min_dist_score",
    "avg_min_dist_score_z020",
    "sim_time_sec",
    "wall_time_sec",
    "sim_to_real_rate",
    "traj_csv",
]


class WindowsTimerResolution:
    """在 Windows 上请求 1ms 计时精度的上下文管理器。"""

    def __init__(self, enable: bool = False, period_ms: int = 1):
        self.enable = bool(enable)
        self.period_ms = int(period_ms)
        self._winmm = None
        self._active = False

    def __enter__(self):
        if not self.enable or os.name != "nt":
            return self

        try:
            self._winmm = ctypes.WinDLL("winmm")
            rc = self._winmm.timeBeginPeriod(self.period_ms)
            if rc == 0:
                self._active = True
                print(f"[INFO] 已启用 Windows 高精度计时 ({self.period_ms}ms)。")
            else:
                print(f"[WARN] 启用高精度计时失败，错误码: {rc}")
        except Exception as exc:
            print(f"[WARN] 启用高精度计时失败: {exc}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.restore()
        return False

    def restore(self) -> None:
        if not self._active or self._winmm is None:
            return
        try:
            rc = self._winmm.timeEndPeriod(self.period_ms)
            if rc != 0:
                print(f"[WARN] 恢复系统计时精度失败，错误码: {rc}")
            else:
                print("[INFO] 已恢复 Windows 默认计时精度。")
        except Exception as exc:
            print(f"[WARN] 恢复系统计时精度失败: {exc}")
        finally:
            self._active = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate drone landing agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",      type=str, required=True,
                   help="Path to model checkpoint (without .zip)")
    p.add_argument("--stage",      type=int, default=4, choices=list(registered_stage_ids()))
    p.add_argument("--episodes",   type=int, default=20)
    p.add_argument("--render",     action="store_true", help="Launch PyBullet GUI")
    p.add_argument("--all_stages", action="store_true",
                   help="Evaluate on all four stages sequentially")
    p.add_argument(
        "--traj_enable",
        action="store_true",
        default=TRAJ_ENABLE_DEFAULT,
        help="Enable detailed trajectory capture during evaluation",
    )
    p.add_argument(
        "--traj_plot",
        action="store_true",
        default=TRAJ_PLOT_DEFAULT,
        help="Generate offline Python trajectory plots after evaluation",
    )
    p.add_argument(
        "--traj_plot_realtime",
        action="store_true",
        default=TRAJ_PLOT_REALTIME_DEFAULT,
        help="Show realtime trajectory plot during evaluation (Python only)",
    )
    p.add_argument(
        "--traj_plot_per_episode",
        action=argparse.BooleanOptionalAction,
        default=TRAJ_PLOT_PER_EPISODE_DEFAULT,
        help="When plotting, output one figure set per episode",
    )
    p.add_argument(
        "--traj_plot_combined",
        action=argparse.BooleanOptionalAction,
        default=TRAJ_PLOT_COMBINED_DEFAULT,
        help="When plotting, output one combined figure set across episodes",
    )
    p.add_argument(
        "--traj_out_dir",
        type=str,
        default=TRAJ_OUT_DIR_DEFAULT,
        help="Root directory for captured trajectory artifacts",
    )
    p.add_argument(
        "--traj_stride",
        type=int,
        default=TRAJ_STRIDE_DEFAULT,
        help="Record one trajectory row every N steps (>=1)",
    )
    p.add_argument(
        "--traj_realtime_refresh",
        type=int,
        default=TRAJ_REALTIME_REFRESH_DEFAULT,
        help="Refresh realtime plots every N steps (>=1)",
    )
    p.add_argument(
        "--eval_max_steps",
        type=int,
        default=None,
        help=(
            "Override episode max steps for evaluation only "
            "(must be >0; <=0 will fall back to config default)"
        ),
    )
    p.add_argument(
        "--hover_height",
        type=float,
        default=None,
        help=(
            "Manual hover target height in meters (hover stages). "
            f"If omitted, interactive evaluation prompts with default {EVAL_HOVER_HEIGHT_DEFAULT:.1f}m."
        ),
    )
    p.add_argument(
        "--eval_v_xy_max",
        type=float,
        default=None,
        help=(
            "Fixed horizontal velocity capability for evaluation only. "
            f"If omitted, prompts with default {EVAL_V_XY_MAX_DEFAULT:.1f}m/s."
        ),
    )
    p.add_argument(
        "--eval_v_z_up_max",
        type=float,
        default=None,
        help=(
            "Fixed upward velocity capability for evaluation only. "
            f"If omitted, prompts with default {EVAL_V_Z_UP_MAX_DEFAULT:.1f}m/s."
        ),
    )
    p.add_argument(
        "--eval_v_z_down_max",
        type=float,
        default=None,
        help=(
            "Fixed downward velocity capability for evaluation only. "
            f"If omitted, prompts with default {EVAL_V_Z_DOWN_MAX_DEFAULT:.1f}m/s."
        ),
    )
    p.add_argument(
        "--render_speed",
        type=float,
        default=RENDER_SPEED_DEFAULT,
        help=(
            "Playback speed multiplier when --render is enabled "
            "(valid: 0=render without sleep, 1.0~10.0=capped speed; default=1.0)"
        ),
    )
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument(
        "--random_seed",
        action="store_true",
        help="Use a fresh random base seed for this evaluation run instead of --seed",
    )
    return p.parse_args()


def sanitize_render_speed(speed: float) -> float:
    """将渲染速度规范化到 {0} U [1, 10]。"""
    try:
        v = float(speed)
    except (TypeError, ValueError):
        print(
            f"[WARN] --render_speed={speed!r} 无效，已回退到默认值 "
            f"{RENDER_SPEED_DEFAULT:.1f}。"
        )
        return RENDER_SPEED_DEFAULT

    if not np.isfinite(v):
        print(
            f"[WARN] --render_speed={v} 无效，已回退到默认值 "
            f"{RENDER_SPEED_DEFAULT:.1f}。"
        )
        return RENDER_SPEED_DEFAULT

    if v == RENDER_SPEED_UNCAPPED:
        return RENDER_SPEED_UNCAPPED

    if 0.0 < v < RENDER_SPEED_MIN:
        print(
            f"[WARN] --render_speed={v} 处于 (0, {RENDER_SPEED_MIN:.1f})，"
            f"已调整为 {RENDER_SPEED_MIN:.1f}x。"
        )
        return RENDER_SPEED_MIN

    if v > RENDER_SPEED_MAX:
        print(
            f"[WARN] --render_speed={v} 超出范围 "
            f"[{RENDER_SPEED_MIN:.1f}, {RENDER_SPEED_MAX:.1f}]，"
            f"已调整为 {RENDER_SPEED_MAX:.1f}x。"
        )
        return RENDER_SPEED_MAX

    if v < 0.0:
        print(
            f"[WARN] --render_speed={v} 不能为负，已回退到默认值 "
            f"{RENDER_SPEED_DEFAULT:.1f}。"
        )
        return RENDER_SPEED_DEFAULT

    return v


def sanitize_eval_max_steps(max_steps: Optional[int], default_steps: int) -> int:
    """规范化评估最大步数，无效输入时回退默认值。"""
    if max_steps is None:
        return int(default_steps)

    try:
        v = int(max_steps)
    except (TypeError, ValueError):
        print(
            f"[WARN] --eval_max_steps={max_steps!r} 无效，已回退到默认值 "
            f"{int(default_steps)}。"
        )
        return int(default_steps)

    if v <= 0:
        print(
            f"[WARN] --eval_max_steps={v} 必须大于 0，已回退到默认值 "
            f"{int(default_steps)}。"
        )
        return int(default_steps)

    return v


def sanitize_positive_int(value: int, default: int, name: str) -> int:
    """规范化正整数参数，无效时回退默认值。"""
    try:
        v = int(value)
    except (TypeError, ValueError):
        print(f"[WARN] --{name}={value!r} 无效，已回退到默认值 {default}。")
        return int(default)

    if v < 1:
        print(f"[WARN] --{name}={v} 必须>=1，已回退到默认值 {default}。")
        return int(default)

    return v


def distance_score(distance: float, *, full: float, zero: float) -> float:
    """Map a distance to a 0-100 score with a linear falloff."""
    if not np.isfinite(distance):
        return 0.0
    d = float(distance)
    if d <= full:
        return 100.0
    if d >= zero:
        return 0.0
    return float(100.0 * (zero - d) / (zero - full))


def _stat(values: List[float], fn: str) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if fn == "mean":
        return float(np.mean(arr))
    if fn == "median":
        return float(np.median(arr))
    if fn == "p90":
        return float(np.percentile(arr, 90))
    if fn == "max":
        return float(np.max(arr))
    raise ValueError(f"Unsupported stat: {fn}")


def resolve_eval_seed(seed: int, random_seed: bool) -> int:
    """返回本次评估使用的 base seed。"""
    if random_seed:
        # os.urandom 不受 numpy/random 全局状态影响，适合作为一次性 run seed。
        seed = int.from_bytes(os.urandom(4), byteorder="little", signed=False)
        print(f"[INFO] Random eval seed: {seed}")
        return seed
    return int(seed)


def _positive_finite_float(value: float, name: str) -> float:
    """校验一个有限正浮点数。"""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字。") from exc

    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} 必须是有限正数。")
    return v


def _prompt_positive_float(label: str, default: float, unit: str) -> float:
    """提示输入正浮点数；空输入和非交互 stdin 使用默认值。"""
    if not sys.stdin.isatty():
        print(f"[INFO] 非交互环境：{label} 使用默认值 {default:.2f}{unit}。")
        return float(default)

    while True:
        try:
            raw = input(f"{label} [default = {default:.2f}{unit}]：").strip()
        except EOFError:
            print(f"[INFO] 输入流关闭：{label} 使用默认值 {default:.2f}{unit}。")
            return float(default)

        if raw == "":
            return float(default)

        try:
            return _positive_finite_float(raw, label)
        except ValueError as exc:
            print(f"输入无效：{exc}")


def _resolve_eval_float(
    cli_value: Optional[float],
    default: float,
    label: str,
    unit: str,
    arg_name: str,
) -> float:
    """从 CLI 或交互/默认输入解析一个评估控制值。"""
    if cli_value is not None:
        try:
            return _positive_finite_float(cli_value, arg_name)
        except ValueError as exc:
            print(f"[ERROR] --{arg_name} 无效：{exc}")
            sys.exit(2)
    return _prompt_positive_float(label, default, unit)


def apply_eval_capability_config(
    env_config: EnvConfig,
    eval_v_xy_max: float,
    eval_v_z_up_max: float,
    eval_v_z_down_max: float,
) -> None:
    """将固定速度能力应用到该评估配置。"""
    eval_v_xy_max = _positive_finite_float(eval_v_xy_max, "eval_v_xy_max")
    eval_v_z_up_max = _positive_finite_float(eval_v_z_up_max, "eval_v_z_up_max")
    eval_v_z_down_max = _positive_finite_float(eval_v_z_down_max, "eval_v_z_down_max")
    if eval_v_z_down_max > eval_v_z_up_max:
        raise ValueError("eval_v_z_down_max 不能大于 eval_v_z_up_max。")

    oc = env_config.observation
    oc.v_xy_min = eval_v_xy_max
    oc.v_xy_max = eval_v_xy_max
    oc.v_z_up_min = eval_v_z_up_max
    oc.v_z_up_max = eval_v_z_up_max
    oc.v_z_down_min = eval_v_z_down_max
    oc.v_z_down_max = eval_v_z_down_max


def resolve_eval_controls(args: argparse.Namespace, env_config: EnvConfig, stages: List[int]) -> tuple:
    """解析固定评估悬停高度和速度能力。"""
    needs_hover_height = any(is_hover_stage(stage) for stage in stages)
    if needs_hover_height:
        hover_height = _resolve_eval_float(
            args.hover_height,
            EVAL_HOVER_HEIGHT_DEFAULT,
            "评估悬停高度",
            "m",
            "hover_height",
        )
    else:
        hover_height = EVAL_HOVER_HEIGHT_DEFAULT

    eval_v_xy_max = _resolve_eval_float(
        args.eval_v_xy_max,
        EVAL_V_XY_MAX_DEFAULT,
        "评估水平最大速度",
        "m/s",
        "eval_v_xy_max",
    )
    eval_v_z_up_max = _resolve_eval_float(
        args.eval_v_z_up_max,
        EVAL_V_Z_UP_MAX_DEFAULT,
        "评估上升最大速度",
        "m/s",
        "eval_v_z_up_max",
    )

    while True:
        eval_v_z_down_max = _resolve_eval_float(
            args.eval_v_z_down_max,
            EVAL_V_Z_DOWN_MAX_DEFAULT,
            "评估下降最大速度",
            "m/s",
            "eval_v_z_down_max",
        )
        if eval_v_z_down_max <= eval_v_z_up_max:
            break
        msg = "eval_v_z_down_max 不能大于 eval_v_z_up_max。"
        if args.eval_v_z_down_max is not None or not sys.stdin.isatty():
            print(f"[ERROR] {msg}")
            sys.exit(2)
        print(f"输入无效：{msg}")

    try:
        apply_eval_capability_config(
            env_config,
            eval_v_xy_max=eval_v_xy_max,
            eval_v_z_up_max=eval_v_z_up_max,
            eval_v_z_down_max=eval_v_z_down_max,
        )
    except ValueError as exc:
        print(f"[ERROR] 评估配置无效：{exc}")
        sys.exit(2)

    print(f"[INFO] Eval hover height: {hover_height:.2f} m")
    print(
        "[INFO] Eval velocity caps: "
        f"v_xy={eval_v_xy_max:.2f} m/s, "
        f"vz_up={eval_v_z_up_max:.2f} m/s, "
        f"vz_down={eval_v_z_down_max:.2f} m/s"
    )
    return hover_height, eval_v_xy_max, eval_v_z_up_max, eval_v_z_down_max


def _draw_platform_outline(ax, center: np.ndarray, half_extents: np.ndarray,
                           color: str = "tab:red", alpha: float = 0.9) -> None:
    """在 3D 图中绘制矩形降落平台顶面的轮廓。"""
    cx, cy, cz = map(float, center)
    hx, hy, hz = map(float, half_extents)
    z_top = cz + hz
    xs = [cx - hx, cx + hx, cx + hx, cx - hx, cx - hx]
    ys = [cy - hy, cy - hy, cy + hy, cy + hy, cy - hy]
    zs = [z_top] * 5
    ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=1.6)


def _write_csv_rows(path: str, fieldnames: List[str], rows: List[Dict]) -> None:
    """按稳定表头顺序将 dict 行写入 CSV。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_stage(
    model: SAC,
    env_config: EnvConfig,
    stage: int,
    n_episodes: int,
    render: bool = False,
    hover_height: Optional[float] = None,
    render_speed: float = 1.0,
    seed: int = 0,
    traj_enable: bool = False,
    traj_stage_dir: Optional[str] = None,
    traj_stride: int = TRAJ_STRIDE_DEFAULT,
    traj_plot_realtime: bool = False,
    traj_realtime_refresh: int = TRAJ_REALTIME_REFRESH_DEFAULT,
) -> dict:
    """运行 n_episodes 次确定性 rollout，并返回汇总统计。"""
    render_mode = "human" if render else None
    strategy = create_strategy(stage, env_config)
    if is_hover_stage(stage) and hover_height is not None:
        strategy.set_fixed_hover_height(hover_height)
    env = DroneLandingEnv(
        env_config,
        strategy=strategy,
        render_mode=render_mode,
    )
    if is_hover_stage(stage):
        env.set_success_mode("eval")

    # GUI 模式下限制循环速度，让推演过程肉眼可观察。
    speed = float(render_speed)
    uncapped_render = render and speed == RENDER_SPEED_UNCAPPED
    step_wall_time = env_config.episode.dt / speed if (render and not uncapped_render) else 0.0

    traj_stride = sanitize_positive_int(traj_stride, TRAJ_STRIDE_DEFAULT, "traj_stride")
    traj_realtime_refresh = sanitize_positive_int(
        traj_realtime_refresh,
        TRAJ_REALTIME_REFRESH_DEFAULT,
        "traj_realtime_refresh",
    )
    if traj_enable and traj_stage_dir:
        os.makedirs(traj_stage_dir, exist_ok=True)
    platform_half_extents = np.array(env_config.platform.half_extents, dtype=np.float32)

    successes, rewards, lengths, dists = [], [], [], []
    avg_min_dists, avg_min_dist_valids = [], []
    min_dist_scores = []
    avg_min_dist_scores_by_zero = {
        zero: [] for zero in AVG_MINDIST_SCORE_ZEROS
    }
    sim_times, wall_times, sim_to_real_rates = [], [], []
    traj_episode_rows: List[Dict] = []
    traj_episode_csv_paths: List[str] = []

    realtime_enabled = bool(traj_plot_realtime)
    plt = None
    rt_fig = rt_ax3d = rt_ax_speed = rt_ax_pos = None
    if realtime_enabled:
        try:
            import matplotlib.pyplot as plt_mod

            plt = plt_mod
            plt.ion()
            rt_fig = plt.figure("Eval Trajectory Realtime", figsize=(16, 5))
            rt_ax3d = rt_fig.add_subplot(1, 3, 1, projection="3d")
            rt_ax_speed = rt_fig.add_subplot(1, 3, 2)
            rt_ax_pos = rt_fig.add_subplot(1, 3, 3)
        except Exception as exc:
            realtime_enabled = False
            print(f"[WARN] 实时绘图初始化失败，已禁用 realtime 模式: {exc}")

    print(f"\n{'─'*60}")
    print(f"  Stage {stage}: Evaluating {n_episodes} episodes…")
    if is_hover_stage(stage) and hover_height is not None:
        print(f"  Fixed hover target height: {hover_height:.2f} m")
    ep_max_steps = int(env_config.episode.max_steps)
    ep_max_sim_time = ep_max_steps * float(env_config.episode.dt)
    print(f"  回合上限: {ep_max_steps} steps ({ep_max_sim_time:.1f}s sim-time)")
    if uncapped_render:
        run_mode = "高倍速渲染（不 sleep）"
    elif render:
        run_mode = f"目标速率 {speed:.2f}x"
    else:
        run_mode = "非可视化全速"
    print(f"  正在进行 {n_episodes} 次仿真（{run_mode}）")
    if traj_enable:
        stride_dt = traj_stride * float(env_config.episode.dt)
        print(f"  轨迹采集: 开启（stride={traj_stride} step, {stride_dt:.3f}s）")
    if realtime_enabled:
        print(f"  实时绘图: 开启（每 {traj_realtime_refresh} step 刷新）")
    print(f"{'─'*60}")

    try:
        for ep in range(n_episodes):
            ep_wall_t0 = time.perf_counter()

            obs, _ = env.reset(seed=seed + ep)

            done = False
            ep_r = 0.0
            ep_len = 0
            min_d = np.inf
            avg_min_dist_active = False
            avg_min_dist_sum = 0.0
            avg_min_dist_count = 0

            episode_rows: List[Dict] = []
            episode_csv_path = None
            if traj_enable and traj_stage_dir:
                episode_dir = os.path.join(traj_stage_dir, f"episode_{ep+1:03d}")
                os.makedirs(episode_dir, exist_ok=True)
                episode_csv_path = os.path.join(episode_dir, "trajectory.csv")

            rt_t: List[float] = []
            rt_dx: List[float] = []
            rt_dy: List[float] = []
            rt_dz: List[float] = []
            rt_px: List[float] = []
            rt_py: List[float] = []
            rt_pz: List[float] = []
            rt_tx: List[float] = []
            rt_ty: List[float] = []
            rt_tz: List[float] = []
            rt_vx: List[float] = []
            rt_vy: List[float] = []
            rt_vz: List[float] = []
            rt_vnorm: List[float] = []

            while not done:
                step_t0 = time.perf_counter()

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_r += reward
                ep_len += 1
                done = terminated or truncated
                step_dist = float(info.get("metric/dist", np.inf))
                horiz_err = float(info.get("metric/horiz_err", np.inf))
                vert_err = float(info.get("metric/vert_err", np.inf))
                min_d = min(min_d, step_dist)

                if (
                    not avg_min_dist_active
                    and horiz_err <= AVG_MINDIST_ENTRY_HORIZ
                    and vert_err <= AVG_MINDIST_ENTRY_VERT
                ):
                    avg_min_dist_active = True
                if avg_min_dist_active and np.isfinite(step_dist):
                    avg_min_dist_sum += step_dist
                    avg_min_dist_count += 1

                # 如果下游功能需要状态，则每步只获取一次。
                need_state = traj_enable or realtime_enabled
                ds = ps = target_pos = None
                if need_state:
                    ds = env._get_drone_state()
                    ps = env._get_platform_state()
                    fps = getattr(env, '_platform_state_filtered', None) or {}
                if traj_enable or realtime_enabled:
                    target_pos = env._get_target_pos(ps["position"])
                    if fps:
                        ftarget_pos = fps["position"] + np.array([0, 0, env.get_hover_height()], dtype=np.float32)

                sim_time = ep_len * float(env_config.episode.dt)
                if traj_enable and ds is not None and ps is not None and target_pos is not None:
                    dist_vec = target_pos - ds["position"]
                    dist_3d = float(np.linalg.norm(dist_vec))
                    dist_xy = float(np.linalg.norm(dist_vec[:2]))
                    speed_norm = float(np.linalg.norm(ds["velocity"]))
                    act = np.asarray(action, dtype=np.float32).reshape(-1)
                    if act.size < 4:
                        act = np.pad(act, (0, 4 - act.size))
                    if (ep_len % traj_stride == 0) or done:
                        episode_rows.append({
                            "sim_time": round(sim_time, 6),
                            "episode": ep + 1,
                            "step": ep_len,
                            "stage": stage,
                            "drone_x": round(float(ds["position"][0]), 6),
                            "drone_y": round(float(ds["position"][1]), 6),
                            "drone_z": round(float(ds["position"][2]), 6),
                            "drone_vx": round(float(ds["velocity"][0]), 6),
                            "drone_vy": round(float(ds["velocity"][1]), 6),
                            "drone_vz": round(float(ds["velocity"][2]), 6),
                            "drone_roll": round(float(ds["euler"][0]), 6),
                            "drone_pitch": round(float(ds["euler"][1]), 6),
                            "drone_yaw": round(float(ds["euler"][2]), 6),
                            "drone_yaw_rate": round(float(ds["yaw_rate"]), 6),
                            "platform_x": round(float(ps["position"][0]), 6),
                            "platform_y": round(float(ps["position"][1]), 6),
                            "platform_z": round(float(ps["position"][2]), 6),
                            "platform_vx": round(float(ps["velocity"][0]), 6),
                            "platform_vy": round(float(ps["velocity"][1]), 6),
                            "platform_vz": round(float(ps["velocity"][2]), 6),
                            "platform_fx": round(float(fps.get("position", ps["position"])[0]), 6) if fps else round(float(ps["position"][0]), 6),
                            "platform_fy": round(float(fps.get("position", ps["position"])[1]), 6) if fps else round(float(ps["position"][1]), 6),
                            "platform_fz": round(float(fps.get("position", ps["position"])[2]), 6) if fps else round(float(ps["position"][2]), 6),
                            "platform_fvx": round(float(fps.get("velocity", ps["velocity"])[0]), 6) if fps else round(float(ps["velocity"][0]), 6),
                            "platform_fvy": round(float(fps.get("velocity", ps["velocity"])[1]), 6) if fps else round(float(ps["velocity"][1]), 6),
                            "platform_fvz": round(float(fps.get("velocity", ps["velocity"])[2]), 6) if fps else round(float(ps["velocity"][2]), 6),
                            "target_x": round(float(target_pos[0]), 6),
                            "target_y": round(float(target_pos[1]), 6),
                            "target_z": round(float(target_pos[2]), 6),
                            "target_fx": round(float(ftarget_pos[0]), 6) if fps else round(float(target_pos[0]), 6),
                            "target_fy": round(float(ftarget_pos[1]), 6) if fps else round(float(target_pos[1]), 6),
                            "target_fz": round(float(ftarget_pos[2]), 6) if fps else round(float(target_pos[2]), 6),
                            "dist_3d": round(dist_3d, 6),
                            "dist_xy": round(dist_xy, 6),
                            "speed_norm": round(speed_norm, 6),
                            "action_0": round(float(act[0]), 6),
                            "action_1": round(float(act[1]), 6),
                            "action_2": round(float(act[2]), 6),
                            "action_3": round(float(act[3]), 6),
                            "platform_hx": round(float(platform_half_extents[0]), 6),
                            "platform_hy": round(float(platform_half_extents[1]), 6),
                            "platform_hz": round(float(platform_half_extents[2]), 6),
                            "done": int(done),
                            "success": int(info.get("episode", {}).get("success", False)) if done else 0,
                        })

                if realtime_enabled and ds is not None and ps is not None and target_pos is not None:
                    rt_t.append(sim_time)
                    rt_dx.append(float(ds["position"][0]))
                    rt_dy.append(float(ds["position"][1]))
                    rt_dz.append(float(ds["position"][2]))
                    rt_px.append(float(ps["position"][0]))
                    rt_py.append(float(ps["position"][1]))
                    rt_pz.append(float(ps["position"][2]))
                    rt_tx.append(float(target_pos[0]))
                    rt_ty.append(float(target_pos[1]))
                    rt_tz.append(float(target_pos[2]))
                    rt_vx.append(float(ds["velocity"][0]))
                    rt_vy.append(float(ds["velocity"][1]))
                    rt_vz.append(float(ds["velocity"][2]))
                    rt_vnorm.append(float(np.linalg.norm(ds["velocity"])))
                    rt_fpx = getattr(env, '_rt_fpx', None)
                    if rt_fpx is None:
                        env._rt_fpx = []; env._rt_fpy = []; env._rt_fpz = []
                        env._rt_ftx = []; env._rt_fty = []; env._rt_ftz = []
                        env._rt_fpv = []
                    if fps:
                        env._rt_fpx.append(float(fps["position"][0]))
                        env._rt_fpy.append(float(fps["position"][1]))
                        env._rt_fpz.append(float(fps["position"][2]))
                        env._rt_ftx.append(float(ftarget_pos[0]))
                        env._rt_fty.append(float(ftarget_pos[1]))
                        env._rt_ftz.append(float(ftarget_pos[2]))
                        env._rt_fpv.append(float(np.linalg.norm(fps["velocity"])))

                    if (ep_len % traj_realtime_refresh == 0) or done:
                        rt_ax3d.cla()
                        rt_ax3d.plot(rt_dx, rt_dy, rt_dz, color="tab:blue", label="drone")
                        rt_ax3d.plot(rt_px, rt_py, rt_pz, color="tab:red", alpha=0.7, label="platform")
                        rt_ax3d.plot(rt_tx, rt_ty, rt_tz, color="tab:green", ls="--", alpha=0.8, label="target")
                        if env._rt_fpx:
                            rt_ax3d.plot(env._rt_fpx, env._rt_fpy, env._rt_fpz, color="tab:red", ls=":", alpha=0.5, label="plat(f)")
                            rt_ax3d.plot(env._rt_ftx, env._rt_fty, env._rt_ftz, color="tab:green", ls=":", alpha=0.5, label="tgt(f)")
                        if rt_px:
                            _draw_platform_outline(
                                rt_ax3d,
                                np.array([rt_px[0], rt_py[0], rt_pz[0]], dtype=np.float32),
                                platform_half_extents,
                                color="tab:red",
                                alpha=0.4,
                            )
                            _draw_platform_outline(
                                rt_ax3d,
                                np.array([rt_px[-1], rt_py[-1], rt_pz[-1]], dtype=np.float32),
                                platform_half_extents,
                                color="tab:red",
                                alpha=0.9,
                            )
                        rt_ax3d.set_title(f"Episode {ep+1}/{n_episodes} Trajectory")
                        rt_ax3d.set_xlabel("X (m)")
                        rt_ax3d.set_ylabel("Y (m)")
                        rt_ax3d.set_zlabel("Z (m)")
                        rt_ax3d.legend(loc="upper right")
                        rt_ax3d.grid(alpha=0.25)

                        rt_ax_speed.cla()
                        rt_ax_speed.plot(rt_t, rt_vx, label="vx")
                        rt_ax_speed.plot(rt_t, rt_vy, label="vy")
                        rt_ax_speed.plot(rt_t, rt_vz, label="vz")
                        rt_ax_speed.plot(rt_t, rt_vnorm, label="|v|", lw=2.0)
                        if env._rt_fpv:
                            rt_ax_speed.plot(rt_t, env._rt_fpv, color="tab:red", ls=":", label="|v|_plat(f)", lw=1.5)
                        rt_ax_speed.set_title("Velocity vs Sim Time")
                        rt_ax_speed.set_xlabel("sim time (s)")
                        rt_ax_speed.set_ylabel("velocity (m/s)")
                        rt_ax_speed.grid(alpha=0.25)
                        rt_ax_speed.legend(loc="upper right")

                        rt_ax_pos.cla()
                        rt_ax_pos.plot(rt_t, rt_dx, label="x")
                        rt_ax_pos.plot(rt_t, rt_dy, label="y")
                        rt_ax_pos.plot(rt_t, rt_dz, label="z")
                        rt_ax_pos.set_title("Drone Position vs Sim Time")
                        rt_ax_pos.set_xlabel("sim time (s)")
                        rt_ax_pos.set_ylabel("position (m)")
                        rt_ax_pos.grid(alpha=0.25)
                        rt_ax_pos.legend(loc="upper right")

                        plt.pause(0.001)

                if render and step_wall_time > 0.0:
                    elapsed = time.perf_counter() - step_t0
                    remain = step_wall_time - elapsed
                    if remain > 0:
                        time.sleep(remain)

            success = info.get("episode", {}).get("success", False)
            episode_info = info.get("episode", {})
            train_score_val = float(episode_info.get("train_score", 0.0))
            eval_score_val = float(episode_info.get("eval_score", 0.0))
            avg_min_dist_valid = avg_min_dist_count > 0
            avg_min_dist = (
                avg_min_dist_sum / float(avg_min_dist_count)
                if avg_min_dist_valid else AVG_MINDIST_ZERO_DEFAULT
            )
            min_dist_score = distance_score(
                min_d, full=DIST_SCORE_FULL, zero=MINDIST_SCORE_ZERO
            )
            avg_min_dist_scores = {
                zero: (
                    distance_score(avg_min_dist, full=DIST_SCORE_FULL, zero=zero)
                    if avg_min_dist_valid else 0.0
                )
                for zero in AVG_MINDIST_SCORE_ZEROS
            }

            ep_wall = max(time.perf_counter() - ep_wall_t0, 1e-9)
            ep_sim = ep_len * env_config.episode.dt
            ep_rate = ep_sim / ep_wall

            successes.append(success)
            rewards.append(ep_r)
            lengths.append(ep_len)
            dists.append(min_d)
            avg_min_dists.append(avg_min_dist)
            avg_min_dist_valids.append(float(avg_min_dist_valid))
            min_dist_scores.append(min_dist_score)
            for zero, score in avg_min_dist_scores.items():
                avg_min_dist_scores_by_zero[zero].append(score)
            sim_times.append(ep_sim)
            wall_times.append(ep_wall)
            sim_to_real_rates.append(ep_rate)

            if traj_enable and episode_csv_path is not None:
                _write_csv_rows(episode_csv_path, TRAJ_COLS, episode_rows)
                traj_episode_csv_paths.append(episode_csv_path)
                traj_episode_rows.append({
                    "episode": ep + 1,
                    "stage": stage,
                    "success": int(success),
                    "reward": round(float(ep_r), 6),
                    "length": ep_len,
                    "train_score": round(train_score_val, 3),
                    "eval_score": round(eval_score_val, 3),
                    "min_dist": round(float(min_d), 6),
                    "avg_min_dist": round(float(avg_min_dist), 6),
                    "avg_min_dist_valid": int(avg_min_dist_valid),
                    "min_dist_score": round(float(min_dist_score), 3),
                    "avg_min_dist_score_z020": round(
                        float(avg_min_dist_scores.get(0.20, 0.0)), 3
                    ),
                    "sim_time_sec": round(float(ep_sim), 6),
                    "wall_time_sec": round(float(ep_wall), 6),
                    "sim_to_real_rate": round(float(ep_rate), 6),
                    "traj_csv": os.path.relpath(episode_csv_path, traj_stage_dir) if traj_stage_dir else episode_csv_path,
                })

            status = "✓ SUCCESS" if success else "✗ FAIL   "
            eval_suffix = ""
            if is_hover_stage(stage):
                eval_suffix = f" | Train={train_score_val:.1f} Eval={eval_score_val:.1f}"
            print(
                f"  Ep {ep+1:3d}/{n_episodes} | {status} | "
                f"R={ep_r:+8.2f} | L={ep_len:4d} | minDist={min_d:.3f}m"
                f" | avgMinDist={avg_min_dist:.3f}m"
                f" | avgValid={int(avg_min_dist_valid)}"
                f" | rate={ep_rate:.2f}x"
                f"{eval_suffix}"
            )
    finally:
        if realtime_enabled and plt is not None and rt_fig is not None:
            try:
                plt.ioff()
                plt.close(rt_fig)
            except Exception:
                pass
        env.close()

    summary = {
        "stage":        stage,
        "episodes":     n_episodes,
        "success_rate": float(np.mean(successes)),
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "mean_length":  float(np.mean(lengths)),
        "mean_min_dist": float(np.mean(dists)),
        "median_min_dist": _stat(dists, "median"),
        "p90_min_dist": _stat(dists, "p90"),
        "max_min_dist": _stat(dists, "max"),
        "mean_avg_min_dist": float(np.mean(avg_min_dists)),
        "median_avg_min_dist": _stat(avg_min_dists, "median"),
        "p90_avg_min_dist": _stat(avg_min_dists, "p90"),
        "max_avg_min_dist": _stat(avg_min_dists, "max"),
        "avg_min_dist_valid_rate": float(np.mean(avg_min_dist_valids)),
        "mean_min_dist_score": float(np.mean(min_dist_scores)),
        "mean_sim_time_sec": float(np.mean(sim_times)),
        "mean_wall_time_sec": float(np.mean(wall_times)),
        "mean_sim_to_real_rate": float(np.mean(sim_to_real_rates)),
    }
    for zero, scores in avg_min_dist_scores_by_zero.items():
        suffix = f"z{int(round(zero * 1000)):03d}"
        summary[f"mean_avg_min_dist_score_{suffix}"] = float(np.mean(scores))

    if traj_enable and traj_stage_dir:
        summary_csv_path = os.path.join(traj_stage_dir, "episode_summary.csv")
        summary_cols = [
            "episode", "stage", "success", "reward", "length",
            "train_score", "eval_score",
            "min_dist", "avg_min_dist", "avg_min_dist_valid",
            "min_dist_score", "avg_min_dist_score_z020",
            "sim_time_sec", "wall_time_sec",
            "sim_to_real_rate", "traj_csv",
        ]
        _write_csv_rows(summary_csv_path, summary_cols, traj_episode_rows)
        meta_path = os.path.join(traj_stage_dir, "stage_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "stage": stage,
                    "episodes": n_episodes,
                    "seed": int(seed),
                    "dt": float(env_config.episode.dt),
                    "max_steps": int(env_config.episode.max_steps),
                    "platform_half_extents": [float(x) for x in platform_half_extents.tolist()],
                },
                f,
                indent=2,
            )
        summary["traj_stage_dir"] = traj_stage_dir
        summary["traj_episode_csv_count"] = len(traj_episode_csv_paths)
        summary["traj_summary_csv"] = summary_csv_path

    return summary


def print_summary(results: list) -> None:
    print(f"\n{'='*60}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Stage':<8} {'Success':>8} {'MeanR':>10} {'MeanL':>8} {'Rate':>8}")
    print(f"  {'─'*56}")
    for r in results:
        print(f"  Stage {r['stage']}  {r['success_rate']:>7.1%}  "
              f"{r['mean_reward']:>+10.2f}  {r['mean_length']:>8.1f}  "
              f"{r['mean_sim_to_real_rate']:>7.2f}x")
        print(f"           mean sim={r['mean_sim_time_sec']:.1f}s, "
              f"mean wall={r['mean_wall_time_sec']:.1f}s")
        print(
            "           minDist: "
            f"mean={r['mean_min_dist']:.4f}m, "
            f"median={r['median_min_dist']:.4f}m, "
            f"p90={r['p90_min_dist']:.4f}m, "
            f"max={r['max_min_dist']:.4f}m, "
            f"score={r['mean_min_dist_score']:.1f}"
        )
        print(
            "           avgMinDist: "
            f"mean={r['mean_avg_min_dist']:.4f}m, "
            f"median={r['median_avg_min_dist']:.4f}m, "
            f"p90={r['p90_avg_min_dist']:.4f}m, "
            f"max={r['max_avg_min_dist']:.4f}m, "
            f"valid={r['avg_min_dist_valid_rate']:.1%}"
        )
        print(
            "           avgMinDist scores: "
            f"z0.20={r['mean_avg_min_dist_score_z200']:.1f}, "
            f"z0.15={r['mean_avg_min_dist_score_z150']:.1f}, "
            f"z0.10={r['mean_avg_min_dist_score_z100']:.1f}, "
            f"z0.08={r['mean_avg_min_dist_score_z080']:.1f}"
        )
    print(f"{'='*60}\n")


def main() -> None:
    args = parse_args()
    eval_seed = resolve_eval_seed(args.seed, args.random_seed)

    env_config = EnvConfig()
    default_eval_max_steps = int(env_config.episode.max_steps)
    eval_max_steps = sanitize_eval_max_steps(args.eval_max_steps, default_eval_max_steps)
    env_config.episode.max_steps = int(eval_max_steps)

    if args.eval_max_steps is not None:
        print(f"[INFO] 本次评估 max_steps={env_config.episode.max_steps}（仅评估生效）。")

    stages = list(registered_stage_ids()) if args.all_stages else [args.stage]
    hover_height, _, _, _ = resolve_eval_controls(args, env_config, stages)

    model      = SAC.load(args.model, device=args.device)

    render_speed = sanitize_render_speed(args.render_speed) if args.render else args.render_speed
    traj_enable = bool(args.traj_enable)
    traj_plot = bool(args.traj_plot)
    traj_plot_realtime = bool(args.traj_plot_realtime)
    traj_plot_per_episode = bool(args.traj_plot_per_episode)
    traj_plot_combined = bool(args.traj_plot_combined)
    traj_stride = sanitize_positive_int(args.traj_stride, TRAJ_STRIDE_DEFAULT, "traj_stride")
    traj_realtime_refresh = sanitize_positive_int(
        args.traj_realtime_refresh,
        TRAJ_REALTIME_REFRESH_DEFAULT,
        "traj_realtime_refresh",
    )

    if (traj_plot or traj_plot_realtime) and not traj_enable:
        traj_enable = True
        print("[INFO] 已自动开启轨迹采集（因为启用了轨迹绘图选项）。")

    traj_run_dir = None
    if traj_enable:
        model_tag = os.path.splitext(os.path.basename(args.model))[0]
        run_tag = time.strftime("%Y%m%d_%H%M%S")
        traj_run_dir = os.path.join(args.traj_out_dir, f"{model_tag}_{run_tag}")
        os.makedirs(traj_run_dir, exist_ok=True)
        print(f"[INFO] 轨迹输出目录: {traj_run_dir}")

    all_results = []

    # GUI 渲染期间在 Windows 上启用高精度计时。
    try:
        with WindowsTimerResolution(enable=args.render):
            for stage in stages:
                stage_traj_dir = (
                    os.path.join(traj_run_dir, f"stage_{stage}")
                    if traj_enable and traj_run_dir
                    else None
                )
                result = evaluate_stage(
                    model, env_config, stage,
                    n_episodes=args.episodes,
                    render=(args.render and stage == stages[-1]),
                    hover_height=hover_height,
                    render_speed=render_speed,
                    seed=eval_seed,
                    traj_enable=traj_enable,
                    traj_stage_dir=stage_traj_dir,
                    traj_stride=traj_stride,
                    traj_plot_realtime=traj_plot_realtime,
                    traj_realtime_refresh=traj_realtime_refresh,
                )
                all_results.append(result)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断评估，已安全退出。")
        return

    if traj_plot and traj_enable and traj_run_dir:
        try:
            from scripts.plot_eval_trajectory import plot_trajectory_directory

            for stage_result in all_results:
                stage_dir = stage_result.get("traj_stage_dir")
                if stage_dir:
                    plot_trajectory_directory(
                        stage_dir=stage_dir,
                        per_episode=traj_plot_per_episode,
                        combined=traj_plot_combined,
                        show=False,
                    )
            print(f"[INFO] 离线轨迹图已生成: {traj_run_dir}")
        except Exception as exc:
            print(f"[WARN] 离线轨迹绘图失败: {exc}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
