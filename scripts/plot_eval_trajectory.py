#!/usr/bin/env python3
# 功能：读取评估轨迹 CSV 并生成无人机、平台和目标轨迹图。
"""
评估轨迹的离线绘图工具。

读取 scripts/evaluate.py 生成的轨迹 CSV，并输出：
  - 每个 episode 的图（3D 轨迹 + 速度 + 距离 + XYZ 位置）
  - 跨 episode 的合并图

用法：
  python scripts/plot_eval_trajectory.py --stage_dir output/data/eval_traj/<run>/stage_1
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _set_axes_equal_3d(ax) -> None:
    """为 3D 坐标轴设置等比例尺度。"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def _draw_platform_outline(ax, cx: float, cy: float, cz: float,
                           hx: float, hy: float, hz: float,
                           color: str = "tab:red", alpha: float = 0.9) -> None:
    z_top = cz + hz
    xs = [cx - hx, cx + hx, cx + hx, cx - hx, cx - hx]
    ys = [cy - hy, cy - hy, cy + hy, cy + hy, cy - hy]
    zs = [z_top] * 5
    ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=1.6)


def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def _plot_episode(df: pd.DataFrame, out_dir: str, episode_name: str) -> None:
    fig = plt.figure(figsize=(14, 9))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    axv = fig.add_subplot(2, 2, 2)
    axd = fig.add_subplot(2, 2, 3)
    axp = fig.add_subplot(2, 2, 4)

    ax3d.plot(df["drone_x"], df["drone_y"], df["drone_z"], label="drone", color="tab:blue")
    ax3d.plot(df["platform_x"], df["platform_y"], df["platform_z"], label="platform", color="tab:red", alpha=0.8)
    ax3d.plot(df["target_x"], df["target_y"], df["target_z"], label="target", color="tab:green", ls="--", alpha=0.8)
    has_filtered = "platform_fx" in df.columns
    if has_filtered:
        ax3d.plot(df["platform_fx"], df["platform_fy"], df["platform_fz"], color="tab:red", ls=":", alpha=0.5, label="plat(f)")
        ax3d.plot(df["target_fx"], df["target_fy"], df["target_fz"], color="tab:green", ls=":", alpha=0.5, label="tgt(f)")

    ax3d.scatter(df["drone_x"].iloc[0], df["drone_y"].iloc[0], df["drone_z"].iloc[0],
                 color="tab:blue", marker="o", s=25, label="start")
    ax3d.scatter(df["drone_x"].iloc[-1], df["drone_y"].iloc[-1], df["drone_z"].iloc[-1],
                 color="tab:blue", marker="x", s=35, label="end")

    hx = float(df["platform_hx"].iloc[0])
    hy = float(df["platform_hy"].iloc[0])
    hz = float(df["platform_hz"].iloc[0])
    _draw_platform_outline(
        ax3d,
        float(df["platform_x"].iloc[0]),
        float(df["platform_y"].iloc[0]),
        float(df["platform_z"].iloc[0]),
        hx, hy, hz,
        alpha=0.4,
    )
    _draw_platform_outline(
        ax3d,
        float(df["platform_x"].iloc[-1]),
        float(df["platform_y"].iloc[-1]),
        float(df["platform_z"].iloc[-1]),
        hx, hy, hz,
        alpha=0.9,
    )

    ax3d.set_title("3D Trajectory")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend(loc="upper right", fontsize=8)
    ax3d.grid(alpha=0.25)
    _set_axes_equal_3d(ax3d)

    t = df["sim_time"]
    axv.plot(t, df["drone_vx"], label="vx")
    axv.plot(t, df["drone_vy"], label="vy")
    axv.plot(t, df["drone_vz"], label="vz")
    axv.plot(t, df["speed_norm"], label="|v|", lw=2.0)
    axv.set_title("Velocity")
    axv.set_xlabel("sim time (s)")
    axv.set_ylabel("m/s")
    axv.grid(alpha=0.25)
    axv.legend(loc="upper right", fontsize=8)

    axd.plot(t, df["dist_3d"], label="dist_3d", color="tab:purple")
    axd.plot(t, df["dist_xy"], label="dist_xy", color="tab:orange")
    axd.set_title("Distance to Target")
    axd.set_xlabel("sim time (s)")
    axd.set_ylabel("m")
    axd.grid(alpha=0.25)
    axd.legend(loc="upper right", fontsize=8)

    axp.plot(t, df["drone_x"], label="x")
    axp.plot(t, df["drone_y"], label="y")
    axp.plot(t, df["drone_z"], label="z")
    axp.set_title("Drone Position")
    axp.set_xlabel("sim time (s)")
    axp.set_ylabel("m")
    axp.grid(alpha=0.25)
    axp.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Trajectory Overview - {episode_name}")
    fig.tight_layout()
    _save(fig, out_dir, f"{episode_name}_overview")


def _plot_combined(dfs: List[pd.DataFrame], names: List[str], out_dir: str) -> None:
    fig = plt.figure(figsize=(18, 5))
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    axv = fig.add_subplot(1, 3, 2)
    axp = fig.add_subplot(1, 3, 3)

    cmap = plt.get_cmap("tab10")
    for i, (df, name) in enumerate(zip(dfs, names)):
        c = cmap(i % 10)
        ax3d.plot(df["drone_x"], df["drone_y"], df["drone_z"], color=c, alpha=0.9, label=name)
        ax3d.plot(df["platform_x"], df["platform_y"], df["platform_z"], color=c, alpha=0.25)
        axv.plot(df["sim_time"], df["speed_norm"], color=c, alpha=0.85, label=name)
        axp.plot(df["sim_time"], df["drone_x"], color=c, alpha=0.35, ls="-")
        axp.plot(df["sim_time"], df["drone_y"], color=c, alpha=0.35, ls="--")
        axp.plot(df["sim_time"], df["drone_z"], color=c, alpha=0.35, ls=":")

    if dfs:
        ref = dfs[0]
        hx = float(ref["platform_hx"].iloc[0])
        hy = float(ref["platform_hy"].iloc[0])
        hz = float(ref["platform_hz"].iloc[0])
        _draw_platform_outline(
            ax3d,
            float(ref["platform_x"].iloc[0]),
            float(ref["platform_y"].iloc[0]),
            float(ref["platform_z"].iloc[0]),
            hx,
            hy,
            hz,
            alpha=0.5,
        )

    ax3d.set_title("3D Trajectory (Combined)")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend(loc="upper right", fontsize=7)
    ax3d.grid(alpha=0.25)
    _set_axes_equal_3d(ax3d)

    axv.set_title("Speed Norm (Combined)")
    axv.set_xlabel("sim time (s)")
    axv.set_ylabel("m/s")
    axv.grid(alpha=0.25)
    axv.legend(loc="upper right", fontsize=7)

    axp.plot([], [], color="black", ls="-", label="x")
    axp.plot([], [], color="black", ls="--", label="y")
    axp.plot([], [], color="black", ls=":", label="z")
    axp.set_title("Drone XYZ (Combined)")
    axp.set_xlabel("sim time (s)")
    axp.set_ylabel("m")
    axp.grid(alpha=0.25)
    axp.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    _save(fig, out_dir, "combined_overview")


def plot_trajectory_directory(stage_dir: str, per_episode: bool = True,
                              combined: bool = True, show: bool = False) -> None:
    """绘制 evaluate.py 创建的某个阶段目录下的轨迹。"""
    episode_csvs = sorted(glob.glob(os.path.join(stage_dir, "episode_*", "trajectory.csv")))
    if not episode_csvs:
        print(f"[WARN] No trajectory CSV found in {stage_dir}")
        return

    figures_dir = os.path.join(stage_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    dfs: List[pd.DataFrame] = []
    names: List[str] = []
    for csv_path in episode_csvs:
        df = pd.read_csv(csv_path)
        ep_name = os.path.basename(os.path.dirname(csv_path))
        dfs.append(df)
        names.append(ep_name)
        if per_episode:
            _plot_episode(df, figures_dir, ep_name)

    if combined:
        _plot_combined(dfs, names, figures_dir)

    if show:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot evaluation trajectories from stage directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage_dir", type=str, required=True)
    p.add_argument("--per_episode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--combined", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plot_trajectory_directory(
        stage_dir=args.stage_dir,
        per_episode=args.per_episode,
        combined=args.combined,
        show=args.show,
    )


if __name__ == "__main__":
    main()
