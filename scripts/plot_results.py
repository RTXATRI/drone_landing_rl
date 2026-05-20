#!/usr/bin/env python3
# 功能：读取训练 CSV 日志并生成奖励、成功率和训练指标图表。
"""
训练后的分析与绘图工具。

读取训练期间导出的 CSV 文件，生成适合论文或 MATLAB 导入的 PDF + PNG 图表。

生成的图：
  1. 学习曲线（奖励随时间变化，按阶段着色）
  2. 各阶段滚动成功率
  3. 奖励分项拆解（堆叠面积图）
  4. 训练过程中的 episode 长度
  5. 训练损失曲线（actor/critic/entropy）
  6. 各阶段到目标距离分布（violin plot）

用法：
    python scripts/plot_results.py --csv_dir output/data/csv/drone_landing
    python scripts/plot_results.py --csv_dir output/data/csv/drone_landing --show
    python scripts/plot_results.py --csv_dir output/data/csv/exp1 --out_dir figs/exp1
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # 无显示环境安全；可用 --show 覆盖
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── 样式 ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":       150,
})

STAGE_COLORS = {1: "#4C72B0", 2: "#55A868", 3: "#C44E52", 4: "#DD8452"}
try:
    from curriculum.strategies import STAGE_SHORT_LABELS
    STAGE_LABELS = {
        stage: f"Stage {stage}: {label}"
        for stage, label in STAGE_SHORT_LABELS.items()
    }
except Exception:
    STAGE_LABELS = {stage: f"Stage {stage}" for stage in STAGE_COLORS}
REWARD_COLS = [
    ("reward_dist",     "#4C72B0", "Distance"),
    ("reward_smooth",   "#DD8452", "Smoothness"),
    ("reward_alive",    "#55A868", "Alive"),
    ("reward_vel_match","#C44E52", "Vel Match"),
    ("reward_yaw",      "#8172B2", "Yaw"),
    ("reward_approach", "#937860", "Approach"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"  [WARN] Not found: {path}")
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {os.path.basename(path)}: {len(df):,} rows")
    return df


def _moving_avg(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 图表生成函数
# ─────────────────────────────────────────────────────────────────────────────

def fig_reward_curve(ep: pd.DataFrame, out: str, window: int = 100) -> None:
    """训练过程中的平滑 episode reward，按课程阶段着色。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    for stage, color in STAGE_COLORS.items():
        sub = ep[ep["stage"] == stage]
        if sub.empty:
            continue
        ax.scatter(sub["timestep"], sub["reward"],
                   s=1, alpha=0.15, color=color, zorder=1)
        ma = _moving_avg(sub["reward"], window)
        ax.plot(sub["timestep"], ma, color=color, lw=2,
                label=STAGE_LABELS[stage], zorder=2)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Episode Return")
    ax.set_title("Training Reward Curve (Smoothed, MA-100)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out, "reward_curve")


def fig_success_rate(ep: pd.DataFrame, out: str, window: int = 50) -> None:
    """各阶段滚动成功率。"""
    fig, ax = plt.subplots(figsize=(10, 4))

    for stage, color in STAGE_COLORS.items():
        sub = ep[ep["stage"] == stage]
        if sub.empty:
            continue
        sr = _moving_avg(sub["success"].astype(float), window)
        ax.plot(sub["timestep"], sr, color=color, lw=2.0,
                label=STAGE_LABELS[stage])

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Rolling Success Rate (MA-{window})")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out, "success_rate")


def fig_episode_length(ep: pd.DataFrame, out: str, window: int = 100) -> None:
    """训练过程中的平滑 episode 长度。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    for stage, color in STAGE_COLORS.items():
        sub = ep[ep["stage"] == stage]
        if sub.empty:
            continue
        ma = _moving_avg(sub["length"], window)
        ax.plot(sub["timestep"], ma, color=color, lw=2,
                label=STAGE_LABELS[stage])

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_title("Episode Length over Training (MA-100)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out, "episode_length")


def fig_reward_components(rw: pd.DataFrame, out: str, downsample: int = 200) -> None:
    """奖励分项贡献的堆叠图。"""
    if rw is None:
        return

    # 下采样，避免图太密
    rw_ds = rw.iloc[::downsample].copy()
    rw_ds = rw_ds.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(rw_ds))

    for col, color, label in REWARD_COLS:
        if col not in rw_ds.columns:
            continue
        vals = rw_ds[col].fillna(0).clip(upper=0).values * -1  # show magnitude
        ax.bar(rw_ds.index, vals, bottom=bottom,
               color=color, label=label, alpha=0.75, width=1.0)
        bottom += vals

    ax.set_xlabel("Sample Index (downsampled)")
    ax.set_ylabel("|Reward Component| Magnitude")
    ax.set_title("Reward Component Breakdown (Penalty Magnitudes)")
    ax.legend(loc="upper right", ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out, "reward_components")


def fig_reward_trends(rw: pd.DataFrame, out: str, window: int = 2000) -> None:
    """训练过程中各奖励分项的趋势。"""
    if rw is None:
        return

    n_cols = len(REWARD_COLS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    axes_flat = axes.flatten()

    for i, (col, color, label) in enumerate(REWARD_COLS):
        ax = axes_flat[i]
        if col not in rw.columns:
            ax.set_visible(False)
            continue
        ma = _moving_avg(rw[col], window)
        ax.plot(rw["timestep"], ma, color=color, lw=1.5)
        ax.set_title(label)
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Timestep")

    fig.suptitle("Reward Component Trends (MA-2000)", fontsize=14)
    fig.tight_layout()
    _save(fig, out, "reward_trends")


def fig_training_loss(tr: pd.DataFrame, out: str) -> None:
    """Actor/critic/entropy coefficient 曲线。"""
    if tr is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col, label, color in [
        (axes[0], "actor_loss",  "Actor Loss",   "#4C72B0"),
        (axes[1], "critic_loss", "Critic Loss",  "#C44E52"),
        (axes[2], "ent_coef",    "Entropy Coef", "#55A868"),
    ]:
        if col not in tr.columns:
            ax.set_visible(False)
            continue
        ax.plot(tr["timestep"], tr[col], color=color, lw=1.5, alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("Timestep")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, out, "training_loss")


def fig_dist_violin(rw: pd.DataFrame, out: str) -> None:
    """各阶段到目标距离的分布（violin plot）。"""
    if rw is None or "metric_dist" not in rw.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    stage_data = []
    positions  = []
    labels     = []
    for stage in [1, 2, 3, 4]:
        sub = rw[rw["stage"] == stage]["metric_dist"].dropna()
        if len(sub) > 10:
            stage_data.append(sub.values)
            positions.append(stage)
            labels.append(f"S{stage}")

    if stage_data:
        parts = ax.violinplot(stage_data, positions=positions,
                              showmedians=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            stage = positions[i]
            pc.set_facecolor(STAGE_COLORS.get(stage, "gray"))
            pc.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

    ax.set_ylabel("Distance to Target (m)")
    ax.set_xlabel("Curriculum Stage")
    ax.set_title("Distance-to-Target Distribution by Stage")
    ax.grid(axis="y", alpha=0.3)
    legend_patches = [mpatches.Patch(color=c, label=STAGE_LABELS[s])
                      for s, c in STAGE_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, out, "dist_violin")


def fig_summary_dashboard(ep: pd.DataFrame, tr: pd.DataFrame, out: str) -> None:
    """适合论文使用的 4 面板总结图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes_flat = axes.flatten()

    # 面板 A：奖励曲线
    ax = axes_flat[0]
    for stage, color in STAGE_COLORS.items():
        sub = ep[ep["stage"] == stage]
        if sub.empty:
            continue
        ma = _moving_avg(sub["reward"], 100)
        ax.plot(sub["timestep"], ma, color=color, lw=2, label=f"S{stage}")
    ax.set_title("(A) Episode Return (MA-100)")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Return")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    # 面板 B：成功率
    ax = axes_flat[1]
    ax.axhline(0.8, color="gray", ls="--", lw=1)
    for stage, color in STAGE_COLORS.items():
        sub = ep[ep["stage"] == stage]
        if sub.empty:
            continue
        sr = _moving_avg(sub["success"].astype(float), 50)
        ax.plot(sub["timestep"], sr, color=color, lw=2, label=f"S{stage}")
    ax.set_title("(B) Success Rate (MA-50)")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Rate")
    ax.set_ylim(-0.05, 1.05); ax.legend(loc="upper left"); ax.grid(alpha=0.3)

    # 面板 C：回合长度
    ax = axes_flat[2]
    for stage, color in STAGE_COLORS.items():
        sub = ep[ep["stage"] == stage]
        if sub.empty:
            continue
        ma = _moving_avg(sub["length"], 100)
        ax.plot(sub["timestep"], ma, color=color, lw=2, label=f"S{stage}")
    ax.set_title("(C) Episode Length (MA-100)")
    ax.set_xlabel("Timestep"); ax.set_ylabel("Steps")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)

    # 面板 D：训练损失
    ax = axes_flat[3]
    if tr is not None and "actor_loss" in tr.columns:
        ax.plot(tr["timestep"], tr["actor_loss"],  lw=1.5, label="Actor",  color="#4C72B0")
        ax.plot(tr["timestep"], tr["critic_loss"], lw=1.5, label="Critic", color="#C44E52")
        ax.set_title("(D) Loss Curves")
        ax.set_xlabel("Timestep"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "training_log.csv\nnot available",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title("(D) Loss Curves")

    fig.suptitle("Training Summary Dashboard", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "dashboard")


# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Post-training analysis & plot generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv_dir", type=str, required=True,
                   help="Directory containing episode_log.csv etc.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory for figures (default: csv_dir/figures)")
    p.add_argument("--show", action="store_true", help="Display plots interactively")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.csv_dir, "figures")
    if args.show:
        matplotlib.use("TkAgg")

    print(f"\nLoading data from: {args.csv_dir}")
    ep = _load(os.path.join(args.csv_dir, "episode_log.csv"))
    rw = _load(os.path.join(args.csv_dir, "reward_log.csv"))
    tr = _load(os.path.join(args.csv_dir, "training_log.csv"))

    if ep is None:
        print("ERROR: episode_log.csv not found — cannot generate plots.")
        sys.exit(1)

    print(f"\nGenerating figures → {out_dir}")
    fig_reward_curve(ep, out_dir)
    fig_success_rate(ep, out_dir)
    fig_episode_length(ep, out_dir)
    fig_reward_components(rw, out_dir)
    fig_reward_trends(rw, out_dir)
    fig_training_loss(tr, out_dir)
    fig_dist_violin(rw, out_dir)
    fig_summary_dashboard(ep, tr, out_dir)

    print(f"\nAll figures saved to: {out_dir}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
