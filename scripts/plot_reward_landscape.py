#!/usr/bin/env python3
# 功能：可视化课程一、二的位置相关奖励函数 3D 地形图和 2D 切片图。
"""
奖励函数可视化：位置奖励地形图和切片图。

修改脚本顶部的 STAGE / VIEW 配置即可选择显示内容。
3D 图支持鼠标拖拽旋转 + 滚轮缩放。
"""

import numpy as np
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════

STAGE = 2          # 要显示的课程：0=全部, 1=课程一, 2=课程二
VIEW  = "both"     # 视图范围："far" / "near" / "both"


# ══════════════════════════════════════════════════════════════════
# 通用工具
# ══════════════════════════════════════════════════════════════════

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _on_scroll(event):
    """滚轮缩放回调：缩放坐标轴范围。"""
    ax = event.inaxes
    if ax is None:
        return
    scale = 0.9 if event.button == "up" else 1.1
    for getter, setter in [
        (ax.get_xlim3d, ax.set_xlim3d),
        (ax.get_ylim3d, ax.set_ylim3d),
        (ax.get_zlim3d, ax.set_zlim3d),
    ]:
        lo, hi = getter()
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 * scale
        setter(mid - half, mid + half)
    event.canvas.draw_idle()


# ══════════════════════════════════════════════════════════════════
# Stage 1 — 参数 & 公式
# ══════════════════════════════════════════════════════════════════

# ── 位置奖励：高斯接近 ──
POS_XY_WEIGHT = 3.0           # XY 位置奖励权重
POS_XY_SIGMA  = 0.25          # XY 高斯 σ (m)
POS_Z_WEIGHT  = 3.5           # Z  位置奖励权重
POS_Z_SIGMA   = 0.20          # Z  高斯 σ (m)

# ── 速度门控 ──
VEL_GATE_XY_INNER = 0.25      # 全惩罚内阈值 (m)
VEL_GATE_XY_OUTER = 0.75      # 零惩罚外阈值 (m)
VEL_GATE_Z_INNER  = 0.20      # 全惩罚内阈值 (m)
VEL_GATE_Z_OUTER  = 0.60      # 零惩罚外阈值 (m)


def stage1_r_pos_xy(h, v):
    return POS_XY_WEIGHT * np.exp(-(h ** 2) / (2.0 * POS_XY_SIGMA ** 2))

def stage1_r_pos_z(h, v):
    return POS_Z_WEIGHT * np.exp(-(v ** 2) / (2.0 * POS_Z_SIGMA ** 2))

def stage1_r_pos_total(h, v):
    return stage1_r_pos_xy(h, v) + stage1_r_pos_z(h, v)

def stage1_vel_gate(h, v):
    g_xy = smoothstep(
        (VEL_GATE_XY_OUTER - h) / (VEL_GATE_XY_OUTER - VEL_GATE_XY_INNER)
    )
    g_z = smoothstep(
        (VEL_GATE_Z_OUTER - v) / (VEL_GATE_Z_OUTER - VEL_GATE_Z_INNER)
    )
    return g_xy * g_z


# ══════════════════════════════════════════════════════════════════
# Stage 2 — 参数 & 公式
# ══════════════════════════════════════════════════════════════════

# ── r_approach：反二次（长尾） ──
POS_APPROACH_WEIGHT   = 3.0   # 峰值权重
POS_APPROACH_SIGMA_XY = 5.0   # 水平半衰半径 (m)
POS_APPROACH_SIGMA_Z  = 3.0   # 垂直半衰半径 (m)

# ── r_precise：高斯（中距精度） ──
AGS_1_WEIGHT   = 1.5    # 近距精度权重
AGS_1_SIGMA_XY = 0.5    # 近距水平 σ (m)
AGS_1_SIGMA_Z  = 0.5    # 近距垂直 σ (m)

# ── r_peak：窄高斯（近距峰值） ──
AGS_2_WEIGHT   = 1.5       # 峰值权重
AGS_2_SIGMA_XY = 0.20      # 峰值水平 σ (m)
AGS_2_SIGMA_Z  = 0.20      # 峰值垂直 σ (m)

AGS_3_WEIGHT   = 1.0       # 峰值权重
AGS_3_SIGMA_XY = 0.05      # 峰值水平 σ (m)
AGS_3_SIGMA_Z  = 0.05      # 峰值垂直 σ (m)

# ── vel_match gate ──
VEL_MATCH_GATE_INNER = 0.10   # 全惩罚内阈值 (m)
VEL_MATCH_GATE_OUTER = 0.50   # 零惩罚外阈值 (m)

# 二元各向异性柯西加权函数模板
def r_anisotropic_cauchy(h, v, weight, sigma_xy, sigma_z):
    return weight / ( 1.0 + (h / sigma_xy) ** 2 + (v / sigma_z) ** 2 )

# 各向异性高斯函数模板
def r_anisotropic_gaussian(h, v, weight, sigma_xy, sigma_z):
    return weight * np.exp( -(h ** 2) / (2.0 * sigma_xy ** 2) - (v ** 2) / (2.0 * sigma_z ** 2))

def stage2_r_pos_total(h, v):
    return (r_anisotropic_cauchy(h, v, POS_APPROACH_WEIGHT, POS_APPROACH_SIGMA_XY, POS_APPROACH_SIGMA_Z)
            + r_anisotropic_gaussian(h, v, AGS_1_WEIGHT, AGS_1_SIGMA_XY, AGS_1_SIGMA_Z)
            + r_anisotropic_gaussian(h, v, AGS_2_WEIGHT, AGS_2_SIGMA_XY, AGS_2_SIGMA_Z)
            + r_anisotropic_gaussian(h, v, AGS_3_WEIGHT, AGS_3_SIGMA_XY, AGS_3_SIGMA_Z)
            )

def stage2_vel_match_gate(h):
    return smoothstep(
        (VEL_MATCH_GATE_OUTER - h)
        / (VEL_MATCH_GATE_OUTER - VEL_MATCH_GATE_INNER)
    )


# ══════════════════════════════════════════════════════════════════
# 课程注册表
# ══════════════════════════════════════════════════════════════════

STAGE_CONFIGS = {
    1: {
        "label": "Stage 1 — Hover Static",
        "func": stage1_r_pos_total,
        "cmap": "viridis",
        "near_range": 3.0,
        "far_range": 10.0,
    },
    2: {
        "label": "Stage 2 — Hover Moving",
        "func": stage2_r_pos_total,
        "cmap": "viridis",
        "near_range": 3.0,
        "far_range": 10.0,
        "horiz_marks": (0.01, 0.03, 0.05, 0.10, 0.20, 0.30),
        "vert_marks": (0.01, 0.03, 0.05, 0.08, 0.10, 0.20),
    },
}


# ══════════════════════════════════════════════════════════════════
# 绘图
# ══════════════════════════════════════════════════════════════════

GRID_RES = 100
SLICE_RES = 500


def plot_stage(config, view):
    """为指定课程生成 r_pos 3D 表面图（远距+近距子图）。"""
    views = []
    if view in ("far", "both"):
        views.append(("Far", config["far_range"]))
    if view in ("near", "both"):
        views.append(("Near", config["near_range"]))

    ncols = len(views)
    fig = plt.figure(figsize=(7 * ncols, 7))
    fig.suptitle(f'{config["label"]} — r_pos', fontsize=13)

    for col, (view_name, max_range) in enumerate(views):
        h = np.linspace(0, max_range, GRID_RES)
        v = np.linspace(0, max_range, GRID_RES)
        H, V = np.meshgrid(h, v)
        Z = config["func"](H, V)

        ax = fig.add_subplot(1, ncols, col + 1, projection="3d")
        ax.plot_surface(H, V, Z, cmap=config["cmap"], alpha=0.85, edgecolor="none")
        ax.set_xlabel("horiz_err (m)")
        ax.set_ylabel("vert_err (m)")
        ax.set_zlabel("Reward")
        ax.set_title(f"{view_name} view (0–{max_range:.0f}m)", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.mpl_connect("scroll_event", _on_scroll)


def _slice_views(config, view):
    """二维切片范围：both 同时显示近距和远距。"""
    if view == "near":
        return [("Near", config["near_range"])]
    if view == "far":
        return [("Far", config["far_range"])]
    return [
        ("Near", config["near_range"]),
        ("Far", config["far_range"]),
    ]


def _add_reference_lines(ax, marks):
    """为 Stage 2 切片图添加关键误差参考线。"""
    x_min, x_max = ax.get_xlim()
    for mark in marks:
        if x_min <= mark <= x_max:
            ax.axvline(mark, color="gray", linestyle=":", linewidth=0.8, alpha=0.45)
            ax.text(
                mark,
                0.98,
                f"{mark:g}",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                ha="right",
                va="top",
                fontsize=8,
                color="gray",
            )


def plot_stage_slices(config, view):
    """绘制 r_pos(h, 0) 与 r_pos(0, v) 两张二维切片图。"""
    views = _slice_views(config, view)
    nrows = len(views)
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 5 * nrows), squeeze=False)
    fig.suptitle(
        f'{config["label"]} — r_pos 2D slices',
        fontsize=13,
    )

    for row, (view_name, max_range) in enumerate(views):
        err = np.linspace(0.0, max_range, SLICE_RES)
        zeros = np.zeros_like(err)
        horiz_slice = config["func"](err, zeros)
        vert_slice = config["func"](zeros, err)

        ax_h = axes[row, 0]
        ax_h.plot(err, horiz_slice, color="tab:blue", linewidth=2.0)
        ax_h.set_xlabel("horiz_err (m), vert_err = 0")
        ax_h.set_ylabel("Reward")
        ax_h.set_title(
            f"Horizontal slice: r_pos(h, 0), {view_name} 0–{max_range:.0f}m",
            fontsize=10,
        )
        ax_h.grid(True, alpha=0.3)
        _add_reference_lines(ax_h, config.get("horiz_marks", ()))

        ax_v = axes[row, 1]
        ax_v.plot(err, vert_slice, color="tab:orange", linewidth=2.0)
        ax_v.set_xlabel("vert_err (m), horiz_err = 0")
        ax_v.set_ylabel("Reward")
        ax_v.set_title(
            f"Vertical slice: r_pos(0, v), {view_name} 0–{max_range:.0f}m",
            fontsize=10,
        )
        ax_v.grid(True, alpha=0.3)
        _add_reference_lines(ax_v, config.get("vert_marks", ()))

    fig.tight_layout(rect=[0, 0, 1, 0.92])


# ══════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════

def main():
    # 合法性检查
    valid_stages = set(STAGE_CONFIGS.keys()) | {0}
    if STAGE not in valid_stages:
        raise ValueError(f"STAGE={STAGE} 不合法，可选：{sorted(valid_stages)}")
    if VIEW not in ("far", "near", "both"):
        raise ValueError(f'VIEW="{VIEW}" 不合法，可选：far / near / both')

    stages = sorted(STAGE_CONFIGS.keys()) if STAGE == 0 else [STAGE]

    for sid in stages:
        print(f"Stage {sid}: {STAGE_CONFIGS[sid]['label']}")
        plot_stage(STAGE_CONFIGS[sid], VIEW)
        plot_stage_slices(STAGE_CONFIGS[sid], VIEW)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
