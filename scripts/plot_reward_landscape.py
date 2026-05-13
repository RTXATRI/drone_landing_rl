#!/usr/bin/env python3
# 功能：可视化课程一、二的位置相关奖励函数 3D 地形图。
"""
奖励函数 3D 可视化：位置奖励地形图。

修改脚本顶部的 STAGE / VIEW 配置即可选择显示内容。
支持鼠标拖拽旋转 + 滚轮缩放。
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
POS_APPROACH_SIGMA_XY = 8.0   # 水平半衰半径 (m)
POS_APPROACH_SIGMA_Z  = 5.0   # 垂直半衰半径 (m)

# ── r_precise：高斯（中距精度） ──
POS_PRECISE_WEIGHT   = 4.0    # 近距精度权重
POS_PRECISE_SIGMA_XY = 0.9    # 近距水平 σ (m)
POS_PRECISE_SIGMA_Z  = 0.5    # 近距垂直 σ (m)

# ── r_peak：窄高斯（近距峰值） ──
POS_PEAK_WEIGHT   = 2.0       # 峰值权重
POS_PEAK_SIGMA_XY = 0.15      # 峰值水平 σ (m)
POS_PEAK_SIGMA_Z  = 0.10      # 峰值垂直 σ (m)

# ── vel_match gate ──
VEL_MATCH_GATE_INNER = 0.15   # 全惩罚内阈值 (m)
VEL_MATCH_GATE_OUTER = 0.50   # 零惩罚外阈值 (m)


def stage2_r_approach(h, v):
    return POS_APPROACH_WEIGHT / (
        1.0
        + (h / POS_APPROACH_SIGMA_XY) ** 2
        + (v / POS_APPROACH_SIGMA_Z) ** 2
    )

def stage2_r_precise(h, v):
    return POS_PRECISE_WEIGHT * np.exp(
        -(h ** 2) / (2.0 * POS_PRECISE_SIGMA_XY ** 2)
        - (v ** 2) / (2.0 * POS_PRECISE_SIGMA_Z ** 2)
    )

def stage2_r_peak(h, v):
    return POS_PEAK_WEIGHT * np.exp(
        -(h ** 2) / (2.0 * POS_PEAK_SIGMA_XY ** 2)
        - (v ** 2) / (2.0 * POS_PEAK_SIGMA_Z ** 2)
    )

def stage2_r_pos_total(h, v):
    return stage2_r_approach(h, v) + stage2_r_precise(h, v) + stage2_r_peak(h, v)

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
    },
}


# ══════════════════════════════════════════════════════════════════
# 绘图
# ══════════════════════════════════════════════════════════════════

GRID_RES = 100


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

    plt.show()


if __name__ == "__main__":
    main()
