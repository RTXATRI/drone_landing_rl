# 功能：定义无人机降落环境的物理参数、观测归一化参数和奖励权重。
"""
环境配置 dataclass。

所有物理常量和环境参数都集中在这里。
调环境参数时，优先只修改这个文件。
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DroneConfig:
    """无人机运动学限制和物理属性。"""
    # 速度限制 (m/s)
    max_vx: float = 2.0
    max_vy: float = 2.0
    max_vz_up: float = 1.0
    max_vz_down: float = 1.0
    max_yaw_rate: float = 1.0  # rad/s

    # 指令平滑（一阶低通滤波系数）
    # 0.0 = 不平滑（瞬时响应），1.0 = 完全平滑（无响应）
    # 建议设置在约 0.6-0.8，用于抑制高频指令抖动
    cmd_filter_alpha: float = 0.7

    # 可视化/碰撞几何尺寸（半长，单位 m）
    half_extents: Tuple[float, float, float] = (0.25, 0.25, 0.08)


@dataclass
class DisturbanceConfig:
    """运动学预训练中的外部扰动和速度跟踪误差。"""
    enabled: bool = True

    # 每个 episode 固定的水平漂移，用于模拟弱稳态风/水流。
    base_wind_speed_min: float = 0.00  # m/s
    base_wind_speed_max: float = 0.08  # m/s

    # 间歇性水平阵风；每步会根据 dt 换算触发概率。
    gust_prob_per_second: float = 0.08
    gust_speed_min: float = 0.05       # m/s
    gust_speed_max: float = 0.22       # m/s
    gust_duration_min: float = 1.0     # s
    gust_duration_max: float = 4.0     # s
    gust_smooth_prob: float = 0.5      # 否则使用脉冲衰减包络

    # 小幅速度跟踪波动，只作用于 XYZ 指令速度。
    tracking_error_enabled: bool = True
    tracking_scale_min: float = 0.98
    tracking_scale_max: float = 1.02


@dataclass
class ObservationConfig:
    """观测归一化、裁剪和随机能力范围。"""
    # 归一化分母
    rel_xy_norm: float = 20.0
    rel_z_norm: float = 10.0
    range_norm: float = 20.0
    height_norm: float = 30.0
    v_xy_norm: float = 20.0
    v_z_up_norm: float = 5.0
    v_z_down_norm: float = 5.0
    roll_rate_norm: float = 1.0
    pitch_rate_norm: float = 1.0
    yaw_rate_norm: float = 1.0

    # 每个 episode 随机化的无人机能力范围
    v_xy_min: float = 8.0
    v_xy_max: float = 20.0
    v_z_up_min: float = 1.0
    v_z_up_max: float = 5.0
    v_z_down_min: float = 1.0
    v_z_down_max: float = 5.0

    # 归一化后的裁剪限制
    clip_signed: float = 1.5
    clip_angle: float = 1.0
    clip_rate: float = 1.0
    clip_nonnegative: float = 1.5
    clip_capability: float = 1.2


@dataclass
class PlatformConfig:
    """降落平台几何尺寸和运动参数。"""
    # 几何尺寸（半长，单位 m）
    half_extents: Tuple[float, float, float] = (0.6, 0.6, 0.08)

    # 降落成功容差
    landing_radius: float = 0.55   # 水平容差 (m)
    landing_height_tol: float = 0.20  # 平台中心上方垂直容差 (m)
    landing_speed_max: float = 0.35   # 接触瞬间最大速度 (m/s)

    # 可被课程策略选择使用的平台运动参数
    max_speed: float = 1.0           # m/s（用于归一化）
    motion_amplitude_x: float = 2.0  # m
    motion_amplitude_y: float = 1.0  # m
    motion_frequency: float = 0.15   # Hz


@dataclass
class EpisodeConfig:
    """Episode 生命周期参数。"""
    max_steps: int = 3600
    dt: float = 0.05          # control timestep (s) → 20 Hz

    # 无人机初始化：围绕平台的圆柱随机出生范围
    init_spawn_radius_min: float = 3.0    # 最小水平半径 (m)
    init_spawn_radius_max: float = 35.0   # 最大水平半径 (m)
    init_spawn_height_min: float = 1.0    # 最小出生高度 (m)
    init_spawn_height_max: float = 30.0   # 最大出生高度 (m)

    # 终止边界
    max_horiz_dist: float = 45.0   # 无人机水平漂移超过该距离则结束 episode
    min_height: float = -0.3       # 无人机低于地面则结束 episode

    # 可被悬停类策略使用的目标高度范围（每个 episode 随机）
    hover_height_min: float = 1.0      # 最小悬停目标高度 (m)
    hover_height_max: float = 10.0     # 最大悬停目标高度 (m)


@dataclass
class EnvConfig:
    """顶层环境配置。"""
    drone: DroneConfig = field(default_factory=DroneConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
