# envs/landing_platform 目录说明

本目录保存降落平台运动模型，通过组合模式注入运动策略，生成静止或移动目标。

## 文件职责

| 文件 | 职责 |
|------|------|
| `moving_platform.py` | 平台模型聚合根，持有运动策略、速度控制器和边界裁剪。通过组合模式注入 `MotionStrategy` 实例。 |
| `speed_controller.py` | 通用速度控制器，多正弦叠加 + 一阶低通滤波，生成无突变标量速度序列。仅课程二激活。 |
| `motions/` | 运动策略包，每种策略实现一种轨迹形状。 |

## 运动策略

| 策略 | 文件 | 类型 | 活动范围 | 使用课程 |
|------|------|------|----------|----------|
| `StaticMotion` | `motions/static_motion.py` | 静止 | 原点 (0,0) | Stage 1 |
| `LissajousMotion` | `motions/lissajous_motion.py` | 闭合曲线路径累积 | 幅值 15-35m（X）/ 10-30m（Y），每 ep 随机 | Stage 2/3/4 |
| `PatrolMotion` | `motions/patrol_motion.py` | 路径累积 | 原点 ↔ 随机目标（15-40m），含波浪偏移 ≤1.2m | Stage 2 |
| `WaypointMotion` | `motions/waypoint_motion.py` | 路径累积 | 随机多边形航点 ∈ [-50,50]²，边距 ≥10m | Stage 2 |

### 策略类型说明

- **路径累积**：位置沿路径按 `speed * dt` 推进，速度来自本步位置差分。Stage 2 的移动策略都必须消费 `SpeedController` 提供的标量速度。
- **连续性要求**：运动策略需要保证 reset 初始位置、航点/端点换段、闭合曲线循环处的坐标连续；`MovingPlatform.reset()` 会同步到策略当前初始状态，避免第一步出现假跳变。

### 组合模式

课程策略通过 `MovingPlatform.set_motion_strategy()` 注入运动策略，在 episode reset 前调用。课程二额外调用 `enable_speed_controller()` 和 `set_boundary(50.0)` 激活统一速度控制和边界裁剪。

### 速度系统

`MotionStrategy.step()` 返回 `(pos_xy, vel_xy)`，其中 `vel_xy` 为完整速度向量（m/s）。Stage 2 的移动策略使用 `SpeedController` 输出的标量速度推进，`MovingPlatform` 直接使用返回值，不做速度缩放。当前速度控制器输出约 `0.3-4.5m/s`，测试要求位置差分速度和报告速度不超过 `5m/s`。

## 迁移提示

真实 ROS/Gazebo/PX4 工程中，平台状态应来自 Gazebo model state、TF、平台 odom、视觉检测或真实传感器。只要输出字段与 `BaseDroneLandingEnv` 需要的状态字典一致，观测构造逻辑就可以继续复用。
