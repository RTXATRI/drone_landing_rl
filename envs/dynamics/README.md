# envs/dynamics 目录说明

本目录保存无人机简化运动学模型。

## 文件职责

- `kinematic_model.py`：根据速度指令积分无人机位置和 yaw，通过一阶低通滤波抑制动作突变，并在 PyBullet 预训练中加入轻量水平风和速度跟踪误差。

## 当前假设

当前模型不模拟真实气动、电机、重力或 PX4 内环，仅用于高层速度控制策略的轻量预训练。为了避免悬停任务过于理想，当前模型会把滤波后的命令速度转换为实际速度：

```text
actual_velocity = filtered_command_velocity * tracking_scale + wind_velocity
```

- `tracking_scale`：每 step、每个 XYZ 分量独立采样，默认 `0.98-1.02`。
- `wind_velocity`：每回合固定弱水平漂移 + 随机间歇水平阵风。
- 当前版本不加入垂直风，yaw_rate 也不加入跟踪误差。

`KinematicDroneModel.velocity` 和环境返回的 `drone_state["velocity"]` 都表示这个实际速度。

## 迁移提示

迁移到 ROS/Gazebo/PX4 后，这里的运动学积分通常会被真实仿真器或飞控状态反馈替代。部署端仍应把策略输出视为速度指令，把飞控/里程计反馈视为实际速度；低通滤波思想可作为部署端安全平滑层的一部分。
