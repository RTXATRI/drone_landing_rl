# configs 目录说明

本目录集中保存工程配置，目的是让环境参数和训练参数可以在不改核心逻辑的情况下调整。

## 文件职责

- `env_config.py`：环境配置，包括无人机速度限制、轻量扰动、平台尺寸、episode 生命周期、观测归一化/裁剪参数。
- `train_config.py`：训练配置，包括 SAC 超参数、并行环境数、`output/` 下的日志/模型/CSV 路径、控制台输出频率、checkpoint 频率、手动课程阶段预算和统计窗口。

## 调参入口

常见环境调参优先修改 `EnvConfig`：

- 观测维度和归一化：`ObservationConfig`
- PyBullet 预训练扰动：`DisturbanceConfig`
- 降落平台尺寸和降落容差：`PlatformConfig`
- episode 时长、初始出生范围和出界边界：`EpisodeConfig`

常见训练调参优先修改 `TrainConfig`：

- SAC 学习率、batch size、buffer size：`SACConfig`
- 成功率窗口和阶段预算：`CurriculumConfig`
- 实验名、输出目录、SB3 主表格/课程短行/checkpoint 频率：`TrainConfig`

## 迁移提示

未来接入 ROS1 + MAVROS + PX4 / Gazebo 时，应尽量保持观测归一化参数和动作速度上限在这里统一管理。奖励常量由具体课程策略自行维护。

`DisturbanceConfig` 只服务当前 Python/PyBullet 预训练后端，用于模拟弱水平漂移、间歇阵风和小幅速度跟踪误差。真实系统中这些效果应自然体现在飞控/里程计反馈的实际速度里，不建议在部署端重复叠加同一套扰动。
