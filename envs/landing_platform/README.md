# envs/landing_platform 目录说明

本目录保存降落平台运动模型，用于生成静止或移动目标。

## 文件职责

- `moving_platform.py`：根据外部策略设置的运动模式生成平台位置、速度、姿态、角速度和检测质量。

## 当前运动模式

| 模式 | 说明 |
| --- | --- |
| `static` | 平台静止 |
| `linear` | X 方向正弦往复 |
| `sinusoidal` | X/Y 独立正弦运动 |
| `figure8` | 8 字形轨迹 |

## 迁移提示

真实 ROS/Gazebo/PX4 工程中，平台状态应来自 Gazebo model state、TF、平台 odom、视觉检测或真实传感器。只要输出字段与 `BaseDroneLandingEnv` 需要的状态字典一致，观测构造逻辑就可以继续复用。
