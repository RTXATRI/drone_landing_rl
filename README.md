# Drone Landing RL 工程总览

此项目还在开发中，目前未完成，最后更新时间：2026-05-08

本文件是工程根目录的总入口说明文档。各主要源码目录内也有中文 `README.md`，用于解释该目录的代码职责和详细参考信息。

## 1. 项目概述

这是一个用 SAC 强化学习训练无人机在静止/移动平台上悬停、跟踪和降落的框架。当前仿真后端是 PyBullet，环境接口基于 Gymnasium，训练由 Stable-Baselines3 驱动。

核心能力：

- 34 维归一化观测，描述目标相对位姿、无人机实际状态、平台状态、检测质量、当前速度能力和上一时刻动作。
- 4 维连续动作，输出机体系速度指令和 yaw rate。
- PyBullet 预训练后端加入轻量水平风、间歇阵风和速度跟踪误差，使悬停不再是完全理想静止问题。
- 基于策略模式的课程学习；每个课程独立定义场景、奖励、成功条件和评估指标。
- 奖励函数、课程逻辑、训练流程和仿真后端解耦，便于从 PyBullet 迁移到 ROS/Gazebo/PX4。
- 支持 TensorBoard、CSV、checkpoint、轨迹采集、离线/实时绘图和模型导出。

环境：`conda activate drone_rl`（Python 3.13.13, PyTorch 2.7.1+cu128, RTX 4070 Ti SUPER）。

## 2. 技术栈

| 模块 | 技术 |
| --- | --- |
| 强化学习 | Stable-Baselines3 SAC |
| 环境接口 | Gymnasium |
| 仿真/渲染 | PyBullet |
| 神经网络 | PyTorch / CUDA |
| 并行采样 | SubprocVecEnv |
| 日志监控 | TensorBoard, CSV |
| 数据分析 | Python, MATLAB 脚本 |

## 3. 快速开始

```powershell
conda activate drone_rl
pip install -r requirements.txt

# 环境自检
python scripts/test_env.py

# 冒烟训练（CPU，快速验证流程）
python scripts/train.py --n_envs 4 --total_steps 20000 --exp_name smoke_test --device cpu

# 启动 TensorBoard 监控
tensorboard --logdir output/logs

# 评估模型
python scripts/evaluate.py --model output\models\drone_landing\model_final --stage 1 --episodes 20 --render
```

Windows 下建议使用 `--no-capture-output`，避免 conda 对输出做 GBK 包装时出现编码问题：

```powershell
conda run --no-capture-output -n drone_rl python scripts/test_env.py
```

## 4. 系统架构

训练入口是 `scripts/train.py`，解析 CLI 参数后交给 `Trainer` 创建并行环境、SAC 模型和回调。

```text
scripts/train.py
  -> EnvConfig / TrainConfig
  -> Trainer
  -> SubprocVecEnv + VecMonitor
  -> DroneLandingEnv
  -> BaseDroneLandingEnv.step()
  -> CurriculumStrategy.compute_reward()
  -> SAC.learn()
  -> callbacks: curriculum / csv / checkpoint
```

核心类职责：

| 类/文件 | 职责 |
| --- | --- |
| `BaseDroneLandingEnv` | 后端无关逻辑：观测构造、动作缩放、通用终止条件、课程策略接口 |
| `DroneLandingEnv` | PyBullet 后端：仿真连接、场景物体、状态同步、接触检测、渲染 |
| `KinematicDroneModel` | 无人机速度指令滤波、水平风/阵风/跟踪误差、位置/yaw 积分 |
| `MovingPlatform` | 平台轨迹生成和速度计算 |
| `CurriculumManager` | 记录 episode 成败、滚动成功率和策略指标 |
| `Trainer` | 创建环境、初始化/恢复模型、启动训练、保存最终模型 |
| `CurriculumCallback` | 记录训练指标、写入 TensorBoard 和周期性命令行日志 |
| `CSVLoggingCallback` | 导出 episode、reward、training 三类 CSV |
| `CheckpointCallback` | 定期保存模型 |

单步环境数据流：

```text
action [-1, 1]
  -> _scale_action()
  -> 机体系速度转世界系速度
  -> KinematicDroneModel.apply_command()
  -> 一阶低通滤波
  -> 速度跟踪误差 + 水平风/阵风扰动
  -> KinematicDroneModel.step()
  -> MovingPlatform.step()
  -> _build_observation()
  -> CurriculumStrategy.compute_reward()
  -> _check_termination()
  -> 返回 obs, reward, terminated, truncated, info
```

## 5. 目录结构

```text
drone_landing_rl/
├── configs/
│   ├── env_config.py              环境物理参数、观测归一化
│   ├── train_config.py            SAC 超参数、课程学习参数、路径配置
│   └── README.md
│
├── envs/
│   ├── base_env.py                后端无关环境基类
│   ├── drone_landing_env.py       PyBullet 后端实现
│   ├── README.md
│   ├── dynamics/
│   │   ├── kinematic_model.py     无人机运动学积分、低通滤波和扰动
│   │   └── README.md
│   ├── landing_platform/
│   │   ├── moving_platform.py     平台运动模型（组合模式）
│   │   ├── speed_controller.py    通用速度控制器
│   │   ├── motions/               运动策略包（Static/Lissajous/Patrol/Waypoint）
│   │   └── README.md
│
├── curriculum/
│   ├── curriculum_manager.py      课程统计和手动课程指标
│   ├── strategies/                课程策略实现与注册
│   │   ├── base_strategy.py       策略抽象基类 + Mixin
│   │   ├── stage1_hover_static.py 课程一
│   │   ├── stage2_hover_moving.py 课程二
│   │   ├── stage3.py              课程三
│   │   └── stage4.py              课程四
│   └── README.md
│
├── training/
│   ├── trainer.py                 训练主流程
│   ├── callbacks.py               课程更新、CSV 导出、checkpoint 保存
│   ├── model_selection.py         最佳模型筛选与复评
│   └── README.md
│
├── scripts/
│   ├── train.py                   训练入口
│   ├── evaluate.py                评估入口
│   ├── test_env.py                环境自检
│   ├── export_model.py            模型导出
│   ├── sweep.py                   超参数扫参
│   ├── plot_results.py            训练结果绘图
│   ├── plot_eval_trajectory.py    评估轨迹绘图
│   ├── plot_eval_trajectory.m     MATLAB 轨迹绘图
│   ├── drone_landing_analysis.m   MATLAB 分析脚本
│   └── README.md
│
├── utils/
│   ├── logger.py                  日志配置
│   └── README.md
│
├── output/                        训练、评估和导出产物（默认被 git 忽略）
├── requirements.txt               Python 依赖
└── README.md                      本文档
```

## 6. 文档导航

需要深入某一层代码时，优先阅读对应目录的中文说明：

| 文档 | 说明 |
| --- | --- |
| `configs/README.md` | 环境配置、训练配置和参数修改入口 |
| `curriculum/README.md` | 课程策略、指标统计和手动阶段切换 |
| `envs/README.md` | Gym 环境分层、PyBullet 后端和 ROS/Gazebo 可替换边界 |
| `envs/dynamics/README.md` | 无人机运动学模型和指令低通滤波 |
| `envs/landing_platform/README.md` | 平台运动模式和策略配置方式 |
| `training/README.md` | SAC 配置、训练编排、TensorBoard 指标、CSV schema、最佳模型筛选 |
| `scripts/README.md` | 全部 CLI 参数表、命令示例、模型导出和 MATLAB 脚本 |
| `utils/README.md` | 日志配置说明 |

> 详细参考信息（完整参数表、指标说明、配置默认值等）请点击上方链接查看对应子目录 README。

## 7. 核心概念速览

### 7.1 观测空间（34 维 Box）

所有输入归一化并按分组裁剪。平移量使用 yaw-FLU 机体系：X 前、Y 左、Z 上。

| 索引 | 含义 | 归一化 |
| --- | --- | --- |
| 0-2 | 当前策略目标点相对无人机的位置，机体系 | x/y `/20m`, z `/10m` |
| 3-5 | 目标相对速度，机体系（平台速度减无人机实际速度） | x/y `/20m/s`, z `/5m/s` |
| 6 | 平台相对无人机 yaw | `/pi` |
| 7 | 相对 yaw_rate | `/yaw_rate_norm` |
| 8 | 目标斜距 | `/20m` |
| 9-11 | 无人机实际速度，机体系（含漂移、阵风和跟踪误差） | x/y `/20m/s`, z `/5m/s` |
| 12-14 | 无人机 roll/pitch/yaw | `/pi` |
| 15 | 无人机 yaw_rate | `/yaw_rate_norm` |
| 16 | 无人机局部高度 | `/30m` |
| 17-19 | 平台 roll/pitch/yaw | `/pi` |
| 20-22 | 平台 roll_rate/pitch_rate/yaw_rate | rate norms |
| 23-25 | 平台速度，机体系 | x/y `/20m/s`, z `/5m/s` |
| 26 | 检测质量 | `[0, 1]` |
| 27-29 | 当前无人机速度能力 | vxy `/20`, vz_down `/5`, vz_up `/5` |
| 30-33 | 上一时刻动作 | `[-1, 1]` |

每个 episode 随机化速度能力：水平上限 `[8, 20] m/s`，上升上限 `[1, 5] m/s`，下降上限 `[1, min(5, vz_up)] m/s`。

### 7.2 动作空间（4 维 Box[-1, 1]）

| 维度 | 含义 | 缩放 |
| --- | --- | --- |
| 0 | `vx_body` 前向速度 | `[-current_v_xy_max, current_v_xy_max]` |
| 1 | `vy_body` 左向速度 | `[-current_v_xy_max, current_v_xy_max]` |
| 2 | `vz` 竖直速度 | 正值用上升上限，负值用下降上限 |
| 3 | `yaw_rate` | `[-max_yaw_rate, max_yaw_rate]` |

动作先在机体系缩放，再用当前无人机 yaw 转到世界系执行。

### 7.3 课程阶段

| 课程 | 类名 | 描述 | 状态 |
| --- | --- | --- | --- |
| Stage 1 | `Stage1HoverStaticStrategy` | 静止平台上悬停 | 可训练 |
| Stage 2 | `Stage2HoverMovingStrategy` | 移动平台上悬停（3 种运动随机：Lissajous/Patrol/Waypoint） | 可训练 |
| Stage 3 | `Stage3Strategy` | 移动平台上悬停（Lissajous 运动） | 奖励未实现 |
| Stage 4 | `Stage4Strategy` | 移动平台上降落（Lissajous 运动） | 奖励未实现 |

每个课程独立定义场景配置、目标点、奖励计算和成功/失败判定。阶段之间由用户手动确认切换（Y/N），不再自动晋级。`--mix_ratio` 支持在后续阶段中穿插一定比例的前一课程回合。

课程二奖励使用三层位置函数叠加：反二次（长尾，提供远距离梯度）+ 高斯（中距精度）+ 宽高斯（近距峰值，σ_xy=0.30m 提供近距连续梯度），配合 smoothstep 门控的速度匹配惩罚。评分使用 `train_score` / `eval_score` 两套容差空间，分别用于训练信号和严格评估。

> 详见 `curriculum/README.md`

### 7.4 训练入口

```powershell
python scripts/train.py
```

关键参数：`--stage`（起始阶段）、`--max_stage`（最高阶段）、`--n_envs`（并行环境数）。阶段预算默认 Stage 1: 60M / Stage 2: 40M / Stage 3/4: 各 30M timesteps，可通过 `--total_steps` 覆盖。

> 完整 CLI 参数表、SAC 配置和最佳模型筛选流程见 `scripts/README.md` 和 `training/README.md`

### 7.5 评估入口

```powershell
python scripts/evaluate.py --model output\models\drone_landing\model_final --stage 1 --episodes 20
```

支持 GUI 渲染（`--render`）、轨迹采集（`--traj_enable`）、实时/离线绘图、多阶段评估（`--all_stages`）。

> 完整 CLI 参数表和命令示例见 `scripts/README.md`

### 7.6 日志与监控

```powershell
tensorboard --logdir output/logs
```

关键 TensorBoard 指标：`rollout/ep_rew_mean`、`curriculum/stage`、`curriculum/rolling_success_rate`。

CSV 输出：

```text
output/data/csv/{exp_name}/
├── episode_log.csv        每回合：timestep, stage, reward, length, success, train_score
├── reward_log_stageN.csv  每步奖励分项（按课程分文件）
└── training_log.csv       定期记录 actor_loss, critic_loss, ent_coef, lr, fps
```

> 完整指标说明和调试信息见 `training/README.md`

### 7.7 模型导出

```powershell
python scripts/export_model.py --model output\models\drone_landing\model_final --format all --verify
```

导出 SAC actor 为 TorchScript（`.pt`）、ONNX（`.onnx`）或 NumPy 权重（`.npz`）。输入 34 维观测，输出 4 维 `[-1, 1]` 动作。

## 8. ROS/Gazebo 迁移要点

`BaseDroneLandingEnv` 定义了 9 个后端相关抽象方法。替换 PyBullet 为 ROS/Gazebo 时，只需实现新后端类：

- `_init_simulation()` / `_reset_simulation()` — 初始化 ROS 节点、Gazebo reset
- `_apply_action()` — publish 到 MAVROS 速度 setpoint topic
- `_step_simulation()` / `_get_drone_state()` / `_get_platform_state()` — 读取 odom/TF
- `_check_contact()` / `render()` / `close()`

观测构造、奖励函数、课程学习和训练流程无需随仿真后端改变。

> 详细迁移检查清单见 `envs/README.md`

## 9. 调参速查

| 修改目标 | 编辑文件 |
| --- | --- |
| 物理参数（速度、滤波、扰动） | `configs/env_config.py` |
| 训练参数（SAC 超参、课程预算） | `configs/train_config.py` |
| 平台运动映射 | `curriculum/strategies/` 中的策略类 |
| 新增轨迹形状 | `envs/landing_platform/motions/` |

> 参数表详见 `configs/README.md`

## 10. 当前状态与已知风险

1. **课程阶段切换为手动控制**：训练在每个阶段预算结束后打印总结，用户需输入 `Y/N` 决定是否继续；非交互式运行默认 `N` 并保存退出。
2. **`--total_steps` 表示每阶段预算**：未传入时使用 `stage_timesteps` 默认值（60M/40M/30M/30M）。
3. **Stage 3/4 奖励待实现**：已注册但 `compute_reward()` 抛出 `NotImplementedError`。
4. **reward CSV 体积过大**：逐步 reward CSV 每步、每环境写入，长训练可快速增长到 GB 级。后续可考虑采样频率开关或仅在 debug 时启用。
5. **仿真和真实系统存在 sim-to-real 缺口**：当前 PyBullet 是运动学体模拟，无真实气动/电机/PX4 内环动力学，适合训练高层速度策略。
6. **策略指标由课程策略声明**：成功判定和评估字段由各策略独立定义。
7. **终端惩罚需谨慎设置**：过大的终端惩罚（>-500）会导致 OOB episode 的 TD 误差产生 critic_loss 尖峰，破坏 Q 函数稳定性。推荐 -500 以下，利用正常 episode 的机会成本而非惩罚幅值来阻止 OOB。
8. **LR 衰减在 resume 时需手动应用**：`SAC.load()` 不会继承 callable learning_rate，需在加载后手动设置 `model.learning_rate = lr_schedule`，否则 LR 冻结在 checkpoint 保存时的固定值。

## 11. 后续工作

1. 用 `drone_rl` 环境跑短训练 smoke test，检查 episode/reward CSV 输出。
2. 分别验证 Stage 1 和 Stage 2 的训练曲线、成功率和评估口径。
3. 若逐步 reward CSV 体积影响长训练，增加采样频率配置或开关。
4. 单个课程稳定后，为 Stage 3/4 设计奖励、指标和评估方式。
5. 若 PyBullet 阶段稳定，规划 Gazebo/PX4 后端迁移。
