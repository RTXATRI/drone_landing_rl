# Drone Landing RL 工程总览

更新时间：2026-04-28

本文件是工程根目录的总入口说明文档，目标是让开发者、研究者或 AI 助手快速理解这个工程的用途、架构、运行方式、关键参数和当前风险。各主要源码目录内也新增了中文 `README.md`，用于解释该目录的代码职责和后续 ROS/Gazebo 迁移边界。

## 1. 一句话概括

这是一个用 SAC 强化学习训练无人机在静止/移动平台上悬停、跟踪和降落的框架。当前仿真后端是 PyBullet，环境接口基于 Gymnasium，训练由 Stable-Baselines3 驱动，并支持由用户手动控制课程阶段。

核心能力：

- 34 维归一化观测，描述目标相对位姿、无人机实际状态、平台状态、检测质量、当前速度能力和上一时刻动作。
- 4 维连续动作，输出机体系速度指令和 yaw rate。
- PyBullet 预训练后端加入轻量水平风、间歇阵风和速度跟踪误差，使悬停不再是完全理想静止问题。
- 基于策略模式的课程学习；每个课程独立定义场景、奖励、成功条件和评估指标。
- 奖励函数、课程逻辑、训练流程和仿真后端解耦，便于从 PyBullet 迁移到 ROS/Gazebo/PX4。
- 支持 TensorBoard、CSV、checkpoint、轨迹采集、离线/实时绘图和模型导出。

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

当前已知 conda 环境：

```powershell
conda activate drone_rl
```

已验证：

```powershell
conda run -n drone_rl python --version
```

结果：

```text
Python 3.13.13
```

Windows 下建议使用 `--no-capture-output`，避免 conda 对输出做 GBK 包装时出现编码问题：

```powershell
conda run --no-capture-output -n drone_rl python scripts\test_env.py
```

2026-04-28 drone_rl 环境兼容性验证（完整）：

**环境配置：**
- Python 3.13.13
- PyTorch 2.7.1+cu128（CUDA 12.8）
- NVIDIA GeForce RTX 4070 Ti SUPER 可用

**功能验证：**
- ✅ `scripts/test_env.py` 全部 25 项测试通过
  - Gym API、观测/动作空间、奖励计算、终止条件、课程切换
  - VecMonitor 指标、扰动语义、性能基准
  - 单环境性能：5,269 FPS（满足 16 环境并行要求）
- ✅ `scripts/train.py` 冒烟训练通过（2 环境 × 64 步）
  - SAC 模型初始化、SubprocVecEnv 并行采样
  - TensorBoard 训练日志、CSV 导出、模型保存
  - 训练完成时间：< 1 秒（CPU 模式）
- ✅ `scripts/evaluate.py` 参数解析正常
- ✅ CUDA 和 PyTorch GPU 支持正常

## 3. 文档导航

根目录 `README.md` 提供全局视角。需要深入某一层代码时，可优先阅读对应目录的中文说明：

| 文档 | 说明 |
| --- | --- |
| `configs/README.md` | 环境配置、训练配置和参数修改入口 |
| `curriculum/README.md` | 课程策略、指标统计和手动阶段切换 |
| `envs/README.md` | Gym 环境分层、PyBullet 后端和 ROS/Gazebo 可替换边界 |
| `envs/dynamics/README.md` | 无人机运动学模型和指令低通滤波 |
| `envs/landing_platform/README.md` | 平台运动模式和策略配置方式 |
| `envs/rewards/README.md` | 通用奖励工具和策略自定义方式 |
| `training/README.md` | 并行环境、SAC 训练编排和 callbacks |
| `scripts/README.md` | 训练、评估、测试、绘图、导出和扫参脚本用途 |
| `utils/README.md` | 日志和通用工具说明 |

## 4. 目录结构

```text
drone_landing_rl/
├── configs/
│   ├── env_config.py              环境物理参数、观测归一化
│   ├── train_config.py            SAC 超参数、课程学习参数、路径配置
│   └── README.md                  配置目录说明
│
├── envs/
│   ├── base_env.py                后端无关环境基类，负责观测/动作/通用终止/策略接口
│   ├── drone_landing_env.py       PyBullet 后端实现
│   ├── square_flight_test.py      独立正方形航线演示
│   ├── README.md                  环境目录说明
│   ├── dynamics/
│   │   ├── kinematic_model.py     无人机运动学积分、一阶低通滤波和轻量扰动
│   │   └── README.md              动力学目录说明
│   ├── landing_platform/
│   │   ├── moving_platform.py     平台运动模式：static/linear/sinusoidal/figure8
│   │   └── README.md              平台目录说明
│   └── rewards/
│       ├── reward_functions.py    可复用奖励计算工具
│       └── README.md              奖励目录说明
│
├── curriculum/
│   ├── curriculum_manager.py      课程统计和手动课程指标
│   └── README.md                  课程目录说明
│
├── training/
│   ├── trainer.py                 训练主流程：SubprocVecEnv + SAC + callbacks
│   ├── callbacks.py               课程更新、CSV 导出、checkpoint 保存
│   └── README.md                  训练目录说明
│
├── scripts/
│   ├── train.py                   训练入口
│   ├── evaluate.py                评估入口，支持渲染和轨迹采集
│   ├── test_env.py                环境自检
│   ├── export_model.py            模型导出：TorchScript/ONNX/NumPy
│   ├── sweep.py                   超参数扫参
│   ├── plot_results.py            训练结果绘图
│   ├── plot_eval_trajectory.py    评估轨迹绘图
│   ├── plot_eval_trajectory.m     MATLAB 轨迹绘图
│   ├── drone_landing_analysis.m   MATLAB 分析脚本
│   └── README.md                  脚本目录说明
│
├── utils/
│   ├── logger.py                  日志配置
│   ├── misc.py                    通用辅助工具
│   └── README.md                  工具目录说明
│
├── logs/                          TensorBoard 和运行日志
├── models/                        模型 checkpoint 和 final 模型
├── data/csv/                      训练/评估 CSV 数据
├── requirements.txt               Python 依赖
└── README.md                      本文档
```

## 5. 系统架构

训练入口是 `scripts/train.py`。它解析 CLI 参数，构造 `EnvConfig` 和 `TrainConfig`，交给 `Trainer` 创建并行环境、SAC 模型和回调。

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

## 6. 环境设计

### 6.1 后端抽象接口

`BaseDroneLandingEnv` 定义 9 个后端相关抽象方法。更换 PyBullet 为 ROS/Gazebo 时，只需要实现这些方法：

```python
_init_simulation()
_reset_simulation(rng)
_apply_action(vel_cmd)
_step_simulation()
_get_drone_state()
_get_platform_state()
_check_contact()
render()
close()
```

其余观测、奖励、课程、终止和日志逻辑无需随仿真后端改变。

### 6.2 PyBullet 后端

当前 `DroneLandingEnv` 使用 PyBullet：

- 每个环境实例创建独立 PyBullet server。
- GUI 模式用于人工观察，DIRECT 模式用于训练。
- 无人机和平台都是 `baseMass=0` 的运动学体。
- PyBullet 主要用于可视化、相机渲染和几何对象管理。
- 接触检测使用解析 AABB 判断，而不是依赖两个质量为 0 的刚体接触。

这意味着当前环境是“高层速度控制任务”的轻量模拟，并非完整气动/电机/飞控动力学仿真。

### 6.3 无人机运动学模型

`KinematicDroneModel` 对速度指令做一阶低通滤波：

```text
v_filtered[t] = alpha * v_filtered[t-1] + (1 - alpha) * v_cmd[t]
```

用途：

- 抑制 RL 策略输出的高频抖动。
- 模拟真实 PX4 速度控制器的带宽限制。
- 让动作突变不会瞬间变成速度突变。

默认滤波系数在 `DroneConfig.cmd_filter_alpha` 中配置，当前为 `0.7`。

在 PyBullet 预训练后端中，滤波后的命令速度还会经过轻量扰动层：

```text
actual_velocity = filtered_command_velocity * tracking_scale + wind_velocity
tracking_scale ~ Uniform(0.98, 1.02)
wind_velocity = base_wind_velocity + gust_velocity
```

当前扰动只作用于水平 XY 方向，不加入垂直风；yaw_rate 第一版不加入跟踪误差。默认参数位于 `DisturbanceConfig`：

- 每回合固定弱漂移：`0.00-0.08m/s`，方向随机。
- 间歇水平阵风：峰值 `0.05-0.22m/s`，持续 `1.0-4.0s`，方向、强度、形状随机。
- 速度跟踪误差：每 step、每个 XYZ 分量独立乘以 `0.98-1.02`。

因此 `drone_state["velocity"]` 表示扰动和跟踪误差后的实际速度，而不是 RL 动作输出速度，也不是单纯的滤波命令速度。观测、奖励、降落判定和悬停保持均使用这个实际速度。

### 6.4 平台运动模型

`MovingPlatform` 支持：

| motion_type | 说明 |
| --- | --- |
| `static` | 平台静止，可由任意课程策略选择 |
| `linear` | X 方向正弦往复 |
| `sinusoidal` | X/Y 独立正弦运动，可由任意课程策略选择 |
| `figure8` | 8 字轨迹 |

课程策略在 episode reset 前通过 `MovingPlatform.set_motion(...)` 指定运动模式。
新增课程不需要修改平台模型，除非需要新的轨迹形状。

## 7. 观测空间

观测空间是 34 维 `Box`，所有输入都做归一化并按分组裁剪。平移量使用 yaw-FLU 机体系：X 前、Y 左、Z 上。XY 由无人机 yaw 旋转到机体系，Z 保持世界竖直方向。

| 索引 | 含义 | 归一化 |
| --- | --- | --- |
| 0-2 | 当前策略目标点相对无人机的位置，机体系；目标语义由活动课程策略定义 | x/y `/20m`, z `/10m` |
| 3-5 | 目标相对速度，机体系；使用平台速度减无人机实际速度 | x/y `/20m/s`, z `/5m/s` |
| 6 | 平台相对无人机 yaw | `/pi` |
| 7 | 相对 yaw_rate | `/yaw_rate_norm` |
| 8 | 目标斜距 | `/20m` |
| 9-11 | 无人机实际速度，机体系；当前 PyBullet 中已包含弱漂移、阵风和跟踪误差 | x/y `/20m/s`, z `/5m/s` |
| 12-14 | 无人机 roll/pitch/yaw | `/pi` |
| 15 | 无人机 yaw_rate | `/yaw_rate_norm` |
| 16 | 无人机局部高度/对地高度/海拔高度抽象量；当前 PyBullet 中用 `drone_z - platform_z` 近似 | `/30m` |
| 17-19 | 平台 roll/pitch/yaw | `/pi` |
| 20-22 | 平台 roll_rate/pitch_rate/yaw_rate | rate norms |
| 23-25 | 平台速度，机体系 | x/y `/20m/s`, z `/5m/s` |
| 26 | 检测质量 | `[0, 1]` |
| 27-29 | 当前无人机速度能力 | vxy `/20`, vz_down `/5`, vz_up `/5` |
| 30-33 | 上一时刻动作 | `[-1, 1]` |

为什么使用相对量：

- 对绝对初始位置不敏感。
- 减少网络学习无关坐标偏移的负担。
- 有利于泛化到不同平台位置和轨迹。

每个 episode 会随机当前无人机速度能力：

- 水平速度上限：`[8, 20] m/s`
- 上升速度上限：`[1, 5] m/s`
- 下降速度上限：`[1, min(5, vz_up)] m/s`

这些能力被同时用于动作缩放和观测输入。

当前初始出生范围：

- 水平半径：`3-35m`
- 高度：`1-30m`
- 水平出界终止边界：`45m`

注意：观测仍保留原归一化和裁剪设置。水平相对位置超过 `20m * 1.5`、相对高度超过 `10m * 1.5` 时会被裁剪到观测上限，但奖励、终止和悬停评分仍使用未裁剪的真实仿真状态。

## 8. 动作空间

动作空间是 4 维连续 `Box[-1, 1]`：

| 维度 | 含义 | 缩放 |
| --- | --- | --- |
| 0 | `vx_body` 前向速度 | `[-current_v_xy_max, current_v_xy_max]` |
| 1 | `vy_body` 左向速度 | `[-current_v_xy_max, current_v_xy_max]` |
| 2 | `vz` 竖直速度 | 正值用上升上限，负值用下降上限 |
| 3 | `yaw_rate` | `[-max_yaw_rate, max_yaw_rate]` |

动作先在机体系缩放，再用当前无人机 yaw 转到世界系执行。

## 9. 目标点与终止条件

目标点由活动课程策略决定。环境引擎只调用 `strategy.get_target_pos(...)`，
不会假设目标是悬停点、平台表面或其他形式。

环境引擎只保留通用终止条件：

- 无人机水平距离平台过远：out-of-bounds。
- 无人机低于最小高度：below ground。
- 达到 `max_steps` 时 truncated。

课程相关成功、失败、接触处理和终止奖励由活动策略独立定义。新增课程可以使用全新的仿真场景、成功判定和评估指标，不需要与已有课程保持一致。

## 10. 奖励函数

奖励由活动课程策略通过 `compute_reward(...)` 独立计算。通用 `RewardCalculator` 已退役，只保留兼容壳用于捕捉遗留误调用。

因此新增课程不需要沿用任何已有课程的奖励项、权重、启用条件或终止奖励。若某个课程需要新的奖励参数，应在对应策略中补充常量、注释和日志字段，并由该策略单独解释和使用。

## 11. 课程学习

课程由 `curriculum/strategies/` 中注册的策略类定义。每个策略独立负责：

- 场景配置和平台运动模式。
- 目标点定义。
- 奖励计算。
- 成功/失败判定。
- 终端奖励和 episode/eval 指标。

阶段切换逻辑：

```text
训练当前阶段到该阶段预算步数结束
-> 打印阶段总结
-> 如果当前阶段 < max_stage，等待用户输入 Y/N
-> Y：进入下一阶段；N：保存模型并退出
```

课程阶段不再自动晋级。成功率和策略声明的指标只作为训练观察信息。

默认关键参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `advance_threshold` | `0.80` | 旧自动晋级阈值，当前仅保留兼容 |
| `window_size` | `30` | 滚动成功率窗口 |
| `eval_freq` | `8000` | 周期性日志和 TensorBoard 写入频率 |
| `min_steps_per_stage` | `60000` | 旧自动晋级参数，当前不参与阶段切换 |
| `max_stage` | 未传入时等于 `--stage` | 本次训练允许手动推进到的最高课程阶段 |

注意：直接运行只训练 `--stage` 指定的单个课程。若希望训练多个课程，需要显式传入 `--max_stage`，并在每个阶段结束后手动输入 `Y/N`：

```powershell
conda run --no-capture-output -n drone_rl python scripts\train.py --stage 1 --max_stage 4
```

## 12. 训练系统

训练入口：

```powershell
conda run --no-capture-output -n drone_rl python scripts\train.py
```

训练 CLI 参数：

| 参数 | 默认值 | 示例 | 说明 |
| --- | --- | --- | --- |
| `--stage` | `1` | `--stage 1` | 起始课程阶段，取值 `1-4` |
| `--max_stage` | 未传入时等于 `--stage` | `--max_stage 4` | 本次训练允许手动推进到的最高课程阶段；大于 `--stage` 时，每个阶段结束后询问 `Y/N` |
| `--resume` | `None` | `--resume models\drone_landing\ckpt_0001000000` | 从 checkpoint 恢复训练，路径不带 `.zip` |
| `--n_envs` | `16` | `--n_envs 8` | 并行采样环境数 |
| `--device` | `cuda` | `--device cpu` | 训练设备，可选 `cuda` 或 `cpu`；如果 CUDA 不可用会回退到 CPU |
| `--exp_name` | `drone_landing` | `--exp_name stage1_hover` | 实验名，用于 `logs/`、`models/`、`data/csv/` 子目录 |
| `--seed` | `42` | `--seed 0` | 训练随机种子 |
| `--total_steps` | `None` | `--total_steps 20000` | 覆盖每个所选课程阶段的训练步数预算；未传入时使用 `configs/train_config.py` 中的 `stage_timesteps` |
| `--batch_size` | `512` | `--batch_size 256` | SAC replay buffer 采样 batch size |
| `--lr` | `3e-4` | `--lr 0.0003` | SAC 学习率 |
| `--log_dir` | `./logs` | `--log_dir logs` | TensorBoard 和运行日志根目录 |
| `--model_dir` | `./models` | `--model_dir models` | checkpoint 和 final 模型根目录 |
| `--csv_dir` | `./data/csv` | `--csv_dir data/csv` | episode/reward/training CSV 输出根目录 |

训练阶段保持随机化：策略可在 reset 时自行采样场景参数，速度能力也在每个 episode 随机采样。评估入口中的固定控制参数只影响验证，不影响训练。

训练时的周期性控制台日志会使用清晰的滚动窗口命名，例如：

```text
[   128,000] Stage 1 BuiltinStrategy | Eps=36 | SuccessRate30=0.0% | StrategyMetric100=0.0
```

其中 `SuccessRate30` 表示最近 30 个 episode 的滚动成功率，窗口大小来自 `CurriculumConfig.window_size`，不是 30 个训练 step。

示例：

```powershell
# 默认只训练起始课程
conda run --no-capture-output -n drone_rl python scripts\train.py

# 手动多阶段课程：每个阶段结束后输入 Y/N
conda run --no-capture-output -n drone_rl python scripts\train.py --stage 1 --max_stage 4

# 快速 CPU 冒烟训练
conda run --no-capture-output -n drone_rl python scripts\train.py --n_envs 4 --total_steps 20000 --exp_name smoke_test --device cpu

# 从指定课程和已有 checkpoint 恢复
conda run --no-capture-output -n drone_rl python scripts\train.py --stage 3 --resume models\drone_landing\ckpt_0001000000 --max_stage 4

# 完整参数示例
conda run --no-capture-output -n drone_rl python scripts\train.py `
  --stage 1 `
  --max_stage 1 `
  --n_envs 16 `
  --device cuda `
  --exp_name stage1_hover `
  --seed 42 `
  --total_steps 60000000 `
  --batch_size 512 `
  --lr 0.0003 `
  --log_dir ./logs `
  --model_dir ./models `
  --csv_dir ./data/csv
```

SAC 默认配置：

| 参数 | 默认值 |
| --- | --- |
| `learning_rate` | `3e-4` |
| `buffer_size` | `1_000_000` |
| `learning_starts` | `10_000` |
| `batch_size` | `512` |
| `tau` | `0.005` |
| `gamma` | `0.99` |
| `train_freq` | `1` |
| `gradient_steps` | `1` |
| `ent_coef` | `auto` |
| `net_arch` | `[256, 256, 256]` |

训练产物：

```text
logs/{exp_name}/
models/{exp_name}/
data/csv/{exp_name}/
```

## 13. 评估系统

评估入口：

```powershell
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model models\drone_landing\model_final --stage 4
```

常用功能：

```powershell
# GUI 渲染
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model models\drone_landing\model_final --stage 4 --render

# 评估全部阶段
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model models\drone_landing\model_final --all_stages

# 使用交互/默认验证参数
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model models\stage1_v2\model_final --stage 1 --episodes 20

# 完全无交互验证：手动指定策略需要的评估参数
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model models\stage1_v2\model_final --stage 1 --episodes 20 --hover_height 5 --eval_v_xy_max 10 --eval_v_z_up_max 3 --eval_v_z_down_max 2

# 覆盖评估回合最大步数
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model models\drone_landing\model_final --stage 1 --eval_max_steps 12000 --hover_height 5 --eval_v_xy_max 10 --eval_v_z_up_max 3 --eval_v_z_down_max 2

# 轨迹采集和离线绘图
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model models\drone_landing\model_final --stage 1 --episodes 10 --hover_height 5 --eval_v_xy_max 10 --eval_v_z_up_max 3 --eval_v_z_down_max 2 --traj_enable --traj_plot --traj_plot_per_episode --traj_plot_combined

# 实时绘图
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model models\drone_landing\model_final --stage 1 --episodes 5 --hover_height 5 --eval_v_xy_max 10 --eval_v_z_up_max 3 --eval_v_z_down_max 2 --render --traj_enable --traj_plot --traj_plot_realtime
```

评估入口支持固定策略目标参数和速度能力。是否使用这些参数由活动课程策略决定：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `hover_height` | `5.0m` | 可选策略目标高度 |
| `eval_v_xy_max` | `10.0m/s` | 评估水平最大速度能力 |
| `eval_v_z_up_max` | `3.0m/s` | 评估上升最大速度能力 |
| `eval_v_z_down_max` | `2.0m/s` | 评估下降最大速度能力 |

如果这些参数未通过 CLI 显式传入，交互式运行会提示输入；直接回车使用默认值。非交互式运行不会阻塞，直接使用默认值。训练入口不使用这些评估固定值，训练仍保留原有速度能力和悬停高度随机化。

轨迹相关默认值：

| 参数 | 默认值 |
| --- | --- |
| `traj_enable` | `false` |
| `traj_plot` | `false` |
| `traj_plot_realtime` | `false` |
| `traj_plot_per_episode` | `true` |
| `traj_plot_combined` | `true` |
| `traj_out_dir` | `data/eval_traj` |
| `traj_stride` | `1` |
| `traj_realtime_refresh` | `10` |

## 14. 日志与数据

TensorBoard：

```powershell
conda run --no-capture-output -n drone_rl tensorboard --logdir logs
```

浏览器打开：

```text
http://localhost:6006
```

关键 TensorBoard 指标：

| 指标 | 含义 |
| --- | --- |
| `rollout/ep_rew_mean` | 平均回合奖励 |
| `rollout/ep_len_mean` | 平均回合长度 |
| `train/actor_loss` | actor loss |
| `train/critic_loss` | critic loss |
| `train/ent_coef` | SAC 熵系数 |
| `curriculum/stage` | 当前课程阶段 |
| `curriculum/rolling_success_rate` | 滚动成功率 |

环境 `info` 中还会提供轻量扰动调试指标，便于训练时确认实际速度来源：

| 指标 | 含义 |
| --- | --- |
| `reward/velocity_toward` | 悬停阶段实际速度方向朝向目标点的奖惩 |
| `metric/stability_coeff` | 悬停稳定速度惩罚系数，范围 `[0, 1]` |
| `metric/velocity_toward_cos` | 实际速度方向与目标方向夹角余弦 |
| `metric/velocity_toward_weight` | 速度朝向奖励的距离权重和速度权重乘积 |
| `metric/wind_speed` | 当前总水平风速大小 |
| `metric/base_wind_speed` | 当前 episode 固定弱漂移大小 |
| `metric/gust_speed` | 当前阵风速度大小 |
| `metric/tracking_scale_mean` | 当前 step 的 XYZ 速度跟踪比例平均值 |

CSV 输出：

```text
data/csv/{exp_name}/
├── episode_log.csv     每回合：timestep, stage, reward, length, success
├── reward_log_stageN.csv 每步核心奖励分项和距离/速度指标
└── training_log.csv    定期记录 actor_loss, critic_loss, ent_coef, lr, fps
```

MATLAB 读取示例：

```matlab
T = readtable('data/csv/drone_landing/episode_log.csv');
stage1 = T(T.stage == 1, :);
plot(stage1.timestep, movmean(stage1.success, 50));
xlabel('Timestep');
ylabel('Success Rate (MA-50)');
title('Curriculum Learning Curve');
```

## 15. 模型导出

`scripts/export_model.py` 可将训练好的 SAC actor 导出为：

- TorchScript：`.pt`
- ONNX：`.onnx`
- NumPy 权重：`.npz`

示例：

```powershell
conda run --no-capture-output -n drone_rl python scripts\export_model.py --model models\drone_landing\model_final --format all --verify
```

导出的模型输入为 34 维观测，输出为 4 维 `[-1, 1]` 动作。

## 16. 后端迁移到 ROS/Gazebo

设计上，迁移只需要新增后端类，例如：

```python
from envs.base_env import BaseDroneLandingEnv

class GazeboEnv(BaseDroneLandingEnv):
    def _init_simulation(self):
        pass

    def _reset_simulation(self, rng):
        pass

    def _apply_action(self, vel_cmd):
        pass

    def _step_simulation(self):
        pass

    def _get_drone_state(self):
        pass

    def _get_platform_state(self):
        pass

    def _check_contact(self):
        pass

    def render(self):
        pass

    def close(self):
        pass
```

可能的 ROS/Gazebo 映射：

| 抽象方法 | ROS/Gazebo 实现思路 |
| --- | --- |
| `_init_simulation()` | 初始化 ROS 节点、连接 Gazebo |
| `_reset_simulation(rng)` | 调用 Gazebo reset service，设置模型位姿 |
| `_apply_action(vel_cmd)` | publish 到 MAVROS 速度 setpoint topic |
| `_step_simulation()` | sleep 一个控制周期或等待状态回调 |
| `_get_drone_state()` | 读取 local_position/odom，`velocity` 应为飞控/里程计融合后的实际速度 |
| `_get_platform_state()` | 读取平台 TF/odom |
| `_check_contact()` | 读取碰撞传感器或接触状态 |
| `render()` | 可选 |
| `close()` | 释放资源 |

迁移后理论上无需修改：

- 观测构造
- 奖励函数
- 课程学习
- 训练流程
- TensorBoard/CSV 日志

## 17. 调参和扩展指南

修改物理参数：编辑 `configs/env_config.py`。

常改参数：

| 参数 | 说明 |
| --- | --- |
| `DroneConfig.max_vx/max_vy` | 无人机基础水平速度 |
| `DroneConfig.cmd_filter_alpha` | 指令滤波强度 |
| `PlatformConfig.landing_radius` | 降落水平容差 |
| `PlatformConfig.motion_frequency` | 平台运动频率 |
| `EpisodeConfig.max_steps` | 每回合最大步数 |

修改训练参数：编辑 `configs/train_config.py` 或使用 CLI。

常改参数：

| 参数 | 说明 |
| --- | --- |
| `n_envs` | 并行环境数 |
| `batch_size` | SAC batch size |
| `learning_rate` | 学习率 |
| `buffer_size` | replay buffer 大小 |
| `window_size` | 滚动成功率窗口 |
| `stage_timesteps` | 各阶段预算 |

修改平台运动：

- 课程到平台运动的映射位于 `curriculum/strategies/` 的各个策略类中。
- 策略通过 `MovingPlatform.set_motion(...)` 选择 `static`、`sinusoidal` 等轨迹；新增课程只需新增并注册策略类。
- 如果需要全新的轨迹形状，再在 `moving_platform.py` 中增加运动模式。

## 18. 当前已有产物和训练观察

已有模型目录：

```text
models/stage1_v1/
models/stage1_v2/
models/smoke_obs34/
```

已有 CSV：

```text
data/csv/stage1_hover/
data/csv/stage1_20260413/
data/csv/smoke_obs34/
```

从当前 CSV 观察：

- `stage1_20260413` 有 8704 个 episode，`success=0` 全部为 0。
- `stage1_hover` 有 544 个 episode，`success=0` 全部为 0。
- 训练奖励可以变为正值，但课程成功率仍长期为 0。

这些是旧训练产物，字段和当前 reward CSV schema 可能不一致。后续新训练应优先查看回合奖励、回合长度、成功率，以及各阶段策略自己声明的指标。

## 19. 已知风险和优先修复建议

### 19.1 策略指标应由课程策略声明

旧实验中使用过 `hover_score` 作为 episode 指标。该指标已移除；新课程不需要沿用它，成功判定和评估字段应由对应策略独立声明。

### 19.2 课程阶段切换已改为手动控制

训练会在每个阶段预算步数结束后打印总结。若本次允许多个阶段，用户需要输入 `Y/N` 决定是否进入下一课程；非交互式运行默认按 `N` 保存退出。

### 19.3 `--total_steps` 语义已改为每阶段预算

`--total_steps` 现在表示每个所选课程阶段的训练步数。未传入 `--max_stage` 时只训练 `--stage` 指定的当前课程。

### 19.4 Stage 2/3/4 奖励待实现

当前仅 Stage 1 保持可训练。Stage 2/3/4 已注册但奖励函数会显式抛出 `NotImplementedError`，不能直接用于正式训练；后续应在对应课程方案阶段单独设计奖励、指标和评估方式。

### 19.5 多阶段 CSV callback 句柄复用

继续到 Stage 2 前，需要修复 `CSVLoggingCallback` 跨多个 `model.learn()` 复用时关闭 reward 文件句柄后未清空 `_reward_fhs/_reward_ws` 的问题，否则后续阶段可能在 flush 已关闭文件时失败。

### 19.6 checkpoint 间隔与并行环境数

当前 `CheckpointCallback.save_freq` 按 callback 调用次数计算，多环境下实际保存间隔为 `save_freq * n_envs` timesteps。后续应统一改为按 `num_timesteps` 计算。

### 19.7 reward CSV 体积过大

现有 `data/csv/stage1_20260413/reward_log.csv` 约 3.63GB。当前每步、每环境写入奖励分项，长训练会快速产生 GB 级数据。

当前新 schema 已精简逐步 reward 列。若仍然过大，后续再考虑 `reward_log_freq`、是否记录 reward CSV 的开关、只记录第 0 个环境，或默认只在 debug/smoke 实验中开启逐步 reward 日志。

### 19.8 仿真和真实系统存在 sim-to-real 缺口

当前 PyBullet 环境是运动学体，没有真实重力、气动、电机动力学或 PX4 内环细节。适合训练高层速度策略，但迁移到真实系统前需要更高保真验证。

## 20. 推荐后续工作顺序

1. 用 `drone_rl` 环境跑短训练 smoke test，并检查简化后的 episode/reward CSV。
2. 对已有 `stage1_v1/v2` 模型做统一评估，确认策略是否实际能稳定悬停。
3. 单个课程稳定后，再为下一个注册课程设计奖励、指标和评估方式。
4. 进入 Stage 2 前修复多阶段 CSV callback 句柄和 checkpoint 间隔问题。
5. 若 PyBullet 阶段稳定，再规划 Gazebo/PX4 后端迁移。

## 21. 常用命令速查

环境自检：

```powershell
conda run --no-capture-output -n drone_rl python scripts\test_env.py
```

快速训练：

```powershell
conda run --no-capture-output -n drone_rl python scripts\train.py --n_envs 4 --total_steps 20000 --exp_name smoke_test --device cpu
```

单课程 debug：

```powershell
conda run --no-capture-output -n drone_rl python scripts\train.py --max_stage 1 --exp_name stage1_debug
```

手动多阶段课程：

```powershell
conda run --no-capture-output -n drone_rl python scripts\train.py --stage 1 --max_stage 4 --exp_name manual_curriculum
```

评估指定课程：

```powershell
conda run --no-capture-output -n drone_rl python scripts\evaluate.py `
  --model models\stage1_v2\model_final `
  --stage 1 `
  --episodes 20 `
  --eval_max_steps 3600 `
  --hover_height 5 `
  --eval_v_xy_max 10 `
  --eval_v_z_up_max 3 `
  --eval_v_z_down_max 2 `
  --device cpu `
  --seed 0 `
  --render_speed 1.0 `
  --traj_enable `
  --traj_plot `
  --traj_plot_realtime `
  --traj_plot_per_episode `
  --traj_plot_combined `
  --traj_out_dir data/eval_traj `
  --traj_stride 1 `
  --traj_realtime_refresh 10
```

常用精简版：

```powershell
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model models\stage1_v2\model_final --stage 1 --episodes 20 --hover_height 5 --eval_v_xy_max 10 --eval_v_z_up_max 3 --eval_v_z_down_max 2
```

评估 CLI 参数：

| 参数 | 示例 | 说明 |
| --- | --- | --- |
| `--model` | `models\stage1_v2\model_final` | 模型路径，不带 `.zip` |
| `--stage` | `1` | 评估课程 id |
| `--episodes` | `20` | 评估 episode 数 |
| `--eval_max_steps` | `3600` | 仅评估生效的单回合最大步数 |
| `--hover_height` | `5` | 可选策略目标高度；不传则交互询问，回车默认 `5m` |
| `--eval_v_xy_max` | `10` | 评估水平速度能力；不传则交互询问，回车默认 `10m/s` |
| `--eval_v_z_up_max` | `3` | 评估上升速度能力；不传则交互询问，回车默认 `3m/s` |
| `--eval_v_z_down_max` | `2` | 评估下降速度能力；不传则交互询问，回车默认 `2m/s` |
| `--render` | 开关 | 启动 PyBullet GUI；需要可视化时加上 |
| `--render_speed` | `1.0` | GUI 播放速度；`0` 表示不 sleep、尽可能快渲染 |
| `--device` | `cpu` | 模型推理设备，评估默认 `cpu` |
| `--seed` | `0` | 评估随机种子 |
| `--traj_enable` | 开关 | 开启详细轨迹采集 |
| `--traj_plot` | 开关 | 评估后生成离线轨迹图 |
| `--traj_plot_realtime` | 开关 | 评估时显示实时轨迹图 |
| `--traj_plot_per_episode` / `--no-traj_plot_per_episode` | 开关 | 是否输出单 episode 图，默认开启 |
| `--traj_plot_combined` / `--no-traj_plot_combined` | 开关 | 是否输出跨 episode 汇总图，默认开启 |
| `--traj_out_dir` | `data/eval_traj` | 轨迹输出根目录 |
| `--traj_stride` | `1` | 每隔多少 step 记录一行轨迹 |
| `--traj_realtime_refresh` | `10` | 实时图每隔多少 step 刷新 |
| `--save_traj` | 开关 | 旧版轨迹保存开关，保留兼容；新流程优先用 `--traj_enable` |

TensorBoard：

```powershell
conda run --no-capture-output -n drone_rl tensorboard --logdir logs
```

模型导出：

```powershell
conda run --no-capture-output -n drone_rl python scripts\export_model.py --model models\drone_landing\model_final --format all --verify
```

## 22. 本轮文档整理记录

2026-04-26：

- 将原 `README.md`、`ARCHITECTURE.md` 和项目分析记录合并到本文件。
- 根目录以 `README.md` 作为总入口；主要源码目录新增目录级 `README.md`。
- 修正原文档中平台目录旧命名，将 `envs/platform/` 统一为真实目录 `envs/landing_platform/`。
- 记录 conda 环境 `drone_rl` 和 Windows 下推荐的 `conda run --no-capture-output` 用法。
- 记录 `scripts/test_env.py` 在 `drone_rl` 环境中全部通过。
- 按源码脚本文件补充中文功能说明，便于后续 ROS/Gazebo 迁移前理解模块边界。

2026-04-28：

- 将运行命令中的 conda 环境名统一更新为 `drone_rl`（原 SAC 环境已弃用）。
- 完整验证 `drone_rl` 环境兼容性：
  - `scripts/test_env.py`：25 项测试全部通过，单环境 FPS 5,269（满足并行要求）
  - `scripts/train.py` 冒烟测试：2 环境 × 64 步训练完成，SAC/SubprocVecEnv/日志/模型保存均正常
  - `scripts/evaluate.py` 参数解析正常
  - PyTorch 2.7.1+cu128、CUDA 12.8、RTX 4070 Ti SUPER GPU 支持验证通过
- 同步 `requirements.txt` 到 `drone_rl` 中已验证的核心包版本范围。
- 更新第 2 节环境验证记录，详细列出功能验证清单和性能基准。
