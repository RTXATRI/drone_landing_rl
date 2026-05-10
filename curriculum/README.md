# curriculum 目录说明

本目录负责课程策略注册与训练统计逻辑。每个课程策略独立定义场景、奖励、成功条件和评估指标；统计层只记录策略输出的结果。

## 文件职责

- `curriculum_manager.py`：记录每个课程的 episode 数、成功数、平均奖励、平均长度和策略指标；维护滚动统计窗口。阶段切换由 `Trainer` 在阶段预算结束后询问用户手动决定。
- `strategies/`：课程策略注册表和内置策略实现。新增课程时优先新增策略类并注册。

## 已注册课程

| 课程 | 文件 | 类名 | 描述 | 状态 |
| --- | --- | --- | --- | --- |
| Stage 1 | `stage1_hover_static.py` | `Stage1HoverStaticStrategy` | 静止平台上悬停 | ✅ 可训练 |
| Stage 2 | `stage2_hover_moving.py` | `Stage2HoverMovingStrategy` | 移动平台上悬停（3 种速度受控运动模式随机池：Lissajous/Patrol/Waypoint，平台真实速度带轻量延迟和误差） | ✅ 可训练 |
| Stage 3 | `stage3.py` | `Stage3Strategy` | 待定 | ❌ 奖励未实现 |
| Stage 4 | `stage4.py` | `Stage4Strategy` | 待定 | ❌ 奖励未实现 |

## 课程策略

策略类负责决定：

- episode 场景如何配置。
- 平台运动模式如何设置（通过 `MovingPlatform.set_motion_strategy()`）。
- 目标点如何计算（悬停课程继承 `HoverStrategyMixin`，降落课程继承 `LandingStrategyMixin`）。
- 奖励和终端奖励如何计算。
- 成功、失败和评估指标如何定义。
- CSV 日志列定义。

策略指标统一命名为无课程前缀的通用名（如 `train_score`、`eval_score`），由 `stage` 列区分课程。`reward_log_stageN.csv` 按课程分文件存储，列名也无前缀。

### 训练、奖励和评估空间

悬停课程把空间判定拆成三套，避免奖励塑形和验收标准互相牵连：

- `TRAIN_SCORE_*`：训练得分空间，用于 `train_score`。
- `EVAL_SCORE_*`：评估得分空间，用于 `eval_score` 和环境级连续保持步数。
- `HOLD_REWARD_*`：保持奖励空间，只在 `compute_reward()` 的保持奖励块附近定义，用于 `reward/hold` 和 `reward/hold_break`。

Stage 1/2 的默认数值保持旧行为：训练空间较宽，评估空间保持原标准，奖励空间默认等于旧评估空间但可以独立调参。训练和评估环境都会按同一套策略参数计算 `train_score` 和 `eval_score`，`success_mode="eval"` 只改变 episode 成功判定选用哪个分数。新增课程时，训练/评估参数属于多方法共享常量，应放在类顶部；奖励塑形参数应放在 `compute_reward()` 对应奖励块附近。

## Stage 2 平台观测

Stage 2 作为 Python 预训练，平台速度直接来自仿真真实速度，不再通过位置变化和 KF 推算。策略会加入轻量观测模拟：默认 2 step 延迟，并对 XY 速度乘以 `Uniform(0.98, 1.02)` 的 episode 级比例误差。平台位置和 yaw 仍保留小噪声与检测质量丢帧标记。

## 穿插训练（Stage 2+）

训练后续课程时，可通过 `env.enable_mix_training(ratio)` 启用穿插训练。每回合以 `ratio` 概率随机选择前一课程策略，其余使用当前课程策略。策略实例通过 `env._strategies` 字典按 `stage_id` 缓存，`env._nominal_stage` 记录真实训练阶段不受混合影响。

`CurriculumCallback` 自动过滤混合回合（`episode_stage != current_stage` 时跳过），避免滚动指标被稀释。TensorBoard 中 `curriculum/actual_mix_ratio` 显示实际混合比例。

## 新增课程指南

1. 新建策略文件（如 `stage3.py`），继承 `HoverStrategyMixin` 或 `LandingStrategyMixin` + `CurriculumStrategy`。
2. 实现 `stage_id()`、`setup_scene()`、`compute_reward()`、`terminal_bonus()` 等方法。
3. 在 `strategies/__init__.py` 的 `STRATEGY_MAP` 中注册。
4. 策略中的 episode 指标使用通用命名（如 `("train_score",)`），不要加课程前缀。
5. 若课程包含悬停保持奖励，按“训练、奖励和评估空间”拆分参数，不要让奖励空间复用评估空间的连续步数状态。

## 阶段切换逻辑

课程不再自动晋级。每个课程训练到配置的预算步数后，训练流程会打印阶段总结；如果本次运行允许多个课程，则等待用户输入 `Y/N` 决定是否进入下一课程。成功率和策略指标只作为观察信息。

## 迁移提示

课程学习主要服务训练阶段。部署到真实 ROS/PX4 工程时，通常不需要训练期的手动阶段切换逻辑；真实工程只需要加载某个训练完成的策略，并由安全监督层决定是否允许进入跟踪或降落模式。
