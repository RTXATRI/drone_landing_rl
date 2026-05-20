# training 目录说明

本目录封装 Stable-Baselines3 SAC 训练流程。

## 文件职责

- `trainer.py`：创建并行环境、初始化或恢复 SAC 模型、按阶段预算启动训练、在多阶段训练时询问用户是否进入下一课程，并保存最终模型。
- `callbacks.py`：实现课程指标记录、周期性 TensorBoard/命令行日志、CSV 日志导出和 checkpoint 保存。
- `model_selection.py`：收集每阶段候选模型，并在阶段正常结束后并行复评、排序和保存 `top_models/model_stageN_best_{rank}.zip`。

## 手动课程控制

当前训练不再自动晋级。`--stage` 表示起始阶段；未传入 `--max_stage` 时只训练当前阶段。若传入更高的 `--max_stage`，每个阶段结束后会打印总结，并等待用户输入 `Y/N` 决定是否继续。

非交互式运行默认选择 `N` 并保存退出。若只想训练单个课程，令 `--max_stage` 等于 `--stage`：

```powershell
conda run --no-capture-output -n drone_rl python scripts\train.py --stage 1 --max_stage 1
```

穿插训练（仅 Stage 2+ 生效）通过 `--mix_ratio` 指定一定比例的回合使用前一课程策略：

```powershell
conda run --no-capture-output -n drone_rl python scripts\train.py --stage 2 --mix_ratio 0.2
```

`--mix_ratio 0.2` 表示 20% 的 episode 使用 Stage 1 策略，80% 使用 Stage 2。默认 `0.0` 不开启。

## SAC 默认配置

| 参数 | 默认值 |
| --- | --- |
| `learning_rate` | `3e-4` |
| `buffer_size` | `3_000_000` |
| `learning_starts` | `10_000` |
| `batch_size` | `2048` |
| `tau` | `0.005` |
| `gamma` | `0.99` |
| `train_freq` | `1` |
| `gradient_steps` | `4` |
| `ent_coef` | `auto` |
| `target_update_interval` | `2` |
| `net_arch` | `[256, 256, 256]` |

默认阶段预算：Stage 1: 60M、Stage 2: 40M、Stage 3/4: 各 30M timesteps。`--total_steps` 会覆盖所有所选阶段的单阶段预算。

## 训练产物

训练默认输出到：

- `output/logs/{exp_name}/` — TensorBoard 和运行日志
- `output/models/{exp_name}/` — checkpoint 和 final 模型
- `output/data/csv/{exp_name}/` — episode/reward/training CSV

当前 checkpoint 是 model-only：周期保存的 `ckpt_*.zip` 和最终保存的 `model_final.zip` 只包含 SAC 模型本体。`--resume` 会恢复模型参数和优化器状态，但 replay buffer 会重新积累；完整 replay buffer 断点恢复暂未实现。

### 最佳模型筛选

阶段正常结束后，默认会进行最佳模型自动筛选。每个阶段先按训练预算分成 `best_grid_count` 个区间，保存每个区间终点的 grid candidate；每个区间内再按训练 rollout 的 episode reward 保留 `best_interval_top_m` 个峰值候选。默认候选上限为 `20 + 20*5 = 120`。训练中的 `train_score` 受 SAC 探索影响较大，因此只用于最终复评统计，不用于训练中入围判断。

最佳模型复评会在阶段 `model.learn()` 正常结束后运行；`Ctrl+C` 或异常中断不会触发当前阶段复评。复评对所有候选使用同一组随机 seed，评估时使用 deterministic action，并按 `combined_score = avg_train_score * 0.3 + avg_eval_score * 0.7` 排序。

任意评估 episode 出现 `oob`、`below_ground` 或 `crashed` 时，该候选标记为 `fail`；普通低分或 `success=False` 不直接等同于失败行为。排序时 `clean` 模型优先，`clean` 内按综合分降序。

最佳模型输出：

```text
output/models/{exp_name}/
├── model_final.zip                 训练流程最后时刻的模型，不等同于 best model
└── top_models/
    ├── model_stageN_best_1.zip     该阶段复评 rank 1 模型
    ├── model_stageN_best_2.zip
    └── ...
```

训练中会临时保存候选到 `stageN_candidates/`；阶段复评结束后，该候选目录会被删除，只保留 `top_models/` 下的最终 top-N 模型。使用最佳模型评估时，路径不带 `.zip`：

```powershell
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model output\models\drone_landing\top_models\model_stage2_best_1 --stage 2
```

## TensorBoard 指标

启动 TensorBoard：

```powershell
conda run --no-capture-output -n drone_rl tensorboard --logdir output/logs
```

关键指标：

| 指标 | 含义 |
| --- | --- |
| `rollout/ep_rew_mean` | 平均回合奖励 |
| `rollout/ep_len_mean` | 平均回合长度 |
| `train/actor_loss` | actor loss |
| `train/critic_loss` | critic loss |
| `train/ent_coef` | SAC 熵系数 |
| `curriculum/stage` | 当前课程阶段 |
| `curriculum/rolling_success_rate` | 滚动成功率 |
| `curriculum/rolling_train_score` | 当前课程训练口径滚动百分制分数 |
| `curriculum/actual_mix_ratio` | 穿插训练实际混合比例（未开启时为 0.0） |

控制台输出分为两类：SB3 主表格 `rollout/...` / `train/...` 由 `TrainConfig.sb3_log_interval_episodes` 控制，按全局完成 episode 数触发；课程状态短行由 `TrainConfig.curriculum_status_interval_episodes` 控制，按当前阶段有效 episode 数触发。checkpoint 由 `TrainConfig.save_freq` 控制，按真实 `num_timesteps` 保存 model-only checkpoint；默认约每 2.4M timesteps 保存一次。

训练时的课程状态行使用短格式，例如：

```text
[60.48M] S2 Hover-Moving | eps=757 | SR30=0.0% | score30=0.0 | len30=3600 | rew30=+12312.6
```

其中 `SR30` 表示最近 30 个 episode 的滚动成功率，窗口大小来自 `CurriculumConfig.window_size`。

## 环境调试指标

环境 `info` 中提供轻量扰动和奖励调试指标：

| 指标 | 含义 |
| --- | --- |
| `reward/pos` | 当前课程的位置奖励分项 |
| `reward/vel` | Stage 1 近距速度惩罚 |
| `reward/vel_match` | Stage 2 近距相对速度匹配惩罚 |
| `reward/velocity_toward` | Stage 1 速度方向朝向目标点的 shaping |
| `reward/yaw` / `reward/yaw_rate` | 偏航角和偏航角速度惩罚 |
| `reward/action` | 动作平滑和幅值惩罚 |
| `metric/horiz_err` / `metric/vert_err` | 目标水平/垂直误差 |
| `metric/rel_speed` / `metric/plat_speed` | Stage 2 相对速度和平台速度 |
| `metric/gate_xy` | Stage 2 水平速度匹配门控 |
| `metric/wind_speed` | 当前总水平风速大小 |
| `metric/base_wind_speed` | 当前 episode 固定弱漂移大小 |
| `metric/gust_speed` | 当前阵风速度大小 |
| `metric/tracking_scale_mean` | 当前 step 的 XYZ 速度跟踪比例平均值 |

## CSV 输出

```text
output/data/csv/{exp_name}/
├── episode_log.csv        每回合：timestep, stage, reward, length, success, train_score
├── reward_log_stageN.csv  每步奖励分项和距离/速度指标（按课程分文件，N 为课程编号）
└── training_log.csv       定期记录 actor_loss, critic_loss, ent_coef, lr, fps
```

`episode_log.csv` 中 `train_score` 统一命名无课程前缀，按 `stage` 列区分课程。`reward_log_stage1.csv` 和 `reward_log_stage2.csv` 按课程分别记录奖励分项；Stage 2 额外包含速度匹配、相对速度、平台速度和速度匹配门控等诊断列。

逐步 reward CSV 会按每步、每环境写入，长训练会快速增长到 GB 级；需要控制磁盘占用时应优先降低记录频率或只在 debug 实验中开启逐步 reward 日志。

## 迁移提示

本目录主要用于 Python 预训练和后续 Gazebo/SITL 训练。部署到真实 ROS/PX4 工程时，一般不需要 `Trainer` 和 callbacks，只需要导出的 actor 模型和运行时观测/动作适配逻辑。
