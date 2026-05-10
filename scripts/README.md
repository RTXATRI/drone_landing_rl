# scripts 目录说明

本目录保存命令行入口和辅助工具，不包含核心环境逻辑。

## 主要入口

- `train.py`：训练入口，创建配置并启动 `Trainer`；课程阶段由 `--stage` 和可选 `--max_stage` 手动控制。
- `evaluate.py`：评估入口，支持 GUI 渲染、轨迹采集、实时绘图、离线绘图，以及评估阶段固定悬停高度和速度能力。
- `test_env.py`：环境自检脚本，用于验证 Gym API、观测、奖励、终止条件、扰动语义和性能。

## train.py CLI 参数

```powershell
conda run --no-capture-output -n drone_rl python scripts\train.py
```

| 参数 | 默认值 | 示例 | 说明 |
| --- | --- | --- | --- |
| `--stage` | `1` | `--stage 1` | 起始课程阶段，取值 `1-4` |
| `--max_stage` | `4` | `--max_stage 4` | 本次训练允许手动推进到的最高课程阶段；大于 `--stage` 时，每个阶段结束后询问 `Y/N` |
| `--resume` | `None` | `--resume output\models\drone_landing\ckpt_0001000000` | 从 model-only checkpoint 恢复训练，路径不带 `.zip`；当前不恢复 replay buffer |
| `--n_envs` | `96` | `--n_envs 8` | 并行采样环境数 |
| `--device` | `cuda` | `--device cpu` | 训练设备，可选 `cuda` 或 `cpu`；如果 CUDA 不可用会回退到 CPU |
| `--exp_name` | `drone_landing` | `--exp_name stage1_hover` | 实验名，用于 `output/logs/`、`output/models/`、`output/data/csv/` 子目录 |
| `--seed` | `42` | `--seed 0` | 训练随机种子 |
| `--total_steps` | `None` | `--total_steps 20000` | 覆盖每个所选课程阶段的训练步数预算；未传入时使用 `configs/train_config.py` 中的 `stage_timesteps` |
| `--batch_size` | `2048` | `--batch_size 1024` | SAC replay buffer 采样 batch size |
| `--lr` | `3e-4` | `--lr 0.0003` | SAC 学习率 |
| `--log_dir` | `./output/logs` | `--log_dir output/logs` | TensorBoard 和运行日志根目录 |
| `--model_dir` | `./output/models` | `--model_dir output/models` | checkpoint 和 final 模型根目录 |
| `--csv_dir` | `./output/data/csv` | `--csv_dir output/data/csv` | episode/reward/training CSV 输出根目录 |
| `--mix_ratio` | `0.0` | `--mix_ratio 0.2` | 穿插训练比例（仅 Stage 2+），使用前一课程策略的回合占比，默认 0.0 不开启 |
| `--best_model_selection` / `--no-best_model_selection` | `True` | `--no-best_model_selection` | 是否在阶段正常结束后自动筛选并复评最佳模型 |
| `--best_grid_count` | `20` | `--best_grid_count 20` | 每阶段等间距采样候选数量，不含起点、包含阶段末尾 |
| `--best_interval_top_m` | `5` | `--best_interval_top_m 5` | 每个等间距区间内按训练 episode reward 额外保留的峰值候选数 |
| `--best_eval_episodes` | `50` | `--best_eval_episodes 50` | 每个候选最终复评的 episode 数 |
| `--best_eval_envs` | `1=50,2=50,3=50,4=50` | `--best_eval_envs 1=50,2=50,3=32` | 最佳模型复评时各阶段使用的并行环境数 |

### 常用命令示例

```powershell
# 默认从 Stage 1 开始，允许手动继续到最高 Stage 4
conda run --no-capture-output -n drone_rl python scripts\train.py

# 只训练单个课程
conda run --no-capture-output -n drone_rl python scripts\train.py --stage 1 --max_stage 1

# 手动多阶段课程
conda run --no-capture-output -n drone_rl python scripts\train.py --stage 1 --max_stage 4

# 快速 CPU 冒烟训练
conda run --no-capture-output -n drone_rl python scripts\train.py --n_envs 4 --total_steps 20000 --exp_name smoke_test --device cpu

# 从指定课程和已有 checkpoint 恢复
conda run --no-capture-output -n drone_rl python scripts\train.py --stage 3 --resume output\models\drone_landing\ckpt_0001000000 --max_stage 4

# 完整参数示例
conda run --no-capture-output -n drone_rl python scripts\train.py `
  --stage 1 `
  --max_stage 4 `
  --n_envs 96 `
  --device cuda `
  --exp_name stage1_hover `
  --seed 42 `
  --total_steps 60000000 `
  --batch_size 2048 `
  --lr 0.0003 `
  --log_dir ./output/logs `
  --model_dir ./output/models `
  --csv_dir ./output/data/csv `
  --best_grid_count 20 `
  --best_interval_top_m 5 `
  --best_eval_episodes 50 `
  --best_eval_envs 1=50,2=50
```

## evaluate.py CLI 参数

```powershell
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model <path> --stage <N>
```

| 参数 | 示例 | 说明 |
| --- | --- | --- |
| `--model` | `output\models\drone_landing\model_final` | 模型路径，不带 `.zip` |
| `--stage` | `1` | 评估课程 id |
| `--all_stages` | 开关 | 评估全部已注册课程阶段 |
| `--episodes` | `20` | 评估 episode 数 |
| `--eval_max_steps` | `3600` | 仅评估生效的单回合最大步数 |
| `--hover_height` | `5` | 可选策略目标高度；不传则交互询问，回车默认 `5m` |
| `--eval_v_xy_max` | `10` | 评估水平速度能力；不传则交互询问，回车默认 `10m/s` |
| `--eval_v_z_up_max` | `3` | 评估上升速度能力；不传则交互询问，回车默认 `3m/s` |
| `--eval_v_z_down_max` | `2` | 评估下降速度能力；不传则交互询问，回车默认 `2m/s` |
| `--render` | 开关 | 启动 PyBullet GUI |
| `--render_speed` | `1.0` | GUI 播放速度；`0` 表示不 sleep、尽可能快渲染 |
| `--device` | `cpu` | 模型推理设备，评估默认 `cpu` |
| `--seed` | `0` | 评估随机种子 |
| `--random_seed` | 开关 | 每次评估启动时生成新的随机 base seed |
| `--traj_enable` | 开关 | 开启详细轨迹采集 |
| `--traj_plot` | 开关 | 评估后生成离线轨迹图 |
| `--traj_plot_realtime` | 开关 | 评估时显示实时轨迹图 |
| `--traj_plot_per_episode` / `--no-traj_plot_per_episode` | 开关 | 是否输出单 episode 图，默认开启 |
| `--traj_plot_combined` / `--no-traj_plot_combined` | 开关 | 是否输出跨 episode 汇总图，默认开启 |
| `--traj_out_dir` | `output/data/eval_traj` | 轨迹输出根目录 |
| `--traj_stride` | `1` | 每隔多少 step 记录一行轨迹 |
| `--traj_realtime_refresh` | `10` | 实时图每隔多少 step 刷新 |
| `--save_traj` | 开关 | 旧版轨迹保存开关，保留兼容；新流程优先用 `--traj_enable` |

### 常用命令示例

```powershell
# GUI 渲染评估
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model output\models\drone_landing\model_final --stage 4 --render

# 评估全部阶段
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model output\models\drone_landing\model_final --all_stages

# 精简评估（固定参数）
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model output\models\drone_landing\model_final --stage 1 --episodes 20 --hover_height 5 --eval_v_xy_max 10 --eval_v_z_up_max 3 --eval_v_z_down_max 2

# 轨迹采集和绘图
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model output\models\drone_landing\model_final --stage 1 --episodes 10 --hover_height 5 --eval_v_xy_max 10 --eval_v_z_up_max 3 --eval_v_z_down_max 2 --traj_enable --traj_plot --traj_plot_per_episode --traj_plot_combined

# 实时绘图
conda run --no-capture-output -n drone_rl python scripts\evaluate.py --model output\models\drone_landing\model_final --stage 1 --episodes 5 --hover_height 5 --eval_v_xy_max 10 --eval_v_z_up_max 3 --eval_v_z_down_max 2 --render --traj_enable --traj_plot --traj_plot_realtime
```

## 工具脚本

- `export_model.py`：导出 SAC actor 为 TorchScript（`.pt`）、ONNX（`.onnx`）或 NumPy 权重（`.npz`），输入 34 维观测、输出 4 维 `[-1,1]` 动作。

  ```powershell
  conda run --no-capture-output -n drone_rl python scripts\export_model.py --model output\models\drone_landing\model_final --format all --verify
  ```

- `record_video.py`：用 PyBullet rgb_array 录制评估视频。
- `sweep.py`：启动网格或随机超参数扫参实验。
- `plot_results.py`：读取训练 CSV 并生成训练曲线。
- `plot_eval_trajectory.py`：读取评估轨迹 CSV 并绘图（Python）。

### MATLAB 脚本

- `plot_eval_trajectory.m`：MATLAB 轨迹绘图。
- `drone_landing_analysis.m`：MATLAB 分析脚本，可从 CSV 读取数据：

  ```matlab
  T = readtable('output/data/csv/drone_landing/episode_log.csv');
  stage1 = T(T.stage == 1, :);
  plot(stage1.timestep, movmean(stage1.success, 50));
  xlabel('Timestep');
  ylabel('Success Rate (MA-50)');
  title('Curriculum Learning Curve');
  ```

## 迁移提示

迁移到 ROS/Gazebo/PX4 时，最重要的脚本是 `export_model.py`。训练好的 actor 应导出后由 ROS 推理节点加载，实时节点不应依赖 Stable-Baselines3 的训练流程。
