# scripts 目录说明

本目录保存命令行入口和辅助工具，不包含核心环境逻辑。

## 主要入口

- `train.py`：训练入口，创建配置并启动 `Trainer`；课程阶段由 `--stage` 和可选 `--max_stage` 手动控制。
- `evaluate.py`：评估入口，支持 GUI 渲染、轨迹采集、实时绘图、离线绘图，以及评估阶段固定悬停高度和速度能力。
- `test_env.py`：环境自检脚本，用于验证 Gym API、观测、奖励、终止条件、扰动语义和性能。

## 工具脚本

- `export_model.py`：导出 SAC actor 为 TorchScript、ONNX 或 NumPy 权重，服务工程部署。
- `record_video.py`：用 PyBullet rgb_array 录制评估视频。
- `sweep.py`：启动网格或随机超参数扫参实验。
- `plot_results.py`：读取训练 CSV 并生成训练曲线。
- `plot_eval_trajectory.py`：读取评估轨迹 CSV 并绘图。
- `plot_eval_trajectory.m`、`drone_landing_analysis.m`：MATLAB 分析和绘图脚本。

## 迁移提示

迁移到 ROS/Gazebo/PX4 时，最重要的脚本是 `export_model.py`。训练好的 actor 应导出后由 ROS 推理节点加载，实时节点不应依赖 Stable-Baselines3 的训练流程。
