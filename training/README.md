# training 目录说明

本目录封装 Stable-Baselines3 SAC 训练流程。

## 文件职责

- `trainer.py`：创建并行环境、初始化或恢复 SAC 模型、按阶段预算启动训练、在多阶段训练时询问用户是否进入下一课程，并保存最终模型。
- `callbacks.py`：实现课程指标记录、周期性 TensorBoard/命令行日志、CSV 日志导出和 checkpoint 保存。

## 手动课程控制

当前训练不再自动晋级。`--stage` 表示起始阶段；未传入 `--max_stage` 时只训练当前阶段。若传入更高的 `--max_stage`，每个阶段结束后会打印总结，并等待用户输入 `Y/N` 决定是否继续。

## 训练产物

训练默认输出到：

- `logs/{exp_name}/`
- `models/{exp_name}/`
- `data/csv/{exp_name}/`

这些目录属于运行产物，不是核心源码。

## 迁移提示

本目录主要用于 Python 预训练和后续 Gazebo/SITL 训练。部署到真实 ROS/PX4 工程时，一般不需要 `Trainer` 和 callbacks，只需要导出的 actor 模型和运行时观测/动作适配逻辑。
