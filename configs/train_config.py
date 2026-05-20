# 功能：定义 SAC 训练超参数、课程学习参数和训练产物路径。
"""
训练超参数配置。

所有 SAC、课程学习和基础设施相关设置都集中在这里。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class SACConfig:
    """SAC 超参数，按 RTX 4070 Ti SUPER + i7-14700KF 调过一版。"""
    learning_rate: float = 3e-4
    buffer_size: int = 3_000_000
    learning_starts: int = 10_000
    batch_size: int = 2048          # 较大的 batch 有利于提高 GPU 利用率
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    target_update_interval: int = 1
    use_sde: bool = False

    # 学习率三段式衰减：恒定(满LR) → 余弦衰减 → 平坦(最低LR)
    lr_decay: bool = True                 # 启用学习率衰减
    lr_decay_start: float = 0.30           # 段1→段2 分界（已完成进度，例：0.50=完成50%时开始衰减）
    lr_decay_end: float = 0.60            # 段2→段3 分界（已完成进度，例：0.80=完成80%时进入平坦）
    lr_decay_min_ratio: float = 0.5       # 最低 LR 占 learning_rate 的比例

    # 网络结构：3 层 MLP，每层 256 个单元
    policy_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "net_arch": [256, 256, 256],
        "log_std_init": -3.0,
    })


@dataclass
class CurriculumConfig:
    """
    课程学习参数。

    阶段切换由用户手动控制：
      - 当前阶段训练到预算步数
      - 打印阶段总结
      - 由用户决定是否继续进入下一阶段
    """
    window_size: int = 30            # 滚动成功率使用的 episode 窗口
    eval_freq: int = 8_000           # 两次课程指标检查之间的步数

    max_stage: int = 4                 # 本次运行允许手动到达的最高阶段

    # 已注册课程的预算步数（可通过 CLI --total_steps 覆盖）
    stage_timesteps: List[int] = field(default_factory=lambda: [
        60_000_000,
        20_000_000,
        30_000_000,
        30_000_000,
    ])


@dataclass
class TrainConfig:
    # ── 并行设置 ─────────────────────────────────────────────────────────────
    # i7-14700KF 有 20 个核心；
    # 课程一 128 个环境正好可以吃满资源但是还留有余量 - 已经测试验证。
    # 课程二 96  个环境正好可以吃满资源但是还留有余量 - 已经测试验证。

    n_envs: int = 96

    # ── 硬件 ─────────────────────────────────────────────────────────────────
    device: str = "cuda"   # "cuda" | "cpu"

    # ── 路径 ─────────────────────────────────────────────────────────────────
    log_dir: str = "./output/logs"
    model_dir: str = "./output/models"
    csv_dir: str = "./output/data/csv"
    reward_step_csv_enabled: bool = False

    # ── 输出频率 ─────────────────────────────────────────────────────────────
    sb3_log_interval_episodes: int = 10             # SB3 主表格输出间隔（全局 episode）
    curriculum_status_interval_episodes: int = 200  # 课程短行输出间隔（当前阶段 episode）
    save_freq: int = 2_400_000                      # checkpoint 保存间隔（真实 timesteps）

    # ── 最佳模型自动筛选 ─────────────────────────────────────────────────────
    best_model_selection_enabled: bool = True
    best_model_grid_count: int = 20
    best_model_interval_top_m: int = 5
    best_model_eval_episodes: int = 50
    best_model_eval_envs_by_stage: Dict[int, int] = field(default_factory=lambda: {
        1: 50,
        2: 50,
        3: 50,
        4: 50,
    })
    best_model_keep_top_n: int = 5
    best_model_train_score_weight: float = 0.5
    best_model_eval_score_weight: float = 0.5

    # ── 可复现性 ─────────────────────────────────────────────────────────────
    seed: int = 42

    # ── 实验名 ───────────────────────────────────────────────────────────────
    exp_name: str = "drone_landing"

    # ── 子配置 ───────────────────────────────────────────────────────────────
    sac: SACConfig = field(default_factory=SACConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
