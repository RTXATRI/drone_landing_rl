# 功能：提供 Windows spawn 友好的训练/评估环境工厂。
"""
轻量环境工厂。

这个模块故意不导入 torch、Stable-Baselines3、Trainer 或 model_selection。
Windows 的 SubprocVecEnv 使用 spawn 时会在 worker 中反序列化这些闭包；
保持 import 链轻量可以避免每个 worker 重复加载 CUDA/cuDNN DLL。
"""

from __future__ import annotations

from configs.env_config import EnvConfig


def make_training_env_fn(env_config: EnvConfig, rank: int, seed: int, stage: int):
    """返回创建单个训练环境的闭包。"""
    env_seed = int(seed) + int(rank)
    stage = int(stage)

    def _init():
        from curriculum.strategies import create_strategy
        from envs.drone_landing_env import DroneLandingEnv

        env = DroneLandingEnv(
            env_config,
            strategy=create_strategy(stage, env_config),
        )
        env.reset(seed=env_seed)
        return env

    return _init


def make_eval_env_fn(env_config: EnvConfig, rank: int, seed: int, stage: int):
    """返回创建单个评估环境的闭包。"""
    env_seed = int(seed) + int(rank)
    stage = int(stage)

    def _init():
        from curriculum.strategies import create_strategy
        from envs.drone_landing_env import DroneLandingEnv

        env = DroneLandingEnv(
            env_config,
            strategy=create_strategy(stage, env_config),
        )
        env.set_success_mode("eval")
        env.reset(seed=env_seed)
        return env

    return _init
