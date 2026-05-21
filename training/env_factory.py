# 功能：提供 Windows spawn 友好的训练/评估环境工厂。
"""
轻量环境工厂。

这个模块故意不导入 torch、Stable-Baselines3、Trainer 或 model_selection。
Windows 的 SubprocVecEnv 使用 spawn 时会在 worker 中反序列化这些闭包；
保持 import 链轻量可以避免每个 worker 重复加载 CUDA/cuDNN DLL。
"""

from __future__ import annotations

import contextlib
import os
import sys

from configs.env_config import EnvConfig


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Temporarily suppress Python and C-extension stdout/stderr noise."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            stdout_fd = os.dup(1)
            try:
                stderr_fd = os.dup(2)
            except OSError:
                os.close(stdout_fd)
                raise
        except OSError:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
            return

        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            try:
                os.dup2(stdout_fd, 1)
            except OSError:
                pass
            finally:
                os.close(stdout_fd)
            try:
                os.dup2(stderr_fd, 2)
            except OSError:
                pass
            finally:
                os.close(stderr_fd)


def _quiet_import_drone_env():
    """Import DroneLandingEnv while hiding PyBullet's native build-time banner."""
    with _suppress_stdout_stderr():
        from envs.drone_landing_env import DroneLandingEnv
    return DroneLandingEnv


def make_training_env_fn(env_config: EnvConfig, rank: int, seed: int, stage: int):
    """返回创建单个训练环境的闭包。"""
    env_seed = int(seed) + int(rank)
    stage = int(stage)

    def _init():
        from curriculum.strategies import create_strategy

        DroneLandingEnv = _quiet_import_drone_env()

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

        DroneLandingEnv = _quiet_import_drone_env()

        env = DroneLandingEnv(
            env_config,
            strategy=create_strategy(stage, env_config),
        )
        env.set_success_mode("eval")
        env.reset(seed=env_seed)
        return env

    return _init
