# 功能：提供 worker 不加载 torch 的 SubprocVecEnv 变体。
"""
轻量 SubprocVecEnv。

Stable-Baselines3 自带的 SubprocVecEnv 在 worker 进程中会导入
`stable_baselines3.common.env_util`，而 SB3 包顶层会导入 torch。Windows spawn 下
这会让每个环境进程加载 CUDA/cuDNN DLL。本类保留 SB3 VecEnv 接口，但 worker
target 放在 `training.spawn_worker`，避免训练 worker 反向加载 torch。
"""

from __future__ import annotations

import multiprocessing as mp
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

from training.spawn_worker import CloudpickleWrapper, subprocess_worker


def _stack_obs(obs_list: list[VecEnvObs] | tuple[VecEnvObs], space: spaces.Space) -> VecEnvObs:
    assert isinstance(obs_list, (list, tuple)), "expected list or tuple of observations"
    assert len(obs_list) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, dict)
        assert isinstance(obs_list[0], dict)
        return {key: np.stack([single_obs[key] for single_obs in obs_list]) for key in space.spaces.keys()}
    if isinstance(space, spaces.Tuple):
        assert isinstance(obs_list[0], tuple)
        return tuple(np.stack([single_obs[i] for single_obs in obs_list]) for i in range(len(space.spaces)))
    return np.stack(obs_list)


class SpawnSafeSubprocVecEnv(VecEnv):
    """SubprocVecEnv compatible wrapper whose worker target does not import SB3."""

    def __init__(self, env_fns: list[Callable[[], gym.Env]], start_method: str | None = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            start_method = "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)], strict=True)
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns, strict=True):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=subprocess_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        super().__init__(len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions, strict=True):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results, strict=True)
        return _stack_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results, strict=True)
        self._reset_seeds()
        self._reset_options()
        return _stack_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray | None]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array`."
            )
            return [None for _ in self.remotes]
        for remote in self.remotes:
            remote.send(("render", None))
        return [remote.recv() for remote in self.remotes]

    def has_attr(self, attr_name: str) -> bool:
        target_remotes = self._get_target_remotes(indices=None)
        for remote in target_remotes:
            remote.send(("has_attr", attr_name))
        return all(remote.recv() for remote in target_remotes)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        if (
            getattr(wrapper_class, "__module__", "") == "stable_baselines3.common.monitor"
            and getattr(wrapper_class, "__name__", "") == "Monitor"
        ):
            return [False for _ in self._get_indices(indices)]
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> list[Any]:
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]
