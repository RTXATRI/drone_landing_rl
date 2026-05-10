# 功能：不依赖 Stable-Baselines3/torch 的 SubprocVecEnv worker。
"""
Windows spawn worker。

这个模块会被每个环境子进程导入，所以必须保持轻量：
不要在这里导入 Stable-Baselines3、torch 或训练模块。
"""

from __future__ import annotations

from typing import Any

import cloudpickle


class CloudpickleWrapper:
    """用 cloudpickle 序列化环境工厂闭包。"""

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: bytes) -> None:
        self.var = cloudpickle.loads(var)


def _get_env_attr(env: Any, name: str) -> Any:
    if hasattr(env, "get_wrapper_attr"):
        return env.get_wrapper_attr(name)
    return getattr(env, name)


def _is_wrapped(env: Any, wrapper_class: type) -> bool:
    try:
        import gymnasium as gym
    except Exception:
        return False

    current = env
    while isinstance(current, gym.Wrapper):
        if isinstance(current, wrapper_class):
            return True
        current = current.env
    return False


def subprocess_worker(remote, parent_remote, env_fn_wrapper: CloudpickleWrapper) -> None:
    parent_remote.close()
    env = env_fn_wrapper.var()
    reset_info: dict[str, Any] | None = {}

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = _get_env_attr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(_get_env_attr(env, data))
            elif cmd == "has_attr":
                try:
                    _get_env_attr(env, data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(_is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except (EOFError, KeyboardInterrupt):
            break
