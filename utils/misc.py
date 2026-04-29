# 功能：提供随机种子、设备选择、参数统计和滑动统计等通用辅助函数。
"""
通用工具函数。

包括：
  - set_all_seeds():  为可复现性统一设置随机种子
  - get_device():     自动选择最佳可用设备
  - count_parameters(): 统计网络参数量
  - format_num():     以易读形式格式化大数字
  - ExponentialMovingAverage: 轻量 EMA 跟踪器
"""

from __future__ import annotations

import random
import os
from typing import Union

import numpy as np


def set_all_seeds(seed: int) -> None:
    """为 Python、NumPy 和 PyTorch 设置随机种子，尽量保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass


def get_device(requested: str = "cuda") -> str:
    """返回最佳可用设备字符串。"""
    try:
        import torch
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def format_num(n: Union[int, float]) -> str:
    """将大数字格式化为便于显示的形式，例如 1_500_000 → '1.5M'。"""
    n = float(n)
    if abs(n) >= 1e9:
        return f"{n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(int(n))


def count_parameters(model) -> int:
    """统计 PyTorch 模型中的可训练参数数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ExponentialMovingAverage:
    """
    轻量 EMA 跟踪器。

    适合在不保存完整历史的情况下平滑日志标量。
    """

    def __init__(self, alpha: float = 0.99):
        self.alpha = alpha
        self._value: float | None = None

    def update(self, x: float) -> float:
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha * self._value + (1.0 - self.alpha) * x
        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    def reset(self) -> None:
        self._value = None


class RollingMean:
    """固定窗口滚动均值（使用循环缓冲区）。"""

    def __init__(self, window: int):
        self._buf   = np.zeros(window, dtype=np.float64)
        self._n     = window
        self._count = 0
        self._ptr   = 0

    def update(self, x: float) -> float:
        self._buf[self._ptr] = x
        self._ptr   = (self._ptr + 1) % self._n
        self._count = min(self._count + 1, self._n)
        return self.mean

    @property
    def mean(self) -> float:
        if self._count == 0:
            return 0.0
        return float(self._buf[:self._count].mean())

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._count  = 0
        self._ptr    = 0
