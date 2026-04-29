# 功能：保留旧奖励计算器入口，防止遗留调用静默训练。
"""
通用 RewardCalculator 已废弃。

奖励现在由具体 curriculum strategy 独立实现。保留这个兼容壳是为了让
遗留导入或误调用尽早失败，而不是返回 0 reward 继续训练。
"""


class RewardCalculator:
    """Deprecated compatibility shell."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def compute(self, *args, **kwargs):
        raise RuntimeError(
            "RewardCalculator is retired; implement reward in the curriculum strategy."
        )
