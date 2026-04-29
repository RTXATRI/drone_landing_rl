# 功能：内置课程策略实现。

from __future__ import annotations

import numpy as np

from curriculum.strategies.base_strategy import CurriculumStrategy, HoverStrategyMixin


class Stage3HoverMovingStrategy(HoverStrategyMixin, CurriculumStrategy):
    label = "Follow | Moving Platform"
    short_label = "Follow-Moving"

    def stage_id(self) -> int:
        return 3

    def setup_scene(self, env, rng: np.random.Generator) -> None:
        self._setup_hover_scene(env, rng, motion="sinusoidal")

    def compute_reward(self, **kwargs):
        raise NotImplementedError(
            "Stage 3 reward is not implemented yet. Design it in the Stage 3 curriculum pass."
        )
