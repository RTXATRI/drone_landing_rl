# 功能：内置课程策略实现。

from __future__ import annotations

import numpy as np

from curriculum.strategies.base_strategy import CurriculumStrategy, LandingStrategyMixin


class Stage4LandMovingStrategy(LandingStrategyMixin, CurriculumStrategy):
    label = "Land   | Moving Platform"
    short_label = "Land-Moving"

    def stage_id(self) -> int:
        return 4

    def setup_scene(self, env, rng: np.random.Generator) -> None:
        self._setup_landing_scene(env, rng, motion="sinusoidal")

    def compute_reward(self, **kwargs):
        raise NotImplementedError(
            "Stage 4 reward is not implemented yet. Design it in the Stage 4 curriculum pass."
        )
