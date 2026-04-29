# 功能：内置课程策略实现。

from __future__ import annotations

import numpy as np

from curriculum.strategies.base_strategy import CurriculumStrategy, LandingStrategyMixin


class Stage2LandStaticStrategy(LandingStrategyMixin, CurriculumStrategy):
    label = "Land   | Static Platform"
    short_label = "Land-Static"

    def stage_id(self) -> int:
        return 2

    def setup_scene(self, env, rng: np.random.Generator) -> None:
        self._setup_landing_scene(env, rng, motion="static")

    def compute_reward(self, **kwargs):
        raise NotImplementedError(
            "Stage 2 reward is not implemented yet. Design it in the Stage 2 curriculum pass."
        )
