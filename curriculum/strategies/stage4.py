# 功能：Stage 4 课程策略（未分配）。

from __future__ import annotations

import numpy as np

from curriculum.strategies.base_strategy import CurriculumStrategy, LandingStrategyMixin


class Stage4Strategy(LandingStrategyMixin, CurriculumStrategy):
    label = "Stage 4"
    short_label = "S4"

    def stage_id(self) -> int:
        return 4

    def setup_scene(self, env, rng: np.random.Generator) -> None:
        from envs.landing_platform.motions.lissajous_motion import LissajousMotion
        self._setup_landing_scene(env, rng, motion_strategy=LissajousMotion())

    def compute_reward(self, **kwargs):
        raise NotImplementedError(
            "Stage 4 reward is not implemented yet."
        )
