# 功能：Stage 3 课程策略（未分配）。

from __future__ import annotations

import numpy as np

from curriculum.strategies.base_strategy import CurriculumStrategy, HoverStrategyMixin


class Stage3Strategy(HoverStrategyMixin, CurriculumStrategy):
    label = "Stage 3"
    short_label = "S3"

    def stage_id(self) -> int:
        return 3

    def setup_scene(self, env, rng: np.random.Generator) -> None:
        from envs.landing_platform.motions.lissajous_motion import LissajousMotion
        self._setup_hover_scene(env, rng, motion_strategy=LissajousMotion())

    def compute_reward(self, **kwargs):
        raise NotImplementedError(
            "Stage 3 reward is not implemented yet."
        )
