# 功能：注册课程策略并提供阶段到策略的工厂方法。

from __future__ import annotations

from typing import Dict, Tuple, Type

from configs.env_config import EnvConfig
from curriculum.strategies.base_strategy import CurriculumStrategy
from curriculum.strategies.stage1_hover_static import Stage1HoverStaticStrategy
from curriculum.strategies.stage2_hover_moving import Stage2HoverMovingStrategy
from curriculum.strategies.stage3 import Stage3Strategy
from curriculum.strategies.stage4 import Stage4Strategy


STRATEGY_MAP: Dict[int, Type[CurriculumStrategy]] = {
    1: Stage1HoverStaticStrategy,
    2: Stage2HoverMovingStrategy,
    3: Stage3Strategy,
    4: Stage4Strategy,
}


def create_strategy(stage: int, env_config: EnvConfig) -> CurriculumStrategy:
    """为已注册的阶段 id 创建课程策略实例。"""
    stage = int(stage)
    try:
        strategy_cls = STRATEGY_MAP[stage]
    except KeyError as exc:
        known = ", ".join(str(s) for s in sorted(STRATEGY_MAP))
        raise ValueError(f"Unknown curriculum stage {stage}. Known stages: {known}") from exc
    return strategy_cls(env_config)


def registered_stage_ids() -> Tuple[int, ...]:
    return tuple(sorted(STRATEGY_MAP))


def get_strategy_class(stage: int) -> Type[CurriculumStrategy]:
    stage = int(stage)
    try:
        return STRATEGY_MAP[stage]
    except KeyError as exc:
        known = ", ".join(str(s) for s in sorted(STRATEGY_MAP))
        raise ValueError(f"Unknown curriculum stage {stage}. Known stages: {known}") from exc


def get_stage_label(stage: int) -> str:
    return str(get_strategy_class(stage).label)


def get_stage_short_label(stage: int) -> str:
    return str(get_strategy_class(stage).short_label)


def is_hover_stage(stage: int) -> bool:
    return bool(get_strategy_class(stage).hover_stage)


def episode_info_keywords() -> Tuple[str, ...]:
    return ("success", "episode_stage", "train_score", "eval_score", "termination")


STAGE_LABELS = {stage: get_stage_label(stage) for stage in registered_stage_ids()}
STAGE_SHORT_LABELS = {
    stage: get_stage_short_label(stage)
    for stage in registered_stage_ids()
}
