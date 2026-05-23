# 功能：评估和模型筛选共用的距离指标评分工具。

from __future__ import annotations

import math

AVG_MIN_DIST_ENTRY_HORIZ = 0.50
AVG_MIN_DIST_ENTRY_VERT = 0.50
AVG_MIN_DIST_ZERO_DEFAULT = 0.20

DIST_SCORE_FULL = 0.005
MIN_DIST_SCORE_ZERO = 0.050
AVG_MIN_DIST_SCORE_ZERO = 0.150


def distance_score(distance: float, *, full: float, zero: float) -> float:
    """Map a distance in meters to a 0-100 score with linear falloff."""
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(d):
        return 0.0
    if d <= full:
        return 100.0
    if d >= zero:
        return 0.0
    return float(100.0 * (zero - d) / (zero - full))


def min_dist_score(distance: float) -> float:
    return distance_score(
        distance,
        full=DIST_SCORE_FULL,
        zero=MIN_DIST_SCORE_ZERO,
    )


def avg_min_dist_score(distance: float, valid: bool) -> float:
    if not bool(valid):
        return 0.0
    return distance_score(
        distance,
        full=DIST_SCORE_FULL,
        zero=AVG_MIN_DIST_SCORE_ZERO,
    )
