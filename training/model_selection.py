# 功能：训练阶段候选模型采样，以及阶段结束后的并行复评排序。

from __future__ import annotations

import gc
import logging
import math
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from configs.env_config import EnvConfig
from configs.train_config import (
    BEST_MODEL_EVAL_ENV_BLOCK,
    TrainConfig,
    normalize_best_model_eval_env_count,
)
from curriculum.strategies import get_strategy_class
from utils.distance_metrics import (
    AVG_MIN_DIST_ENTRY_HORIZ,
    AVG_MIN_DIST_ENTRY_VERT,
    AVG_MIN_DIST_ZERO_DEFAULT,
    avg_min_dist_score as compute_avg_min_dist_score,
    min_dist_score as compute_min_dist_score,
)

if TYPE_CHECKING:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecMonitor

logger = logging.getLogger(__name__)

FAILURE_TERMINATIONS = {"oob", "below_ground", "crashed"}
ROUND1_EPISODES = 10
ROUND2_EPISODES = 20
FINAL_ROUND_SEED_OFFSET = ROUND1_EPISODES + ROUND2_EPISODES


def _progress_tqdm():
    """Return a Rich-styled tqdm when available, otherwise plain tqdm."""
    try:
        from tqdm import TqdmExperimentalWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", TqdmExperimentalWarning)
            from tqdm.rich import tqdm
        return tqdm
    except Exception:
        from tqdm import tqdm
        return tqdm


def _log_file_only(record_logger: logging.Logger, message: str, level: int = logging.INFO) -> None:
    """Write progress messages to file logs without colliding with tqdm output."""
    root = logging.getLogger()
    record = record_logger.makeRecord(
        record_logger.name,
        level,
        __file__,
        0,
        message,
        args=(),
        exc_info=None,
    )
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and level >= handler.level:
            handler.handle(record)


@dataclass
class ModelCandidate:
    stage: int
    interval: int
    kind: str
    timestep: int
    candidate_reward: float
    path: str


@dataclass
class EvaluatedModel:
    rank: int
    stage: int
    candidate_kind: str
    source_timestep: int
    candidate_reward: float
    episodes: int
    avg_train_score: float
    avg_eval_score: float
    avg_min_dist_score: float
    min_dist_score: float
    combined_score: float
    success_rate: float
    failure_count: int
    avg_reward: float
    avg_length: float
    mean_min_dist: float
    mean_avg_min_dist: float
    avg_min_dist_valid_rate: float
    marker: str
    source_path: str
    final_path: str


def _info_stage(info: dict, episode_info: Optional[dict]) -> int:
    if episode_info is not None and "stage" in episode_info:
        return int(episode_info.get("stage", 0))
    return int(info.get("stage", 0))


def _safe_score(value: float) -> str:
    return f"{float(value):07.2f}".replace("-", "m").replace(".", "d")


def _safe_reward(value: float) -> str:
    prefix = "p" if value >= 0.0 else "m"
    return f"{prefix}{abs(float(value)):.1f}".replace(".", "d")


def _candidate_name(candidate: ModelCandidate) -> str:
    rew = _safe_reward(candidate.candidate_reward)
    return (
        f"s{candidate.stage}_{candidate.kind}_i{candidate.interval:02d}_"
        f"ts{candidate.timestep:010d}_rew{rew}"
    )


def _replace_zip(src_no_ext: str, dst_no_ext: str) -> None:
    shutil.copyfile(src_no_ext + ".zip", dst_no_ext + ".zip")


def _remove_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)


def _remove_stage_top_models(top_dir: str, stage: int) -> None:
    if not os.path.isdir(top_dir):
        return
    prefix = f"model_stage{int(stage)}_best_"
    for name in os.listdir(top_dir):
        if name.startswith(prefix) and name.endswith(".zip"):
            os.remove(os.path.join(top_dir, name))


def _remove_temp_top_models(top_dir: str, stage: int) -> None:
    if not os.path.isdir(top_dir):
        return
    prefix = f".tmp_model_stage{int(stage)}_best_"
    for name in os.listdir(top_dir):
        if name.startswith(prefix) and name.endswith(".zip"):
            os.remove(os.path.join(top_dir, name))


class BestModelCandidateCallback(BaseCallback):
    """
    Collect per-stage model candidates without running extra evaluation during training.

    Candidates are collected in two ways:
      - one grid snapshot at each evenly spaced interval boundary
      - top-M episode-reward snapshots inside each interval
    """

    def __init__(
        self,
        *,
        save_root: str,
        grid_count: int,
        interval_top_m: int,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_root = save_root
        self.grid_count = max(1, int(grid_count))
        self.interval_top_m = max(0, int(interval_top_m))
        self._enabled = True
        self._stage = 1
        self._stage_start_step = 0
        self._stage_budget = 1
        self._next_grid_interval = 1
        self._candidates: Dict[int, List[ModelCandidate]] = {}
        self._grid_candidates: Dict[tuple, ModelCandidate] = {}
        self._interval_top: Dict[tuple, List[ModelCandidate]] = {}

    def start_stage(self, stage: int, start_step: int, stage_budget: int) -> None:
        self._stage = int(stage)
        self._stage_start_step = int(start_step)
        self._stage_budget = max(1, int(stage_budget))
        self._next_grid_interval = 1

    def candidates_for_stage(self, stage: int) -> List[ModelCandidate]:
        stage = int(stage)
        candidates = self._candidates.get(stage, [])
        unique: Dict[str, ModelCandidate] = {}
        for candidate in candidates:
            unique[candidate.path] = candidate
        return sorted(
            unique.values(),
            key=lambda c: (c.timestep, c.kind, c.candidate_reward),
        )

    def save_stage_final(self, stage: int, model: "SAC") -> Optional[ModelCandidate]:
        if not self._enabled:
            return None
        stage = int(stage)
        interval = self.grid_count
        old_grid = self._grid_candidates.get((stage, interval))
        if old_grid is not None:
            self._remove_candidate(old_grid)
        path = self._candidate_path(
            ModelCandidate(
                stage=stage,
                interval=interval,
                kind="stagefinal",
                timestep=int(model.num_timesteps),
                candidate_reward=0.0,
                path="",
            )
        )
        model.save(path)
        candidate = ModelCandidate(
            stage=stage,
            interval=interval,
            kind="stagefinal",
            timestep=int(model.num_timesteps),
            candidate_reward=0.0,
            path=path,
        )
        self._grid_candidates[(stage, interval)] = candidate
        self._register_candidate(candidate)
        return candidate

    def _candidate_dir(self, stage: int) -> str:
        path = os.path.join(self.save_root, f"stage{int(stage)}_candidates")
        os.makedirs(path, exist_ok=True)
        return path

    def _candidate_path(self, candidate: ModelCandidate) -> str:
        base = _candidate_name(candidate)
        return os.path.join(self._candidate_dir(candidate.stage), base)

    def _stage_elapsed(self) -> int:
        return max(0, int(self.num_timesteps) - self._stage_start_step)

    def _interval_for_elapsed(self, elapsed: int) -> int:
        if elapsed <= 0:
            return 1
        ratio = min(1.0, float(elapsed) / float(self._stage_budget))
        return max(1, min(self.grid_count, int(math.ceil(ratio * self.grid_count))))

    def _grid_boundary_step(self, interval: int) -> int:
        return int(math.ceil(self._stage_budget * interval / self.grid_count))

    def _register_candidate(self, candidate: ModelCandidate) -> None:
        self._candidates.setdefault(candidate.stage, []).append(candidate)

    def _remove_candidate(self, candidate: ModelCandidate) -> None:
        try:
            os.remove(candidate.path + ".zip")
        except FileNotFoundError:
            pass
        self._candidates[candidate.stage] = [
            item for item in self._candidates.get(candidate.stage, [])
            if item.path != candidate.path
        ]

    def _save_candidate(
        self,
        *,
        kind: str,
        interval: int,
        reward: float,
    ) -> ModelCandidate:
        candidate = ModelCandidate(
            stage=self._stage,
            interval=int(interval),
            kind=str(kind),
            timestep=int(self.num_timesteps),
            candidate_reward=float(reward),
            path="",
        )
        candidate.path = self._candidate_path(candidate)
        self.model.save(candidate.path)
        self._register_candidate(candidate)
        return candidate

    def _save_due_grid_candidates(self) -> None:
        elapsed = self._stage_elapsed()
        while self._next_grid_interval <= self.grid_count:
            boundary = self._grid_boundary_step(self._next_grid_interval)
            if elapsed < boundary:
                break
            candidate = self._save_candidate(
                kind="grid",
                interval=self._next_grid_interval,
                reward=0.0,
            )
            self._grid_candidates[(self._stage, self._next_grid_interval)] = candidate
            if self.verbose >= 1:
                _log_file_only(
                    logger,
                    (
                        "Best-model grid candidate saved: "
                        f"stage={self._stage} interval={self._next_grid_interval} "
                        f"ts={self.num_timesteps}"
                    ),
                )
            self._next_grid_interval += 1

    def _maybe_save_top_reward_candidate(self, interval: int, reward: float) -> None:
        if self.interval_top_m <= 0:
            return
        key = (self._stage, int(interval))
        bucket = self._interval_top.setdefault(key, [])
        if len(bucket) >= self.interval_top_m:
            worst = min(bucket, key=lambda c: (c.candidate_reward, c.timestep))
            if reward <= worst.candidate_reward:
                return
        candidate = self._save_candidate(
            kind="rewardtop",
            interval=interval,
            reward=reward,
        )
        bucket.append(candidate)
        bucket.sort(key=lambda c: (c.candidate_reward, c.timestep), reverse=True)
        while len(bucket) > self.interval_top_m:
            removed = bucket.pop()
            self._remove_candidate(removed)

    def _on_step(self) -> bool:
        if not self._enabled:
            return True

        self._save_due_grid_candidates()

        rewards_by_timestep: Dict[int, float] = {}
        stages_by_timestep: Dict[int, int] = {}
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue
            ep_stage = _info_stage(info, ep)
            if ep_stage != self._stage:
                continue
            ts = int(self.num_timesteps)
            reward = float(ep.get("r", 0.0))
            rewards_by_timestep[ts] = max(rewards_by_timestep.get(ts, reward), reward)
            stages_by_timestep[ts] = ep_stage

        for ts, reward in rewards_by_timestep.items():
            if stages_by_timestep.get(ts) != self._stage:
                continue
            interval = self._interval_for_elapsed(ts - self._stage_start_step)
            self._maybe_save_top_reward_candidate(interval, reward)

        return True


def _format_duration(seconds: float) -> str:
    if not math.isfinite(float(seconds)) or seconds < 0.0:
        return "--:--"
    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _empty_summary() -> dict:
    return {
        "episodes": 0,
        "avg_reward": 0.0,
        "avg_length": 0.0,
        "success_rate": 0.0,
        "avg_train_score": 0.0,
        "avg_eval_score": 0.0,
        "mean_min_dist": 0.0,
        "mean_avg_min_dist": 0.0,
        "avg_min_dist_valid_rate": 0.0,
        "min_dist_score": 0.0,
        "avg_min_dist_score": 0.0,
        "failure_count": 0,
    }


def _episode_result_from_info(
    *,
    episode_idx: int,
    info: dict,
    min_dist: float,
    avg_min_dist: float,
    avg_min_valid: bool,
) -> Optional[dict]:
    ep = info.get("episode")
    if ep is None:
        return None

    termination = str(info.get("termination", ep.get("termination", "none")))
    min_score = compute_min_dist_score(min_dist)
    avg_min_score = compute_avg_min_dist_score(
        avg_min_dist,
        avg_min_valid,
    )
    return {
        "episode": int(episode_idx),
        "reward": float(ep.get("r", 0.0)),
        "length": int(ep.get("l", 0)),
        "success": bool(ep.get("success", info.get("success", False))),
        "train_score": float(ep.get("train_score", info.get("train_score", 0.0))),
        "eval_score": float(ep.get("eval_score", info.get("eval_score", 0.0))),
        "min_dist": float(min_dist),
        "avg_min_dist": float(avg_min_dist),
        "avg_min_dist_valid": bool(avg_min_valid),
        "min_dist_score": min_score,
        "avg_min_dist_score": avg_min_score,
        "termination": termination,
        "failure": termination in FAILURE_TERMINATIONS,
    }


def _summarize_episode_results(results: List[dict]) -> dict:
    if not results:
        return _empty_summary()

    return {
        "episodes": len(results),
        "avg_reward": float(np.mean([r["reward"] for r in results])),
        "avg_length": float(np.mean([r["length"] for r in results])),
        "success_rate": float(np.mean([float(r["success"]) for r in results])),
        "avg_train_score": float(np.mean([r["train_score"] for r in results])),
        "avg_eval_score": float(np.mean([r["eval_score"] for r in results])),
        "mean_min_dist": float(np.mean([r["min_dist"] for r in results])),
        "mean_avg_min_dist": float(np.mean([r["avg_min_dist"] for r in results])),
        "avg_min_dist_valid_rate": float(
            np.mean([float(r["avg_min_dist_valid"]) for r in results])
        ),
        "min_dist_score": float(np.mean([r["min_dist_score"] for r in results])),
        "avg_min_dist_score": float(
            np.mean([r["avg_min_dist_score"] for r in results])
        ),
        "failure_count": int(sum(1 for r in results if r["failure"])),
    }


@dataclass
class CandidateEvalState:
    candidate: ModelCandidate
    original_index: int
    episodes: int = 0
    reward_total: float = 0.0
    length_total: float = 0.0
    success_total: float = 0.0
    train_score_total: float = 0.0
    eval_score_total: float = 0.0
    min_dist_total: float = 0.0
    avg_min_dist_total: float = 0.0
    avg_min_dist_valid_total: float = 0.0
    min_dist_score_total: float = 0.0
    avg_min_dist_score_total: float = 0.0
    failure_count: int = 0
    combined: float = 0.0
    marker: str = "clean"

    def add_summary(self, summary: dict) -> None:
        n = int(summary.get("episodes", 0))
        if n <= 0:
            return
        self.episodes += n
        self.reward_total += float(summary["avg_reward"]) * n
        self.length_total += float(summary["avg_length"]) * n
        self.success_total += float(summary["success_rate"]) * n
        self.train_score_total += float(summary["avg_train_score"]) * n
        self.eval_score_total += float(summary["avg_eval_score"]) * n
        self.min_dist_total += float(summary["mean_min_dist"]) * n
        self.avg_min_dist_total += float(summary["mean_avg_min_dist"]) * n
        self.avg_min_dist_valid_total += float(summary["avg_min_dist_valid_rate"]) * n
        self.min_dist_score_total += float(summary["min_dist_score"]) * n
        self.avg_min_dist_score_total += float(summary["avg_min_dist_score"]) * n
        self.failure_count += int(summary["failure_count"])

    def summary(self) -> dict:
        if self.episodes <= 0:
            return _empty_summary()
        n = float(self.episodes)
        return {
            "episodes": int(self.episodes),
            "avg_reward": self.reward_total / n,
            "avg_length": self.length_total / n,
            "success_rate": self.success_total / n,
            "avg_train_score": self.train_score_total / n,
            "avg_eval_score": self.eval_score_total / n,
            "mean_min_dist": self.min_dist_total / n,
            "mean_avg_min_dist": self.avg_min_dist_total / n,
            "avg_min_dist_valid_rate": self.avg_min_dist_valid_total / n,
            "min_dist_score": self.min_dist_score_total / n,
            "avg_min_dist_score": self.avg_min_dist_score_total / n,
            "failure_count": int(self.failure_count),
        }


def _combined_score(
    summary: dict,
    *,
    train_weight: float,
    eval_weight: float,
    avg_min_dist_weight: float,
    min_dist_weight: float,
) -> float:
    return float(
        float(summary["avg_train_score"]) * train_weight
        + float(summary["avg_eval_score"]) * eval_weight
        + float(summary["avg_min_dist_score"]) * avg_min_dist_weight
        + float(summary["min_dist_score"]) * min_dist_weight
    )


def _update_state_score(
    state: CandidateEvalState,
    *,
    train_weight: float,
    eval_weight: float,
    avg_min_dist_weight: float,
    min_dist_weight: float,
) -> dict:
    summary = state.summary()
    state.combined = _combined_score(
        summary,
        train_weight=train_weight,
        eval_weight=eval_weight,
        avg_min_dist_weight=avg_min_dist_weight,
        min_dist_weight=min_dist_weight,
    )
    state.marker = "fail" if int(summary["failure_count"]) > 0 else "clean"
    return summary


def _evaluated_sort_key(
    *,
    marker: str,
    summary: dict,
    combined: float,
    candidate: ModelCandidate,
) -> tuple:
    marker_rank = 0 if marker == "clean" else 1
    return (
        marker_rank,
        int(summary["failure_count"]),
        -float(combined),
        -float(summary["avg_eval_score"]),
        -float(summary["avg_train_score"]),
        -float(summary["avg_min_dist_score"]),
        -float(summary["min_dist_score"]),
        -int(candidate.timestep),
    )


def _score_sort_key(state: CandidateEvalState) -> tuple:
    summary = state.summary()
    return (
        -float(state.combined),
        -float(summary["avg_eval_score"]),
        -float(summary["avg_train_score"]),
        -float(summary["avg_min_dist_score"]),
        -float(summary["min_dist_score"]),
        -int(state.candidate.timestep),
    )


def _final_state_sort_key(state: CandidateEvalState) -> tuple:
    summary = state.summary()
    return _evaluated_sort_key(
        marker=state.marker,
        summary=summary,
        combined=state.combined,
        candidate=state.candidate,
    )


def _ceil_fraction(total: int, fraction: float) -> int:
    return max(1, int(math.ceil(max(0, int(total)) * float(fraction))))


class BestModelEvalProgress:
    """Real-time progress display for per-stage best-model reevaluation."""

    def __init__(
        self,
        *,
        stage: int,
        round_label: str,
        candidates_total: int,
        episodes_per_candidate: int,
        train_weight: float,
        eval_weight: float,
        avg_min_dist_weight: float,
        min_dist_weight: float,
    ):
        self.stage = int(stage)
        self.round_label = str(round_label)
        self.candidates_total = max(0, int(candidates_total))
        self.episodes_per_candidate = max(1, int(episodes_per_candidate))
        self.train_weight = float(train_weight)
        self.eval_weight = float(eval_weight)
        self.avg_min_dist_weight = float(avg_min_dist_weight)
        self.min_dist_weight = float(min_dist_weight)
        self.total_episodes = self.candidates_total * self.episodes_per_candidate
        self.completed_episodes = 0

        self._bar = None
        self._tqdm_cls = None
        self._interactive = bool(sys.stdout.isatty())
        self._started_at = 0.0
        self._batch_started_at = 0.0
        self._batch_start_index = 0
        self._batch_end_index = 0
        self._active_slots = 0
        self._candidate_index = 0
        self._candidate: Optional[ModelCandidate] = None

        self._best_key: Optional[tuple] = None
        self._best_label = "none"

    def __enter__(self) -> "BestModelEvalProgress":
        self._started_at = time.monotonic()
        if self._interactive:
            try:
                tqdm = _progress_tqdm()
                self._tqdm_cls = tqdm
                self._bar = tqdm(
                    total=self.total_episodes,
                    desc=f"Stage {self.stage} {self.round_label}",
                    unit="eps",
                    dynamic_ncols=True,
                    leave=True,
                    file=sys.stdout,
                    mininterval=0.5,
                )
            except Exception:
                self._interactive = False
                self._bar = None

        if not self._interactive:
            logger.info(
                "Best-model eval progress: stage=%s candidates=%s "
                "round=%s episodes_per_candidate=%s total_episodes=%s",
                self.stage,
                self.candidates_total,
                self.round_label,
                self.episodes_per_candidate,
                self.total_episodes,
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def on_batch_start(
        self,
        *,
        start_index: int,
        end_index: int,
        active_slots: int,
    ) -> None:
        self._batch_start_index = int(start_index)
        self._batch_end_index = int(end_index)
        self._active_slots = int(active_slots)
        self._batch_started_at = time.monotonic()
        self._refresh_bar()

    def on_episode(self, result: dict) -> None:
        self.completed_episodes += 1
        if self._bar is not None:
            self._bar.update(1)
            self._refresh_bar()

    def on_candidate_done(
        self,
        *,
        index: int,
        candidate: ModelCandidate,
        summary: dict,
        combined: float,
        marker: str,
        elapsed: float,
    ) -> None:
        self._candidate_index = int(index)
        self._candidate = candidate
        key = _evaluated_sort_key(
            marker=marker,
            summary=summary,
            combined=combined,
            candidate=candidate,
        )
        if self._best_key is None or key < self._best_key:
            self._best_key = key
            self._best_label = f"{marker}:{float(combined):.2f}"

        candidate_tag = f"{candidate.kind}@{candidate.timestep}"

        message = (
            f"[{self._candidate_index}/{self.candidates_total}] {marker} "
            f"score={float(combined):.2f} "
            f"train={float(summary['avg_train_score']):.1f} "
            f"eval={float(summary['avg_eval_score']):.1f} "
            f"avgD={float(summary['avg_min_dist_score']):.1f} "
            f"minD={float(summary['min_dist_score']):.1f} "
            f"fail={int(summary['failure_count'])} "
            f"eps={int(summary['episodes'])} "
            f"time={elapsed:.1f}s "
            f"{candidate_tag}"
        )
        file_message = (
            f"{message} "
            f"sr={float(summary['success_rate']):.1%} "
            f"avg_len={float(summary['avg_length']):.0f}"
        )

        if self._interactive and self._tqdm_cls is not None:
            self._tqdm_cls.write(message, file=sys.stdout)
            _log_file_only(logger, file_message)
        else:
            logger.info("%s | %s", self._noninteractive_prefix(), message)

        self._refresh_bar()

    def _noninteractive_prefix(self) -> str:
        elapsed = max(time.monotonic() - self._started_at, 1e-9)
        rate = self.completed_episodes / elapsed
        remaining = self.total_episodes - self.completed_episodes
        eta = _format_duration(remaining / rate) if rate > 0.0 else "--:--"
        return (
            "Best-model eval progress: "
            f"{self.round_label} | "
            f"batch={self._batch_start_index}-{self._batch_end_index}/"
            f"{self.candidates_total} | "
            f"active={self._active_slots} | "
            f"{self.completed_episodes}/{self.total_episodes} eps | "
            f"{rate:.2f} ep/s | ETA {eta}"
        )

    def _refresh_bar(self) -> None:
        if self._bar is None:
            return

        postfix = (
            f"batch={self._batch_start_index}-{self._batch_end_index}/"
            f"{self.candidates_total} "
            f"active={self._active_slots} "
            f"eps={self.completed_episodes}/{self.total_episodes} "
            f"best={self._best_label}"
        )
        self._bar.set_postfix_str(postfix, refresh=False)


def _build_eval_env(
    env_config: EnvConfig,
    *,
    stage: int,
    n_envs: int,
    seed: int,
) -> "VecMonitor":
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from training.env_factory import make_eval_env_fn
    from training.spawn_vec_env import SpawnSafeSubprocVecEnv
    from curriculum.strategies import episode_info_keywords

    fns = [
        make_eval_env_fn(env_config, rank=rank, seed=seed, stage=stage)
        for rank in range(n_envs)
    ]
    if n_envs > 1:
        env = SpawnSafeSubprocVecEnv(fns, start_method="spawn")
    else:
        env = DummyVecEnv(fns)
    return VecMonitor(env, info_keywords=episode_info_keywords())


def _evaluate_loaded_model(
    model: "SAC",
    vec_env: "VecMonitor",
    *,
    n_episodes: int,
    seed: int,
    progress: Optional[BestModelEvalProgress] = None,
) -> dict:
    model.set_env(vec_env)
    n_envs = int(vec_env.num_envs)
    results = []
    episode_idx = 0
    for batch_start in range(0, n_episodes, n_envs):
        batch_size = min(n_envs, n_episodes - batch_start)
        vec_env.seed(seed + batch_start)
        obs = vec_env.reset()
        active = np.zeros(n_envs, dtype=bool)
        active[:batch_size] = True
        min_dists = np.full(n_envs, np.inf, dtype=np.float64)
        avg_min_dist_active = np.zeros(n_envs, dtype=bool)
        avg_min_dist_sums = np.zeros(n_envs, dtype=np.float64)
        avg_min_dist_counts = np.zeros(n_envs, dtype=np.int64)
        while np.any(active):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            for env_idx, info in enumerate(infos):
                if not active[env_idx]:
                    continue
                step_dist = float(info.get("metric/dist", np.inf))
                horiz_err = float(info.get("metric/horiz_err", np.inf))
                vert_err = float(info.get("metric/vert_err", np.inf))
                if np.isfinite(step_dist):
                    min_dists[env_idx] = min(float(min_dists[env_idx]), step_dist)
                if (
                    not bool(avg_min_dist_active[env_idx])
                    and horiz_err <= AVG_MIN_DIST_ENTRY_HORIZ
                    and vert_err <= AVG_MIN_DIST_ENTRY_VERT
                ):
                    avg_min_dist_active[env_idx] = True
                if bool(avg_min_dist_active[env_idx]) and np.isfinite(step_dist):
                    avg_min_dist_sums[env_idx] += step_dist
                    avg_min_dist_counts[env_idx] += 1

                if not dones[env_idx]:
                    continue
                avg_min_valid = int(avg_min_dist_counts[env_idx]) > 0
                avg_min_dist = (
                    float(avg_min_dist_sums[env_idx])
                    / float(avg_min_dist_counts[env_idx])
                    if avg_min_valid else AVG_MIN_DIST_ZERO_DEFAULT
                )
                next_episode_idx = episode_idx + 1
                result = _episode_result_from_info(
                    episode_idx=next_episode_idx,
                    info=info,
                    min_dist=float(min_dists[env_idx]),
                    avg_min_dist=avg_min_dist,
                    avg_min_valid=bool(avg_min_valid),
                )
                if result is not None:
                    episode_idx = next_episode_idx
                    results.append(result)
                    if progress is not None:
                        progress.on_episode(result)
                active[env_idx] = False

    return _summarize_episode_results(results)


@dataclass
class _EvalSlot:
    index: int
    vec_env: "VecMonitor"
    model: Optional["SAC"] = None
    state: Optional[CandidateEvalState] = None
    candidate_index: int = 0
    started_at: float = 0.0
    results: List[dict] = field(default_factory=list)
    episode_idx: int = 0


@dataclass
class _EvalBlockRun:
    slot: _EvalSlot
    obs: Any
    active: np.ndarray
    min_dists: np.ndarray
    avg_min_dist_active: np.ndarray
    avg_min_dist_sums: np.ndarray
    avg_min_dist_counts: np.ndarray


def _start_eval_block(
    slot: _EvalSlot,
    *,
    n_episodes: int,
    block_start: int,
    seed: int,
) -> Optional[_EvalBlockRun]:
    n_envs = int(slot.vec_env.num_envs)
    batch_size = min(n_envs, int(n_episodes) - int(block_start))
    if batch_size <= 0:
        return None

    slot.vec_env.seed(int(seed) + int(block_start))
    obs = slot.vec_env.reset()
    active = np.zeros(n_envs, dtype=bool)
    active[:batch_size] = True
    return _EvalBlockRun(
        slot=slot,
        obs=obs,
        active=active,
        min_dists=np.full(n_envs, np.inf, dtype=np.float64),
        avg_min_dist_active=np.zeros(n_envs, dtype=bool),
        avg_min_dist_sums=np.zeros(n_envs, dtype=np.float64),
        avg_min_dist_counts=np.zeros(n_envs, dtype=np.int64),
    )


def _process_eval_block_infos(
    run: _EvalBlockRun,
    *,
    dones: np.ndarray,
    infos: List[dict],
    progress: Optional[BestModelEvalProgress],
) -> None:
    for env_idx, info in enumerate(infos):
        if not run.active[env_idx]:
            continue

        step_dist = float(info.get("metric/dist", np.inf))
        horiz_err = float(info.get("metric/horiz_err", np.inf))
        vert_err = float(info.get("metric/vert_err", np.inf))
        if np.isfinite(step_dist):
            run.min_dists[env_idx] = min(float(run.min_dists[env_idx]), step_dist)
        if (
            not bool(run.avg_min_dist_active[env_idx])
            and horiz_err <= AVG_MIN_DIST_ENTRY_HORIZ
            and vert_err <= AVG_MIN_DIST_ENTRY_VERT
        ):
            run.avg_min_dist_active[env_idx] = True
        if bool(run.avg_min_dist_active[env_idx]) and np.isfinite(step_dist):
            run.avg_min_dist_sums[env_idx] += step_dist
            run.avg_min_dist_counts[env_idx] += 1

        if not dones[env_idx]:
            continue

        avg_min_valid = int(run.avg_min_dist_counts[env_idx]) > 0
        avg_min_dist = (
            float(run.avg_min_dist_sums[env_idx])
            / float(run.avg_min_dist_counts[env_idx])
            if avg_min_valid else AVG_MIN_DIST_ZERO_DEFAULT
        )
        next_episode_idx = run.slot.episode_idx + 1
        result = _episode_result_from_info(
            episode_idx=next_episode_idx,
            info=info,
            min_dist=float(run.min_dists[env_idx]),
            avg_min_dist=avg_min_dist,
            avg_min_valid=bool(avg_min_valid),
        )
        if result is not None:
            run.slot.episode_idx = next_episode_idx
            run.slot.results.append(result)
            if progress is not None:
                progress.on_episode(result)
        run.active[env_idx] = False


def _run_eval_block_concurrently(
    runs: List[_EvalBlockRun],
    *,
    progress: Optional[BestModelEvalProgress],
) -> None:
    while any(bool(np.any(run.active)) for run in runs):
        active_runs = [run for run in runs if bool(np.any(run.active))]
        for run in active_runs:
            if run.slot.model is None:
                raise RuntimeError("Cannot step a best-model eval slot without a model.")
            action, _ = run.slot.model.predict(run.obs, deterministic=True)
            run.slot.vec_env.step_async(action)

        for run in active_runs:
            obs, _rewards, dones, infos = run.slot.vec_env.step_wait()
            run.obs = obs
            _process_eval_block_infos(
                run,
                dones=dones,
                infos=list(infos),
                progress=progress,
            )


def _evaluate_loaded_model_batch(
    slots: List[_EvalSlot],
    *,
    n_episodes: int,
    seed: int,
    progress: Optional[BestModelEvalProgress] = None,
) -> List[dict]:
    for slot in slots:
        if slot.model is None:
            raise RuntimeError("Cannot evaluate an empty best-model eval slot.")
        slot.model.set_env(slot.vec_env)
        slot.results = []
        slot.episode_idx = 0

    for block_start in range(0, int(n_episodes), BEST_MODEL_EVAL_ENV_BLOCK):
        runs = [
            run for slot in slots
            if (run := _start_eval_block(
                slot,
                n_episodes=n_episodes,
                block_start=block_start,
                seed=seed,
            )) is not None
        ]
        _run_eval_block_concurrently(runs, progress=progress)

    return [_summarize_episode_results(slot.results) for slot in slots]


def evaluate_stage_candidates(
    *,
    candidates: List[ModelCandidate],
    stage: int,
    env_config: EnvConfig,
    train_config: TrainConfig,
    device: str,
) -> List[EvaluatedModel]:
    from stable_baselines3 import SAC

    if not train_config.best_model_selection_enabled:
        return []

    if not candidates:
        logger.info("No best-model candidates collected for stage %s; skipping evaluation.", stage)
        return []

    metric_keys = tuple(get_strategy_class(stage).episode_metrics)
    if "train_score" not in metric_keys or "eval_score" not in metric_keys:
        logger.warning(
            "Stage %s does not expose train_score/eval_score; skipping best-model evaluation.",
            stage,
        )
        return []

    valid_candidates: List[ModelCandidate] = []
    for candidate in candidates:
        if os.path.exists(candidate.path + ".zip"):
            valid_candidates.append(candidate)
        else:
            logger.warning("Candidate zip missing; skipping: %s.zip", candidate.path)

    if not valid_candidates:
        logger.info(
            "No existing best-model candidate zips found for stage %s; skipping evaluation.",
            stage,
        )
        return []

    eval_episodes = max(1, int(train_config.best_model_eval_episodes))
    configured_envs = normalize_best_model_eval_env_count(
        int(train_config.best_model_eval_envs_by_stage.get(stage, 1))
    )
    parallel_candidates = max(1, configured_envs // BEST_MODEL_EVAL_ENV_BLOCK)
    model_root = os.path.join(train_config.model_dir, train_config.exp_name)
    top_dir = os.path.join(model_root, "top_models")
    candidate_dir = os.path.join(model_root, f"stage{stage}_candidates")
    evaluated_dir = os.path.join(model_root, f"stage{stage}_evaluated")
    keep_top_n = max(1, int(train_config.best_model_keep_top_n))

    logger.info(
        "Evaluating %s best-model candidates for stage %s: "
        "round1=%s round2=%s final_episodes=%s eval_env_budget=%s "
        "slot_envs=%s parallel_candidates=%s",
        len(valid_candidates),
        stage,
        ROUND1_EPISODES,
        ROUND2_EPISODES,
        eval_episodes,
        configured_envs,
        BEST_MODEL_EVAL_ENV_BLOCK,
        parallel_candidates,
    )

    train_weight = float(train_config.best_model_train_score_weight)
    eval_weight = float(train_config.best_model_eval_score_weight)
    avg_min_dist_weight = float(train_config.best_model_avg_min_dist_score_weight)
    min_dist_weight = float(train_config.best_model_min_dist_score_weight)
    seed_base = int(train_config.seed) + int(stage) * 1_000_000

    states = [
        CandidateEvalState(candidate=candidate, original_index=idx)
        for idx, candidate in enumerate(valid_candidates, start=1)
    ]

    def _release_models(models: List[object]) -> None:
        models.clear()
        gc.collect()
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _close_slots(slots: List[_EvalSlot]) -> None:
        for slot in slots:
            slot.vec_env.close()

    def _evaluate_round(
        *,
        round_label: str,
        round_states: List[CandidateEvalState],
        episodes: int,
        seed_offset: int,
    ) -> bool:
        if not round_states:
            return True

        round_parallel = max(1, min(parallel_candidates, len(round_states)))
        round_seed = seed_base + int(seed_offset)
        logger.info(
            "Best-model %s: candidates=%s episodes=%s env_budget=%s "
            "slot_envs=%s parallel_candidates=%s seed=%s",
            round_label,
            len(round_states),
            episodes,
            configured_envs,
            BEST_MODEL_EVAL_ENV_BLOCK,
            round_parallel,
            round_seed,
        )

        slots: List[_EvalSlot] = []
        try:
            for slot_idx in range(round_parallel):
                slots.append(_EvalSlot(
                    index=slot_idx,
                    vec_env=_build_eval_env(
                        env_config,
                        stage=stage,
                        n_envs=BEST_MODEL_EVAL_ENV_BLOCK,
                        seed=round_seed,
                    ),
                ))
        except NotImplementedError:
            _close_slots(slots)
            logger.warning(
                "Stage %s evaluation is not implemented; skipping best-model evaluation.",
                stage,
            )
            return False
        except Exception:
            _close_slots(slots)
            raise

        try:
            with BestModelEvalProgress(
                stage=stage,
                round_label=round_label,
                candidates_total=len(round_states),
                episodes_per_candidate=episodes,
                train_weight=train_weight,
                eval_weight=eval_weight,
                avg_min_dist_weight=avg_min_dist_weight,
                min_dist_weight=min_dist_weight,
            ) as progress:
                indexed_states = list(enumerate(round_states, start=1))
                for batch_start in range(0, len(indexed_states), round_parallel):
                    batch = indexed_states[batch_start:batch_start + round_parallel]
                    progress.on_batch_start(
                        start_index=batch[0][0],
                        end_index=batch[-1][0],
                        active_slots=len(batch),
                    )
                    active_slots = slots[:len(batch)]
                    loaded_models: List[object] = []
                    try:
                        for slot, (candidate_index, state) in zip(
                            active_slots,
                            batch,
                            strict=True,
                        ):
                            slot.state = state
                            slot.candidate_index = int(candidate_index)
                            slot.started_at = time.monotonic()
                            slot.model = SAC.load(
                                state.candidate.path,
                                env=slot.vec_env,
                                device=device,
                            )
                            loaded_models.append(slot.model)

                        round_summaries = _evaluate_loaded_model_batch(
                            active_slots,
                            n_episodes=episodes,
                            seed=round_seed,
                            progress=progress,
                        )
                    except NotImplementedError:
                        logger.warning(
                            "Stage %s evaluation is not implemented; "
                            "skipping best-model evaluation.",
                            stage,
                        )
                        return False
                    finally:
                        for slot in active_slots:
                            slot.model = None
                        _release_models(loaded_models)

                    for slot, round_summary in zip(
                        active_slots,
                        round_summaries,
                        strict=True,
                    ):
                        if slot.state is None:
                            continue
                        slot.state.add_summary(round_summary)
                        cumulative = _update_state_score(
                            slot.state,
                            train_weight=train_weight,
                            eval_weight=eval_weight,
                            avg_min_dist_weight=avg_min_dist_weight,
                            min_dist_weight=min_dist_weight,
                        )
                        progress.on_candidate_done(
                            index=slot.candidate_index,
                            candidate=slot.state.candidate,
                            summary=cumulative,
                            combined=slot.state.combined,
                            marker=slot.state.marker,
                            elapsed=max(0.0, time.monotonic() - slot.started_at),
                        )
                        slot.state = None
                        slot.candidate_index = 0
                        slot.started_at = 0.0
        finally:
            _close_slots(slots)
        return True

    def _select_after_round1(
        round_states: List[CandidateEvalState],
    ) -> List[CandidateEvalState]:
        ranked = sorted(round_states, key=_score_sort_key)
        total = len(ranked)
        clean = [state for state in ranked if state.marker == "clean"]
        half = _ceil_fraction(total, 0.50)
        cap70 = _ceil_fraction(total, 0.70)

        if len(clean) <= half:
            selected = ranked[:half]
            rule = "top50_all"
        elif len(clean) > cap70:
            selected = clean[:cap70]
            rule = "top70_clean"
        else:
            selected = clean
            rule = "all_clean"

        logger.info(
            "Best-model round1 selected %s/%s for round2 | clean=%s "
            "half=%s cap70=%s rule=%s",
            len(selected),
            total,
            len(clean),
            half,
            cap70,
            rule,
        )
        return selected

    def _select_after_round2(
        round_states: List[CandidateEvalState],
    ) -> List[CandidateEvalState]:
        ranked = sorted(round_states, key=_score_sort_key)
        total = len(ranked)
        cap = _ceil_fraction(total, 0.30)
        clean = [state for state in ranked if state.marker == "clean"]

        if clean:
            selected = clean[:cap] if len(clean) > cap else clean
            rule = "clean"
        else:
            selected = ranked[:cap]
            rule = "top30_all_fail"

        logger.info(
            "Best-model round2 selected %s/%s for final | clean=%s cap30=%s rule=%s",
            len(selected),
            total,
            len(clean),
            cap,
            rule,
        )
        return selected

    if not _evaluate_round(
        round_label="round1",
        round_states=states,
        episodes=ROUND1_EPISODES,
        seed_offset=0,
    ):
        return []

    round2_states = _select_after_round1(states)
    if not _evaluate_round(
        round_label="round2",
        round_states=round2_states,
        episodes=ROUND2_EPISODES,
        seed_offset=ROUND1_EPISODES,
    ):
        return []

    final_states = _select_after_round2(round2_states)
    if not _evaluate_round(
        round_label="final",
        round_states=final_states,
        episodes=eval_episodes,
        seed_offset=FINAL_ROUND_SEED_OFFSET,
    ):
        return []

    if not final_states:
        return []

    final_states.sort(key=_final_state_sort_key)
    evaluated: List[EvaluatedModel] = []
    os.makedirs(top_dir, exist_ok=True)
    _remove_temp_top_models(top_dir, stage)
    copied_top_models = []

    for rank, state in enumerate(final_states[:keep_top_n], start=1):
        candidate = state.candidate
        summary = state.summary()
        marker = state.marker
        final_path = os.path.join(top_dir, f"model_stage{stage}_best_{rank}")
        temp_path = os.path.join(top_dir, f".tmp_model_stage{stage}_best_{rank}")
        _replace_zip(candidate.path, temp_path)
        copied_top_models.append((temp_path, final_path))
        evaluated.append(EvaluatedModel(
            rank=rank,
            stage=stage,
            candidate_kind=candidate.kind,
            source_timestep=candidate.timestep,
            candidate_reward=float(candidate.candidate_reward),
            episodes=int(summary["episodes"]),
            avg_train_score=float(summary["avg_train_score"]),
            avg_eval_score=float(summary["avg_eval_score"]),
            avg_min_dist_score=float(summary["avg_min_dist_score"]),
            min_dist_score=float(summary["min_dist_score"]),
            combined_score=float(state.combined),
            success_rate=float(summary["success_rate"]),
            failure_count=int(summary["failure_count"]),
            avg_reward=float(summary["avg_reward"]),
            avg_length=float(summary["avg_length"]),
            mean_min_dist=float(summary["mean_min_dist"]),
            mean_avg_min_dist=float(summary["mean_avg_min_dist"]),
            avg_min_dist_valid_rate=float(summary["avg_min_dist_valid_rate"]),
            marker=marker,
            source_path=candidate.path + ".zip",
            final_path=final_path + ".zip",
        ))

    _remove_stage_top_models(top_dir, stage)
    for temp_path, final_path in copied_top_models:
        os.replace(temp_path + ".zip", final_path + ".zip")

    _remove_dir(candidate_dir)
    _remove_dir(evaluated_dir)

    best = evaluated[0]
    logger.info(
        "Top %s stage %s model(s) saved to %s | best=%s marker=%s "
        "combined=%.2f eps=%s train=%.1f eval=%.1f avgD=%.1f minD=%.1f",
        len(evaluated),
        stage,
        top_dir,
        best.final_path,
        best.marker,
        best.combined_score,
        best.episodes,
        best.avg_train_score,
        best.avg_eval_score,
        best.avg_min_dist_score,
        best.min_dist_score,
    )

    return evaluated
