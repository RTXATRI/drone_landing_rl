# 功能：训练阶段候选模型采样，以及阶段结束后的并行复评排序。

from __future__ import annotations

import gc
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from configs.env_config import EnvConfig
from configs.train_config import TrainConfig
from curriculum.strategies import get_strategy_class

if TYPE_CHECKING:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecMonitor

logger = logging.getLogger(__name__)

FAILURE_TERMINATIONS = {"oob", "below_ground", "crashed"}


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
    avg_train_score: float
    avg_eval_score: float
    combined_score: float
    success_rate: float
    failure_count: int
    avg_reward: float
    avg_length: float
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
                logger.info(
                    "Best-model grid candidate saved: stage=%s interval=%s ts=%s",
                    self._stage,
                    self._next_grid_interval,
                    self.num_timesteps,
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
        -int(candidate.timestep),
    )


class BestModelEvalProgress:
    """Real-time progress display for per-stage best-model reevaluation."""

    def __init__(
        self,
        *,
        stage: int,
        candidates_total: int,
        episodes_per_candidate: int,
    ):
        self.stage = int(stage)
        self.candidates_total = max(0, int(candidates_total))
        self.episodes_per_candidate = max(1, int(episodes_per_candidate))
        self.total_episodes = self.candidates_total * self.episodes_per_candidate
        self.completed_episodes = 0

        self._bar = None
        self._tqdm_cls = None
        self._interactive = bool(sys.stderr.isatty())
        self._started_at = 0.0
        self._candidate_started_at = 0.0
        self._candidate_index = 0
        self._candidate: Optional[ModelCandidate] = None

        self._current_count = 0
        self._current_train_total = 0.0
        self._current_eval_total = 0.0
        self._current_success_total = 0.0
        self._current_failure_count = 0

        self._best_key: Optional[tuple] = None
        self._best_label = "none"

    def __enter__(self) -> "BestModelEvalProgress":
        self._started_at = time.monotonic()
        if self._interactive:
            try:
                from tqdm import tqdm

                self._tqdm_cls = tqdm
                self._bar = tqdm(
                    total=self.total_episodes,
                    desc=f"Stage {self.stage} best eval",
                    unit="eps",
                    dynamic_ncols=True,
                    mininterval=0.5,
                )
            except Exception:
                self._interactive = False
                self._bar = None

        if not self._interactive:
            logger.info(
                "Best-model eval progress: stage=%s candidates=%s "
                "episodes_per_candidate=%s total_episodes=%s",
                self.stage,
                self.candidates_total,
                self.episodes_per_candidate,
                self.total_episodes,
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def on_candidate_start(self, index: int, candidate: ModelCandidate) -> None:
        self._candidate_index = int(index)
        self._candidate = candidate
        self._candidate_started_at = time.monotonic()
        self._current_count = 0
        self._current_train_total = 0.0
        self._current_eval_total = 0.0
        self._current_success_total = 0.0
        self._current_failure_count = 0
        self._refresh_bar()

    def on_episode(self, result: dict) -> None:
        self.completed_episodes += 1
        self._current_count += 1
        self._current_train_total += float(result.get("train_score", 0.0))
        self._current_eval_total += float(result.get("eval_score", 0.0))
        self._current_success_total += float(result.get("success", False))
        if bool(result.get("failure", False)):
            self._current_failure_count += 1

        if self._bar is not None:
            self._bar.update(1)
            self._refresh_bar()

    def on_candidate_done(
        self,
        *,
        summary: dict,
        combined: float,
        marker: str,
    ) -> None:
        candidate = self._candidate
        if candidate is not None:
            key = _evaluated_sort_key(
                marker=marker,
                summary=summary,
                combined=combined,
                candidate=candidate,
            )
            if self._best_key is None or key < self._best_key:
                self._best_key = key
                self._best_label = f"{marker}:{float(combined):.2f}"

        elapsed = max(0.0, time.monotonic() - self._candidate_started_at)
        message = (
            f"[{self._candidate_index}/{self.candidates_total}] {marker} "
            f"combined={float(combined):.2f} "
            f"train={float(summary['avg_train_score']):.2f} "
            f"eval={float(summary['avg_eval_score']):.2f} "
            f"sr={float(summary['success_rate']):.1%} "
            f"fail={int(summary['failure_count'])} "
            f"avg_len={float(summary['avg_length']):.0f} "
            f"time={elapsed:.1f}s"
        )
        if candidate is not None:
            message += (
                f" kind={candidate.kind} ts={candidate.timestep:010d}"
            )

        if self._interactive and self._tqdm_cls is not None:
            self._tqdm_cls.write(message)
            _log_file_only(logger, message)
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
            f"{self._candidate_index}/{self.candidates_total} candidates | "
            f"{self.completed_episodes}/{self.total_episodes} eps | "
            f"{rate:.2f} ep/s | ETA {eta}"
        )

    def _refresh_bar(self) -> None:
        if self._bar is None:
            return

        candidate = self._candidate
        if candidate is None:
            candidate_text = f"cand={self._candidate_index}/{self.candidates_total}"
        else:
            candidate_text = (
                f"cand={self._candidate_index}/{self.candidates_total} "
                f"kind={candidate.kind} ts={candidate.timestep:010d}"
            )

        if self._current_count > 0:
            avg_train = self._current_train_total / self._current_count
            avg_eval = self._current_eval_total / self._current_count
            sr = self._current_success_total / self._current_count
        else:
            avg_train = 0.0
            avg_eval = 0.0
            sr = 0.0

        postfix = (
            f"{candidate_text} "
            f"cand_ep={self._current_count}/{self.episodes_per_candidate} "
            f"best={self._best_label} "
            f"current=train={avg_train:.1f} eval={avg_eval:.1f} "
            f"sr={sr:.1%} fail={self._current_failure_count}"
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
        while np.any(active):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            for env_idx, info in enumerate(infos):
                if not active[env_idx] or not dones[env_idx]:
                    continue
                ep = info.get("episode")
                if ep is None:
                    continue
                episode_idx += 1
                termination = str(info.get("termination", ep.get("termination", "none")))
                result = {
                    "episode": episode_idx,
                    "reward": float(ep.get("r", 0.0)),
                    "length": int(ep.get("l", 0)),
                    "success": bool(ep.get("success", info.get("success", False))),
                    "train_score": float(ep.get("train_score", info.get("train_score", 0.0))),
                    "eval_score": float(ep.get("eval_score", info.get("eval_score", 0.0))),
                    "termination": termination,
                    "failure": termination in FAILURE_TERMINATIONS,
                }
                results.append(result)
                if progress is not None:
                    progress.on_episode(result)
                active[env_idx] = False

    if not results:
        return {
            "episodes": 0,
            "avg_reward": 0.0,
            "avg_length": 0.0,
            "success_rate": 0.0,
            "avg_train_score": 0.0,
            "avg_eval_score": 0.0,
            "failure_count": 0,
        }

    return {
        "episodes": len(results),
        "avg_reward": float(np.mean([r["reward"] for r in results])),
        "avg_length": float(np.mean([r["length"] for r in results])),
        "success_rate": float(np.mean([float(r["success"]) for r in results])),
        "avg_train_score": float(np.mean([r["train_score"] for r in results])),
        "avg_eval_score": float(np.mean([r["eval_score"] for r in results])),
        "failure_count": int(sum(1 for r in results if r["failure"])),
    }


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
    configured_envs = int(train_config.best_model_eval_envs_by_stage.get(stage, 1))
    n_envs = max(1, min(configured_envs, eval_episodes))
    model_root = os.path.join(train_config.model_dir, train_config.exp_name)
    top_dir = os.path.join(model_root, "top_models")
    candidate_dir = os.path.join(model_root, f"stage{stage}_candidates")
    evaluated_dir = os.path.join(model_root, f"stage{stage}_evaluated")
    keep_top_n = max(1, int(train_config.best_model_keep_top_n))

    logger.info(
        "Evaluating %s best-model candidates for stage %s: episodes=%s eval_envs=%s",
        len(valid_candidates),
        stage,
        eval_episodes,
        n_envs,
    )

    train_weight = float(train_config.best_model_train_score_weight)
    eval_weight = float(train_config.best_model_eval_score_weight)
    seed_base = int(train_config.seed) + int(stage) * 1_000_000
    raw_rows = []

    try:
        eval_env = _build_eval_env(
            env_config,
            stage=stage,
            n_envs=n_envs,
            seed=seed_base,
        )
    except NotImplementedError:
        logger.warning("Stage %s evaluation is not implemented; skipping best-model evaluation.", stage)
        return []

    try:
        with BestModelEvalProgress(
            stage=stage,
            candidates_total=len(valid_candidates),
            episodes_per_candidate=eval_episodes,
        ) as progress:
            for idx, candidate in enumerate(valid_candidates, start=1):
                progress.on_candidate_start(idx, candidate)
                try:
                    model = SAC.load(candidate.path, env=eval_env, device=device)
                    summary = _evaluate_loaded_model(
                        model,
                        eval_env,
                        n_episodes=eval_episodes,
                        seed=seed_base,
                        progress=progress,
                    )
                except NotImplementedError:
                    logger.warning(
                        "Stage %s evaluation is not implemented; skipping best-model evaluation.",
                        stage,
                    )
                    return []
                combined = (
                    summary["avg_train_score"] * train_weight
                    + summary["avg_eval_score"] * eval_weight
                )
                marker = "fail" if summary["failure_count"] > 0 else "clean"
                raw_rows.append({
                    "candidate": candidate,
                    "summary": summary,
                    "combined": combined,
                    "marker": marker,
                    "original_index": idx,
                })
                progress.on_candidate_done(
                    summary=summary,
                    combined=combined,
                    marker=marker,
                )
                del model
                gc.collect()
                import torch

                if str(device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        eval_env.close()

    if not raw_rows:
        return []

    def _sort_key(row: dict):
        return _evaluated_sort_key(
            marker=row["marker"],
            summary=row["summary"],
            combined=row["combined"],
            candidate=row["candidate"],
        )

    raw_rows.sort(key=_sort_key)
    evaluated: List[EvaluatedModel] = []
    os.makedirs(top_dir, exist_ok=True)
    _remove_temp_top_models(top_dir, stage)
    copied_top_models = []

    for rank, row in enumerate(raw_rows[:keep_top_n], start=1):
        candidate = row["candidate"]
        summary = row["summary"]
        marker = row["marker"]
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
            avg_train_score=float(summary["avg_train_score"]),
            avg_eval_score=float(summary["avg_eval_score"]),
            combined_score=float(row["combined"]),
            success_rate=float(summary["success_rate"]),
            failure_count=int(summary["failure_count"]),
            avg_reward=float(summary["avg_reward"]),
            avg_length=float(summary["avg_length"]),
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
        "Top %s stage %s model(s) saved to %s | best=%s marker=%s combined=%.2f",
        len(evaluated),
        stage,
        top_dir,
        best.final_path,
        best.marker,
        best.combined_score,
    )

    return evaluated
