# 功能：训练阶段候选模型采样，以及阶段结束后的并行复评排序。

from __future__ import annotations

import gc
import logging
import math
import os
import shutil
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
                results.append({
                    "episode": episode_idx,
                    "reward": float(ep.get("r", 0.0)),
                    "length": int(ep.get("l", 0)),
                    "success": bool(ep.get("success", info.get("success", False))),
                    "train_score": float(ep.get("train_score", info.get("train_score", 0.0))),
                    "eval_score": float(ep.get("eval_score", info.get("eval_score", 0.0))),
                    "termination": termination,
                    "failure": termination in FAILURE_TERMINATIONS,
                })
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
        len(candidates),
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
        for idx, candidate in enumerate(candidates, start=1):
            if not os.path.exists(candidate.path + ".zip"):
                logger.warning("Candidate zip missing; skipping: %s.zip", candidate.path)
                continue
            try:
                model = SAC.load(candidate.path, env=eval_env, device=device)
                summary = _evaluate_loaded_model(
                    model,
                    eval_env,
                    n_episodes=eval_episodes,
                    seed=seed_base,
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
            raw_rows.append({
                "candidate": candidate,
                "summary": summary,
                "combined": combined,
                "marker": "fail" if summary["failure_count"] > 0 else "clean",
                "original_index": idx,
            })
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
        marker_rank = 0 if row["marker"] == "clean" else 1
        return (
            marker_rank,
            row["summary"]["failure_count"],
            -row["combined"],
            -row["summary"]["avg_eval_score"],
            -row["summary"]["avg_train_score"],
            -row["candidate"].timestep,
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
