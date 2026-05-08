#!/usr/bin/env python3
# 功能：使用 PyBullet rgb_array 渲染评估回合并导出 MP4 视频。
"""
将评估 rollout 录制为 MP4。

采集 PyBullet rgb_array 帧，并使用 OpenCV 或 imageio 编码。

用法：
    python scripts/record_video.py \
        --model output/models/drone_landing/model_final \
        --stage 4 \
        --episodes 3 \
        --out eval_stage4.mp4
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs.env_config import EnvConfig
from curriculum.strategies import create_strategy, registered_stage_ids
from envs.drone_landing_env import DroneLandingEnv
from stable_baselines3 import SAC


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    type=str, required=True)
    p.add_argument("--stage",    type=int, default=4, choices=list(registered_stage_ids()))
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--out",      type=str, default="eval_video.mp4")
    p.add_argument("--fps",      type=int, default=20)
    p.add_argument("--seed",     type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import imageio
        writer_lib = "imageio"
    except ImportError:
        try:
            import cv2
            writer_lib = "cv2"
        except ImportError:
            print("ERROR: Install imageio or opencv-python to record video.")
            print("  pip install imageio[ffmpeg]   # recommended")
            sys.exit(1)

    env_config = EnvConfig()
    model      = SAC.load(args.model, device="cpu")
    env        = DroneLandingEnv(
        env_config,
        strategy=create_strategy(args.stage, env_config),
        render_mode="rgb_array",
    )

    frames = []
    print(f"Recording {args.episodes} episodes (stage {args.stage})…")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done   = False
        ep_len = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            done = terminated or truncated
            ep_len += 1

        success = info.get("episode", {}).get("success", False)
        print(f"  Ep {ep+1}: {'SUCCESS' if success else 'fail'}, {ep_len} steps, "
              f"{len(frames)} frames so far")

    env.close()

    if not frames:
        print("No frames captured — aborting.")
        return

    print(f"\nEncoding {len(frames)} frames @ {args.fps} fps → {args.out}")

    if writer_lib == "imageio":
        import imageio
        imageio.mimwrite(args.out, frames, fps=args.fps, quality=8)
    else:
        import cv2
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
        )
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()

    print(f"Video saved → {args.out}  ({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
