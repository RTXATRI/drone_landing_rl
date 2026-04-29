#!/usr/bin/env python3
# 功能：将 Stable-Baselines3 SAC actor 导出为 TorchScript、ONNX 或 NumPy 权重。
"""
面向部署的模型导出工具。

将训练好的 SB3 SAC 策略转换为：
  - TorchScript (.pt) — 用于 C++/移动端部署
  - ONNX (.onnx)      — 用于 TensorRT / 跨平台推理
  - NumPy weights     — 用于手动重实现（例如嵌入式系统）

导出的模型接收 34 维观测，并输出 [-1, 1] 中的 4 维动作。

用法：
    # 导出所有格式
    python scripts/export_model.py --model models/drone_landing/model_final

    # 只导出 TorchScript
    python scripts/export_model.py --model models/drone_landing/model_final --format torchscript

    # 对比 SB3 输出，验证导出模型
    python scripts/export_model.py --model models/drone_landing/model_final --verify
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  type=str, required=True)
    p.add_argument("--format", type=str, default="all",
                   choices=["all", "torchscript", "onnx", "numpy"])
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--verify",  action="store_true")
    return p.parse_args()


class DeterministicActorWrapper(torch.nn.Module):
    """
    包装 SB3 SAC actor 以便导出。
    输入：  observation tensor (B, 34)
    输出：action tensor in [-1, 1] (B, 4)  ← deterministic (mean)
    """

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs, deterministic=True)


def export_torchscript(actor_wrapper: torch.nn.Module, out_path: str) -> None:
    dummy = torch.zeros(1, 34, dtype=torch.float32)
    ts    = torch.jit.trace(actor_wrapper, dummy)
    torch.jit.save(ts, out_path)
    print(f"  TorchScript → {out_path}")


def export_onnx(actor_wrapper: torch.nn.Module, out_path: str) -> None:
    dummy = torch.zeros(1, 34, dtype=torch.float32)
    torch.onnx.export(
        actor_wrapper,
        dummy,
        out_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch_size"},
                      "action":      {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  ONNX       → {out_path}")


def export_numpy(actor_wrapper: torch.nn.Module, out_path: str) -> None:
    """将权重矩阵保存为 npz，便于手动重实现。"""
    weights = {}
    for name, param in actor_wrapper.named_parameters():
        key = name.replace(".", "_")
        weights[key] = param.detach().cpu().numpy()
    np.savez(out_path, **weights)
    print(f"  NumPy npz  → {out_path}.npz")
    print("  Keys:", list(weights.keys()))


def verify(actor_wrapper: torch.nn.Module, torchscript_path: str = None,
           onnx_path: str = None) -> None:
    print("\n  Verifying outputs…")
    obs_np = np.random.randn(5, 34).astype(np.float32)
    obs_t  = torch.from_numpy(obs_np)

    with torch.no_grad():
        ref = actor_wrapper(obs_t).numpy()

    if torchscript_path and os.path.exists(torchscript_path):
        ts  = torch.jit.load(torchscript_path)
        out = ts(obs_t).numpy()
        err = np.abs(ref - out).max()
        ok  = "✓" if err < 1e-5 else "✗"
        print(f"  {ok} TorchScript max error: {err:.2e}")

    if onnx_path and os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path)
            out  = sess.run(["action"], {"observation": obs_np})[0]
            err  = np.abs(ref - out).max()
            ok   = "✓" if err < 1e-4 else "✗"
            print(f"  {ok} ONNX       max error: {err:.2e}")
        except ImportError:
            print("  ℹ  onnxruntime not installed — skipping ONNX verification.")


def main():
    args = parse_args()

    from stable_baselines3 import SAC
    print(f"Loading model: {args.model}")
    model = SAC.load(args.model, device="cpu")

    actor_wrapper = DeterministicActorWrapper(model.policy.actor)
    actor_wrapper.eval()

    out_dir  = args.out_dir or os.path.dirname(args.model)
    base     = os.path.basename(args.model)
    ts_path  = os.path.join(out_dir, f"{base}_actor.pt")
    ox_path  = os.path.join(out_dir, f"{base}_actor.onnx")
    np_path  = os.path.join(out_dir, f"{base}_weights")

    print(f"Exporting to: {out_dir}")
    with torch.no_grad():
        if args.format in ("all", "torchscript"):
            export_torchscript(actor_wrapper, ts_path)
        if args.format in ("all", "onnx"):
            try:
                export_onnx(actor_wrapper, ox_path)
            except Exception as e:
                print(f"  ONNX export failed: {e}")
        if args.format in ("all", "numpy"):
            export_numpy(actor_wrapper, np_path)

    if args.verify:
        verify(actor_wrapper,
               ts_path if args.format in ("all", "torchscript") else None,
               ox_path  if args.format in ("all", "onnx") else None)

    print("\nExport complete.")


if __name__ == "__main__":
    main()
