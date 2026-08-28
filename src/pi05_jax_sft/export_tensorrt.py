#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pi0.5 PyTorch sample_actions to a Torch-TensorRT engine.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint root containing numeric step dirs. Defaults to config checkpoint_dir.",
    )
    parser.add_argument("--checkpoint-step", type=int, default=None, help="Step to load. Defaults to latest.")
    parser.add_argument("--output", required=True, help="Output .ep path for torch.export/Torch-TensorRT.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-prompt", default=None)
    parser.add_argument("--lerobot-home", default=None)
    parser.add_argument("--num-steps", type=int, default=10, help="Flow sampling steps used by sample_actions.")
    parser.add_argument("--fp32", action="store_true", help="Compile in FP32 instead of FP16.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from pi05_jax_sft.trt_policy import build_example_inputs, compile_sample_actions_to_tensorrt

    from pi05_jax_sft.project_config import load_config
    from pi05_jax_sft.runtime import (
        build_train_config,
        configure_lerobot_home,
        ensure_openpi_importable,
        latest_checkpoint_step,
    )

    cfg = load_config(args.config)
    lerobot_home = Path(args.lerobot_home) if args.lerobot_home else cfg.resolve_path(cfg.data.hf_lerobot_home)
    try:
        configure_lerobot_home(lerobot_home)
    except PermissionError:
        configure_lerobot_home(Path(tempfile.gettempdir()) / "lerobot")
    ensure_openpi_importable(cfg.project_root)

    from openpi.policies import policy_config as pc

    train_cfg = build_train_config(cfg)
    object.__setattr__(train_cfg.model, "pytorch_compile_mode", None)

    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else train_cfg.checkpoint_dir
    step = args.checkpoint_step if args.checkpoint_step is not None else latest_checkpoint_step(checkpoint_root)
    checkpoint_path = checkpoint_root / str(step)
    prompt = args.task_prompt or cfg.data.task_name

    print(f"Loading checkpoint: {checkpoint_path}")
    policy = pc.create_trained_policy(
        train_cfg,
        checkpoint_path,
        default_prompt=prompt,
        pytorch_device=args.device,
    )

    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_obs = {
        "top_image": dummy_img,
        "chest_image": dummy_img,
        "wrist_image": dummy_img,
        "state": np.zeros(7, dtype=np.float32),
        "prompt": prompt,
    }

    print("Building example TensorRT inputs from OpenPI transforms...")
    _, example_inputs = build_example_inputs(policy, dummy_obs, args.device)
    print("Example input shapes:")
    for idx, tensor in enumerate(example_inputs):
        print(f"  input[{idx}]: shape={tuple(tensor.shape)} dtype={tensor.dtype}")

    print(f"Compiling TensorRT engine -> {args.output}")
    compile_sample_actions_to_tensorrt(
        policy,
        example_inputs,
        device=args.device,
        num_steps=args.num_steps,
        engine_path=args.output,
        fp16=not args.fp32,
    )
    print("Done.")


if __name__ == "__main__":
    main()
