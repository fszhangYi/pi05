#!/usr/bin/env python
"""Offline benchmark: replay logged frames through Pi05Inference and measure latency."""

import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image

# Ensure src/ is importable
_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from pi05_jax_sft.serve import Pi05Inference, build_train_config, configure_lerobot_home


def load_step(step_dir: pathlib.Path):
    """Load a single logged step."""
    with open(step_dir / "robot_state.json") as f:
        robot_state = np.array(json.load(f), dtype=np.float32)
    images = {}
    for cam in ("top", "chest", "wrist2"):
        img_path = step_dir / f"{cam}.jpg"
        if img_path.exists():
            images[cam] = img_path.read_bytes()
        else:
            raise FileNotFoundError(f"Missing {img_path}")
    return images, robot_state


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True, type=pathlib.Path, help="Path to a single run dir, e.g. logs/20260615/114825")
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint-dir", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--task-prompt", type=str, default=None)
    parser.add_argument("--lerobot-home", type=str, default="/tmp/lerobot")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    step_dirs = sorted([d for d in args.log_dir.iterdir() if d.is_dir() and d.name.startswith("step_")])
    if not step_dirs:
        print(f"No step directories found in {args.log_dir}")
        sys.exit(1)

    print(f"Found {len(step_dirs)} steps in {args.log_dir}")

    configure_lerobot_home(args.lerobot_home)
    import lerobot.common.datasets.utils as lerobot_utils
    lerobot_utils.init_hf_datasets_cache()
    import lerobot.common.datasets.lerobot_dataset

    train_config, data_config, created_data_config = build_train_config(str(args.config))

    ckpt_dir = args.checkpoint_dir
    if args.checkpoint_step is not None:
        ckpt_dir = ckpt_dir / str(args.checkpoint_step)
    ckpt_dir = ckpt_dir.expanduser().resolve()
    print(f"Loading checkpoint: {ckpt_dir}")

    inferencer = Pi05Inference(
        config=train_config,
        data_config=data_config,
        checkpoint_dir=ckpt_dir,
        task_prompt=args.task_prompt,
        device=args.device,
    )
    print(f"Warm-up done. Ready to benchmark.")

    latencies_ms = []
    for step_dir in step_dirs:
        images, robot_state = load_step(step_dir)
        image_data = {
            "top_image": images["top"],
            "chest_image": images["chest"],
            "wrist_image": images["wrist2"],
        }
        t0 = time.monotonic()
        chunk = inferencer.infer_chunk(image_data, robot_state, task_prompt=args.task_prompt)
        t1 = time.monotonic()
        lat = (t1 - t0) * 1000
        latencies_ms.append(lat)
        print(f"  {step_dir.name}: {lat:.2f} ms  (chunk shape {chunk.shape})")

    arr = np.array(latencies_ms)
    print("\n========== Benchmark Results ==========")
    print(f"  Frames    : {len(arr)}")
    print(f"  Total     : {arr.sum():.1f} ms")
    print(f"  Mean      : {arr.mean():.2f} ms")
    print(f"  Median    : {np.median(arr):.2f} ms")
    print(f"  Std       : {arr.std():.2f} ms")
    print(f"  Min       : {arr.min():.2f} ms")
    print(f"  Max       : {arr.max():.2f} ms")
    print(f"  P99       : {np.percentile(arr, 99):.2f} ms")
    print(f"  P95       : {np.percentile(arr, 95):.2f} ms")
    print("=======================================")


if __name__ == "__main__":
    main()
