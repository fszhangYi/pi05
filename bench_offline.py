#!/usr/bin/env python3
"""Offline inference benchmark using logged frames from serve.py."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/hww/pi05_jax_sft/src')
sys.path.insert(0, '/home/ubuntu/hww/pi05_jax_sft/external/openpi/src')

import numpy as np

from pi05_jax_sft.serve import Pi05Inference


def main() -> None:
    log_root = Path('/home/ubuntu/hww/pi05_jax_sft/logs/20260611')
    # Use the latest run (by dir name)
    runs = sorted(log_root.iterdir())
    if not runs:
        print("No log runs found.")
        sys.exit(1)

    log_dir = runs[-1]
    print(f"Using log run: {log_dir.name}")

    step_dirs = sorted(log_dir.glob('step_*'))
    print(f"Total frames: {len(step_dirs)}")

    inferencer = Pi05Inference(
        config_path='/home/ubuntu/hww/pi05_jax_sft/configs/pi05_tonglu0602_example.yaml',
        checkpoint_step=24000,
        checkpoint_dir='/home/ubuntu/hww/pi05_jax_sft/artifacts/checkpoints/pi05_tonglu0602_lora/8gpu_lora',
        num_steps_override=None,
        task_prompt=None,
        lerobot_home=None,
    )

    times_ms = []
    for step_dir in step_dirs:
        # Load robot state
        robot_state = json.loads((step_dir / 'robot_state.json').read_text())
        robot_state = np.array(robot_state, dtype=np.float32)

        # Load images as bytes (same as TCP receive)
        image_bytes = {
            'top_image': (step_dir / 'top.jpg').read_bytes(),
            'chest_image': (step_dir / 'chest.jpg').read_bytes(),
            'wrist_image': (step_dir / 'wrist2.jpg').read_bytes(),
        }

        t0 = time.monotonic()
        chunk = inferencer.infer_chunk(image_bytes, robot_state)
        t1 = time.monotonic()
        dt = (t1 - t0) * 1000.0
        times_ms.append(dt)
        print(f"{step_dir.name}: {dt:.1f} ms  (chunk shape {chunk.shape})")

    arr = np.array(times_ms)
    print(f"\n{'='*50}")
    print(f"Frames processed : {len(arr)}")
    print(f"Mean             : {arr.mean():.2f} ms")
    print(f"Median           : {np.median(arr):.2f} ms")
    print(f"Std dev          : {arr.std():.2f} ms")
    print(f"Min              : {arr.min():.2f} ms")
    print(f"Max              : {arr.max():.2f} ms")
    print(f"P99              : {np.percentile(arr, 99):.2f} ms")
    print(f"P95              : {np.percentile(arr, 95):.2f} ms")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
