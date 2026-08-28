"""Compute state/action normalization statistics for pi0.5 training.

Reads state and actions directly from the LeRobot parquet files, skipping
image decoding entirely. This is 10-50x faster than going through the
full TransformedDataset pipeline.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from pi05_jax_sft.project_config import load_config
from pi05_jax_sft.runtime import configure_lerobot_home
from pi05_jax_sft.runtime import ensure_openpi_importable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="YAML config path.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap frame count for quick debugging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    configure_lerobot_home(cfg.resolve_path(cfg.data.hf_lerobot_home))
    ensure_openpi_importable(cfg.project_root)

    from openpi.shared import normalize

    dataset_root = Path(os.environ["HF_LEROBOT_HOME"]) / cfg.data.repo_id
    parquet_files = sorted(dataset_root.glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under {dataset_root}.\n"
            "Run prepare_dataset.sh first to convert raw data to LeRobot format."
        )

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    total_frames = 0

    for pf in tqdm.tqdm(parquet_files, desc="norm stats", unit="file"):
        # Read only the two numeric columns — images are stored as paths and are skipped.
        df = pd.read_parquet(pf, columns=["state", "actions"])

        states = np.stack(df["state"].tolist()).astype(np.float32)   # (N, 7)
        actions = np.stack(df["actions"].tolist()).astype(np.float32) # (N, 7)

        if args.max_frames is not None:
            remaining = args.max_frames - total_frames
            if remaining <= 0:
                break
            states = states[:remaining]
            actions = actions[:remaining]

        state_stats.update(states)
        action_stats.update(actions)
        total_frames += len(states)

        if args.max_frames is not None and total_frames >= args.max_frames:
            break

    print(f"Processed {total_frames} frames from {len(parquet_files)} parquet file(s).")

    # assets_dirs = (assets_base_dir / project.name).resolve()  — same formula as TrainConfig.
    output_dir = cfg.resolve_path(cfg.paths.assets_base_dir) / cfg.project.name / cfg.data.repo_id
    output_dir.mkdir(parents=True, exist_ok=True)

    normalize.save(output_dir, {
        "state":   state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    })
    print(f"Norm stats written to {output_dir}/norm_stats.json")


if __name__ == "__main__":
    main()
