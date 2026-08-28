from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pi05_jax_sft.project_config import load_config
from pi05_jax_sft.runtime import build_train_config
from pi05_jax_sft.runtime import configure_lerobot_home
from pi05_jax_sft.runtime import ensure_openpi_importable
from pi05_jax_sft.runtime import latest_checkpoint_step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline inference on a trained checkpoint.")
    p.add_argument("--config", required=True, help="YAML config path.")
    p.add_argument("--checkpoint-step", type=int, default=None, help="Step to load (default: latest).")
    p.add_argument("--sample-index", type=int, default=0, help="Dataset index for inference.")
    p.add_argument("--output", default=None, help="Output .npz path (optional).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    configure_lerobot_home(cfg.resolve_path(cfg.data.hf_lerobot_home))
    ensure_openpi_importable(cfg.project_root)

    from openpi.policies import policy_config
    from openpi.training import data_loader as data_loader_lib

    train_cfg = build_train_config(cfg)
    checkpoint_root = train_cfg.checkpoint_dir
    step = args.checkpoint_step if args.checkpoint_step is not None else latest_checkpoint_step(checkpoint_root)
    checkpoint_dir = checkpoint_root / str(step)

    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    dataset = data_loader_lib.create_torch_dataset(data_cfg, train_cfg.model.action_horizon, train_cfg.model)
    sample = dataset[args.sample_index]

    policy = policy_config.create_trained_policy(
        train_cfg, checkpoint_dir, repack_transforms=data_cfg.repack_transforms
    )
    prediction = policy.infer(sample)
    predicted_actions = np.asarray(prediction["actions"])

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = (
            cfg.resolve_path(cfg.paths.eval_output_dir)
            / f"{cfg.project.name}_step{step}_sample{args.sample_index}.npz"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, predicted_actions=predicted_actions)

    print(json.dumps({
        "checkpoint_dir": str(checkpoint_dir),
        "sample_index": args.sample_index,
        "predicted_actions_shape": list(predicted_actions.shape),
        "output_path": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
