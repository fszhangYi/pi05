from __future__ import annotations

import argparse
import json

from pi05_jax_sft.project_config import load_config
from pi05_jax_sft.runtime import build_train_config
from pi05_jax_sft.runtime import configure_lerobot_home
from pi05_jax_sft.runtime import ensure_openpi_importable
from pi05_jax_sft.runtime import load_openpi_script


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch pi0.5 JAX fine-tuning.")
    p.add_argument("--config", required=True, help="YAML config path.")
    p.add_argument("--print-only", action="store_true", help="Print resolved TrainConfig and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    configure_lerobot_home(cfg.resolve_path(cfg.data.hf_lerobot_home))
    ensure_openpi_importable(cfg.project_root)

    train_cfg = build_train_config(cfg)
    summary = {
        "config_path": str(cfg.config_path),
        "assets_dir": str(train_cfg.assets_dirs),
        "checkpoint_dir": str(train_cfg.checkpoint_dir),
        "repo_id": cfg.data.repo_id,
        "weight_loader": cfg.resolve_maybe_remote(cfg.paths.base_checkpoint_path),
        "fsdp_devices": train_cfg.fsdp_devices,
        "batch_size": train_cfg.batch_size,
        "num_train_steps": train_cfg.num_train_steps,
        "discrete_state_input": cfg.model.discrete_state_input,
        "action_horizon": cfg.model.action_horizon,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.print_only:
        return

    train_module = load_openpi_script(cfg.project_root, "train")
    train_module.main(train_cfg)


if __name__ == "__main__":
    main()
