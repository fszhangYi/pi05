"""Launch Tonglu-style pi0.5 **PyTorch** full fine-tuning (openpi train_pytorch).

Matches hww artifacts/checkpoints/*/mlu_full_ft* (model.safetensors + metadata.pt).
Requires multi-GPU (config fsdp_devices, typically 8) and pi05_base_pytorch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pi05_jax_sft.project_config import load_config
from pi05_jax_sft.runtime import build_train_config
from pi05_jax_sft.runtime import configure_lerobot_home
from pi05_jax_sft.runtime import count_visible_gpus
from pi05_jax_sft.runtime import ensure_openpi_importable
from pi05_jax_sft.runtime import load_openpi_script
from pi05_jax_sft.runtime import resolve_base_checkpoint


def _preflight(cfg, train_cfg) -> None:
    """Hard-fail before torchrun full-FT; do not silently downscale."""
    base = resolve_base_checkpoint(cfg)
    weights = base / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(
            f"PyTorch base weights not found: {weights}\n"
            "Place openpi pi0.5 PyTorch base under paths.base_checkpoint_path "
            "(expected .../pi05_base_pytorch/model.safetensors).\n"
            "See docs/tonglu_mlu_full_ft_reproduce.md"
        )

    if not train_cfg.pytorch_weight_path:
        raise RuntimeError(
            f"TrainConfig.pytorch_weight_path is unset for base={base}. "
            "Use a PyTorch base dir (name pi05_base_pytorch or containing model.safetensors)."
        )

    norm_stats = Path(train_cfg.assets_dirs) / cfg.data.repo_id / "norm_stats.json"
    if not norm_stats.is_file():
        raise FileNotFoundError(
            f"Norm stats not found: {norm_stats}\n"
            f"Run: bash scripts/compute_norm_stats.sh {cfg.config_path}"
        )

    dataset_root = cfg.resolve_path(cfg.data.hf_lerobot_home) / cfg.data.repo_id
    if not any(dataset_root.glob("data/**/*.parquet")):
        raise FileNotFoundError(
            f"No LeRobot parquet files under {dataset_root}\n"
            f"Run: bash scripts/prepare_dataset.sh {cfg.config_path}"
        )

    nproc = int(os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("WORLD_SIZE") or "0")
    if nproc <= 0:
        nproc = count_visible_gpus()
    need = int(cfg.training.fsdp_devices)
    if nproc < need:
        raise RuntimeError(
            f"Tonglu mlu_full_ft* full FT expects ≥{need} GPUs (config training.fsdp_devices), "
            f"but only {nproc} visible.\n"
            "This recipe does not downscale batch/cards. Use an 8×~80GB machine, "
            "or switch to the LoRA smoke path (scripts/train_8gpu.sh + smoke YAML)."
        )

    if train_cfg.batch_size % need != 0:
        raise ValueError(
            f"batch_size {train_cfg.batch_size} must be divisible by fsdp_devices/nproc {need}"
        )

    variants = (cfg.model.paligemma_variant, cfg.model.action_expert_variant)
    if any(v.endswith("_lora") for v in variants):
        raise RuntimeError(
            f"train_pytorch full-FT entry rejects LoRA variants {variants}. "
            "Use gemma_2b + gemma_300m (see configs/pi05_tonglu*_full_ft_*.yaml)."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch pi0.5 PyTorch full fine-tuning (Tonglu mlu_full_ft*).")
    p.add_argument("--config", required=True, help="YAML config path.")
    p.add_argument("--print-only", action="store_true", help="Print resolved summary and exit.")
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip checks (debug only; not for reproduce).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    configure_lerobot_home(cfg.resolve_path(cfg.data.hf_lerobot_home))
    ensure_openpi_importable(cfg.project_root)

    train_cfg = build_train_config(cfg)
    if not args.print_only and not args.skip_preflight:
        _preflight(cfg, train_cfg)

    summary = {
        "backend": "pytorch",
        "config_path": str(cfg.config_path),
        "assets_dir": str(train_cfg.assets_dirs),
        "checkpoint_dir": str(train_cfg.checkpoint_dir),
        "repo_id": cfg.data.repo_id,
        "pytorch_weight_path": train_cfg.pytorch_weight_path,
        "pytorch_training_precision": train_cfg.pytorch_training_precision,
        "paligemma_variant": cfg.model.paligemma_variant,
        "action_expert_variant": cfg.model.action_expert_variant,
        "fsdp_devices": train_cfg.fsdp_devices,
        "batch_size": train_cfg.batch_size,
        "num_train_steps": train_cfg.num_train_steps,
        "discrete_state_input": cfg.model.discrete_state_input,
        "action_horizon": cfg.model.action_horizon,
        "visible_gpus": count_visible_gpus(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.print_only:
        return

    train_module = load_openpi_script(cfg.project_root, "train_pytorch")
    train_module.train_loop(train_cfg)


if __name__ == "__main__":
    main()
