from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

from pi05_jax_sft.project_config import PipelineConfig


def configure_lerobot_home(home: Path) -> None:
    home = home.expanduser().resolve()
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_LEROBOT_HOME"] = str(home)
    # New lerobot raises ValueError if LEROBOT_HOME is set (deprecated).
    # Remove it unconditionally so any pre-existing shell value doesn't break the import.
    os.environ.pop("LEROBOT_HOME", None)
    os.environ.setdefault("HF_HOME", str(home.parent))


def ensure_openpi_importable(project_root: Path) -> None:
    for root in [
        project_root / "external" / "openpi" / "src",
        project_root / "external" / "openpi" / "packages" / "openpi-client" / "src",
    ]:
        if root.exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)


def load_openpi_script(project_root: Path, script_name: str) -> ModuleType:
    script_path = project_root / "external" / "openpi" / "scripts" / f"{script_name}.py"
    if not script_path.exists():
        raise FileNotFoundError(f"openpi script not found: {script_path}")
    spec = importlib.util.spec_from_file_location(f"openpi_script_{script_name}", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_train_config(cfg: PipelineConfig):
    ensure_openpi_importable(cfg.project_root)
    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.training import config as train_config
    from openpi.training import optimizer
    from openpi.training import weight_loaders
    from pi05_jax_sft.company_policy import CompanyWristInputs, CompanyWristOutputs

    model_cfg = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=cfg.model.action_horizon,
        max_token_len=cfg.model.max_token_len,
        discrete_state_input=cfg.model.discrete_state_input,
        paligemma_variant=cfg.model.paligemma_variant,
        action_expert_variant=cfg.model.action_expert_variant,
        rtc_simulated_delay=cfg.model.rtc_simulated_delay,
    )

    action_semantics = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper_cmd"]
    if cfg.model.policy_action_dim >= 8:
        action_semantics.append("task_status")

    data_transforms = transforms.Group(
        inputs=[CompanyWristInputs()],
        outputs=[CompanyWristOutputs(action_dim=cfg.model.policy_action_dim)],
    )

    return train_config.TrainConfig(
        name=cfg.project.name,
        project_name=cfg.project.project_name,
        exp_name=cfg.project.exp_name,
        model=model_cfg,
        data=train_config.SimpleDataConfig(
            repo_id=cfg.data.repo_id,
            base_config=train_config.DataConfig(prompt_from_task=cfg.data.prompt_from_task),
            data_transforms=lambda model: data_transforms,
            model_transforms=train_config.ModelTransformFactory(
                default_prompt=None if cfg.data.prompt_from_task else cfg.data.task_name
            ),
        ),
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        num_train_steps=cfg.training.num_train_steps,
        log_interval=cfg.training.log_interval,
        save_interval=cfg.training.save_interval,
        keep_period=cfg.training.keep_period,
        seed=cfg.training.seed,
        wandb_enabled=cfg.training.wandb_enabled,
        overwrite=cfg.training.overwrite,
        resume=cfg.training.resume,
        fsdp_devices=cfg.training.fsdp_devices,
        ema_decay=cfg.training.ema_decay,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=cfg.training.warmup_steps,
            peak_lr=cfg.training.peak_lr,
            decay_steps=cfg.training.decay_steps,
            decay_lr=cfg.training.decay_lr,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=cfg.training.clip_gradient_norm),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            cfg.resolve_maybe_remote(cfg.paths.base_checkpoint_path)
        ),
        assets_base_dir=str(cfg.resolve_path(cfg.paths.assets_base_dir)),
        checkpoint_base_dir=str(cfg.resolve_path(cfg.paths.checkpoint_base_dir)),
        policy_metadata={
            "camera_mapping": {
                "base_0_rgb": "chest_image",
                "left_wrist_0_rgb": "wrist_image",
                "right_wrist_0_rgb": "top_image",
            },
            "state_semantics": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
            "action_semantics": action_semantics,
            "action_rotation_convention": "xyz_extrinsic_euler",
            "normalize_rx_to_2pi": cfg.data.normalize_rx_to_2pi,
        },
    )


def latest_checkpoint_step(checkpoint_root: Path) -> int:
    steps = [int(p.name) for p in checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not steps:
        raise FileNotFoundError(f"No numeric checkpoint steps in {checkpoint_root}")
    return max(steps)

