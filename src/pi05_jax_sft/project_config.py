from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import yaml


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectSettings:
    name: str = "pi05_company_jax"
    exp_name: str = "full_finetune"
    project_name: str = "openpi"


@dataclass(frozen=True)
class DataSettings:
    repo_id: str = "company/pi05_company_task"
    raw_root: str = "/home/znyyb/hww/vla/cam_100_15"
    dataset_format: str = "company_steps"
    annotation_root: str | None = None
    hf_lerobot_home: str = "./data/lerobot"
    robot_type: str = "custom"
    fps: int = 15
    wrist_image_prefix: str = "rgb_wrist_1"
    chest_image_prefix: str | None = None
    top_image_prefix: str | None = None
    task_name: str = "抓取镜头模组"
    task_prompt_template: str | None = None
    prompt_from_task: bool = False
    normalize_rx_to_2pi: bool = False
    gripper_action_source: str = "actions"
    max_episodes: int | None = None


@dataclass(frozen=True)
class ModelSettings:
    action_horizon: int = 10
    # Number of action dims exposed by the policy output transform. Keep 7 for
    # legacy checkpoints; set 8 when dim 7 carries task_status/progress.
    policy_action_dim: int = 7
    max_token_len: int = 200
    # Must be true for pi0.5: state is discretised into the language prompt prefix.
    discrete_state_input: bool = True
    # ---- variant selection ----
    # Full fine-tune  : paligemma_variant="gemma_2b",      action_expert_variant="gemma_300m"
    # LoRA (backbone) : paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m"
    # LoRA (both)     : paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    # Training-Time RTC: null/0 disables it; N samples simulated prefix delays from [0, N).
    rtc_simulated_delay: int | None = None


@dataclass(frozen=True)
class TrainingSettings:
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 8
    num_train_steps: int = 20_000
    log_interval: int = 100
    save_interval: int = 2_000
    keep_period: int = 10_000
    peak_lr: float = 5e-5
    decay_lr: float = 5e-5
    warmup_steps: int = 1_000
    decay_steps: int = 500_000
    ema_decay: float | None = 0.999
    fsdp_devices: int = 8
    clip_gradient_norm: float = 1.0
    wandb_enabled: bool = False
    overwrite: bool = False
    resume: bool = False


@dataclass(frozen=True)
class PathSettings:
    assets_base_dir: str = "./artifacts/assets"
    checkpoint_base_dir: str = "./artifacts/checkpoints"
    base_checkpoint_path: str = "./pretrained/pi05_base/params"
    eval_output_dir: str = "./artifacts/eval"


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    project_root: Path
    project: ProjectSettings = field(default_factory=ProjectSettings)
    data: DataSettings = field(default_factory=DataSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    paths: PathSettings = field(default_factory=PathSettings)

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def resolve_maybe_remote(self, value: str) -> str:
        if "://" in value:
            return value
        return str(self.resolve_path(value))


def _build_section(section_type: type[Any], data: dict[str, Any] | None) -> Any:
    payload = data or {}
    allowed = {f.name for f in section_type.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in payload.items() if k in allowed}
    return section_type(**filtered)


def load_config(config_path: str | Path) -> PipelineConfig:
    path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return PipelineConfig(
        config_path=path,
        project_root=default_project_root(),
        project=_build_section(ProjectSettings, raw.get("project")),
        data=_build_section(DataSettings, raw.get("data")),
        model=_build_section(ModelSettings, raw.get("model")),
        training=_build_section(TrainingSettings, raw.get("training")),
        paths=_build_section(PathSettings, raw.get("paths")),
    )

