"""Convert raw company/Tonglu episodes to a LeRobot dataset for pi0.5 SFT.

Supported formats:

- ``company_steps``: the original wrist-only dataset under ``cam_100_15``.
- ``tonglu_annotation``: ``raw_data/<episode_id>`` plus
  ``annotation/<episode_id>.txt``. The annotation slices each raw episode and
  provides row/column values used to generate the task prompt.

State  (7D): [x, y, z, rx, ry, rz, gripper_obs]
Action (7D): [dx, dy, dz, drx, dry, drz, gripper_cmd]

The 6D action is always recomputed as ``inv(T_current) @ T_target`` from
observed Cartesian poses; the converter never trusts ``steps.json`` Cartesian
actions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import os
import re
import resource
import shutil
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from pi05_jax_sft.pose_utils import normalize_rx_to_2pi
from pi05_jax_sft.project_config import PipelineConfig
from pi05_jax_sft.project_config import load_config
from pi05_jax_sft.runtime import configure_lerobot_home


_FRAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
_LEROBOT_EPISODE_FILE_RE = re.compile(r"episode_(?P<idx>\d+)\.[^.]+$")
STATE_DIM = 7
ACTION_DIM = 7
TONG_LU_PROMPT = (
    "pick the workpiece from the cardboard box in order and place it in the "
    "{row} row, {column} column of the grid tray"
)


def _pose_to_se3(pose: np.ndarray) -> np.ndarray:
    """[x,y,z,rx,ry,rz] (xyz-extrinsic Euler, radians) -> 4x4 SE3."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    return T


def _se3_to_pose(T: np.ndarray) -> np.ndarray:
    """4x4 SE3 -> [x,y,z,rx,ry,rz] (xyz-extrinsic Euler, radians)."""
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_euler("xyz")])


def _relative_pose(curr: np.ndarray, target: np.ndarray) -> np.ndarray:
    """inv(T_curr) @ T_target expressed as xyz-extrinsic Euler pose."""
    T_rel = np.linalg.inv(_pose_to_se3(curr)) @ _pose_to_se3(target)
    return _se3_to_pose(T_rel)


def _relative_poses(cart: np.ndarray) -> np.ndarray:
    """Vectorized inv(T_t) @ T_{t+1} for an entire Cartesian pose sequence."""
    rel = np.zeros((len(cart), 6), dtype=np.float32)
    if len(cart) <= 1:
        return rel

    rotations = Rotation.from_euler("xyz", cart[:, 3:6])
    inv_curr = rotations[:-1].inv()
    rel[:-1, :3] = inv_curr.apply(cart[1:, :3] - cart[:-1, :3]).astype(np.float32)
    rel[:-1, 3:6] = (inv_curr * rotations[1:]).as_euler("xyz").astype(np.float32)
    return rel


def _episode_sort_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.name)) if path.name.isdigit() else (1, path.name)


def _read_gripper(values: Any, expected_len: int, episode_dir: Path, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        raise ValueError(f"{episode_dir}: {name} length mismatch, got shape {arr.shape}")
    return arr


def _collect_frame_map(episode_dir: Path, prefix: str) -> dict[int, Path]:
    frames: dict[int, Path] = {}
    for p in episode_dir.iterdir():
        m = _FRAME_RE.match(p.name)
        if m and m.group("prefix") == prefix:
            frames[int(m.group("idx"))] = p
    return frames


def _collect_frame_maps(episode_dir: Path, prefixes: list[str]) -> dict[str, dict[int, Path]]:
    wanted = set(prefixes)
    frames = {prefix: {} for prefix in prefixes}
    for p in episode_dir.iterdir():
        m = _FRAME_RE.match(p.name)
        if m and m.group("prefix") in wanted:
            frames[m.group("prefix")][int(m.group("idx"))] = p
    return frames


def _require_frames(
    episode_dir: Path,
    prefix: str,
    frame_indices: list[int],
    label: str,
) -> list[Path]:
    frame_map = _collect_frame_map(episode_dir, prefix)
    missing = [idx for idx in frame_indices if idx not in frame_map]
    if missing:
        preview = ", ".join(str(x) for x in missing[:8])
        raise ValueError(f"{episode_dir}: missing {label} frames for prefix {prefix}: {preview}")
    return [frame_map[idx] for idx in frame_indices]


def _require_frames_from_map(
    episode_dir: Path,
    frame_map: dict[int, Path],
    prefix: str,
    frame_indices: list[int],
    label: str,
) -> list[Path]:
    missing = [idx for idx in frame_indices if idx not in frame_map]
    if missing:
        preview = ", ".join(str(x) for x in missing[:8])
        raise ValueError(f"{episode_dir}: missing {label} frames for prefix {prefix}: {preview}")
    return [frame_map[idx] for idx in frame_indices]


def _read_annotation(path: Path) -> dict[str, int]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"{path}: expected at least two columns, got {line!r}")
        rows.append([float(parts[0]), float(parts[1])])

    if len(rows) < 12:
        raise ValueError(f"{path}: expected 12 annotation rows, got {len(rows)}")

    start = int(rows[0][1])
    end = int(rows[2][1])
    row = int(rows[10][1])
    column = int(rows[11][1])
    if start < 0 or end < start:
        raise ValueError(f"{path}: invalid frame range start={start}, end={end}")
    return {"start": start, "end": end, "row": row, "column": column}


def _format_task(cfg: PipelineConfig, row: int | None = None, column: int | None = None) -> str:
    template = cfg.data.task_prompt_template or cfg.data.task_name
    if row is not None and column is not None:
        return template.format(row=row, column=column)
    return template


@dataclass(frozen=True)
class EpisodeRecord:
    episode_dir: Path
    frame_paths: dict[str, list[Path]]
    state: np.ndarray
    actions: np.ndarray
    task: str
    source_episode: str
    start_frame: int
    end_frame: int
    row: int | None = None
    column: int | None = None


def _build_state_and_actions(
    episode_dir: Path,
    cart: np.ndarray,
    grip_obs: np.ndarray,
    *,
    normalize_rx: bool,
    gripper_action_source: str,
    grip_act: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if normalize_rx:
        state_cart = normalize_rx_to_2pi(cart)
    else:
        state_cart = cart.copy()

    state = np.concatenate([state_cart, grip_obs[:, None]], axis=-1).astype(np.float32)
    actions = np.zeros((len(cart), ACTION_DIM), dtype=np.float32)
    actions[:, :6] = _relative_poses(cart)

    if gripper_action_source == "actions":
        if grip_act is None:
            raise ValueError(f"{episode_dir}: gripper_action_source=actions but actions gripper is missing")
        actions[:, 6] = grip_act.astype(np.float32)
    elif gripper_action_source == "next_observation":
        if len(grip_obs) > 1:
            actions[:-1, 6] = grip_obs[1:].astype(np.float32)
        actions[-1, 6] = grip_obs[-1].astype(np.float32)
    elif gripper_action_source == "current_observation":
        actions[:, 6] = grip_obs.astype(np.float32)
    else:
        raise ValueError(
            f"{episode_dir}: unsupported gripper_action_source={gripper_action_source!r}; "
            "use actions, next_observation, or current_observation"
        )

    return state, actions


def _load_steps(episode_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    steps = json.loads((episode_dir / "steps.json").read_text(encoding="utf-8"))
    cart = np.asarray(steps["observations"]["cartesian_position"], dtype=np.float64)
    if cart.ndim != 2 or cart.shape[1] != 6:
        raise ValueError(f"{episode_dir}: cartesian_position must have shape (T, 6), got {cart.shape}")

    grip_obs = _read_gripper(steps["observations"]["gripper_position"], len(cart), episode_dir, "obs gripper")
    grip_act = None
    if "gripper_position" in steps.get("actions", {}):
        grip_act = _read_gripper(steps["actions"]["gripper_position"], len(cart), episode_dir, "action gripper")
    return cart, grip_obs, grip_act


def _load_company_episode(episode_dir: Path, cfg: PipelineConfig) -> EpisodeRecord:
    cart, grip_obs, grip_act = _load_steps(episode_dir)
    state, actions = _build_state_and_actions(
        episode_dir,
        cart,
        grip_obs,
        normalize_rx=cfg.data.normalize_rx_to_2pi,
        gripper_action_source=cfg.data.gripper_action_source,
        grip_act=grip_act,
    )

    frame_indices = list(range(len(cart)))
    frame_maps = _collect_frame_maps(episode_dir, [cfg.data.wrist_image_prefix])
    frame_paths = {
        "wrist_image": _require_frames_from_map(
            episode_dir,
            frame_maps[cfg.data.wrist_image_prefix],
            cfg.data.wrist_image_prefix,
            frame_indices,
            "wrist",
        ),
    }
    return EpisodeRecord(
        episode_dir=episode_dir,
        frame_paths=frame_paths,
        state=state,
        actions=actions,
        task=_format_task(cfg),
        source_episode=episode_dir.name,
        start_frame=0,
        end_frame=len(cart) - 1,
    )


def _load_tonglu_episode(episode_dir: Path, cfg: PipelineConfig) -> EpisodeRecord:
    if cfg.data.annotation_root is None:
        raise ValueError("data.annotation_root is required for dataset_format=tonglu_annotation")
    ann_path = cfg.resolve_path(cfg.data.annotation_root) / f"{episode_dir.name}.txt"
    if not ann_path.exists():
        raise FileNotFoundError(f"annotation not found: {ann_path}")

    ann = _read_annotation(ann_path)
    cart_all, grip_obs_all, _grip_act = _load_steps(episode_dir)
    if ann["end"] >= len(cart_all):
        raise ValueError(f"{episode_dir}: annotation end={ann['end']} >= timesteps={len(cart_all)}")

    frame_indices = list(range(ann["start"], ann["end"] + 1))
    cart = cart_all[frame_indices]
    grip_obs = grip_obs_all[frame_indices]
    state, actions = _build_state_and_actions(
        episode_dir,
        cart,
        grip_obs,
        normalize_rx=cfg.data.normalize_rx_to_2pi,
        gripper_action_source=cfg.data.gripper_action_source,
        grip_act=None,
    )

    if not cfg.data.chest_image_prefix or not cfg.data.top_image_prefix:
        raise ValueError("Tonglu conversion requires chest_image_prefix and top_image_prefix")

    frame_maps = _collect_frame_maps(
        episode_dir,
        [cfg.data.chest_image_prefix, cfg.data.top_image_prefix, cfg.data.wrist_image_prefix],
    )
    frame_paths = {
        "chest_image": _require_frames_from_map(
            episode_dir,
            frame_maps[cfg.data.chest_image_prefix],
            cfg.data.chest_image_prefix,
            frame_indices,
            "chest",
        ),
        "top_image": _require_frames_from_map(
            episode_dir,
            frame_maps[cfg.data.top_image_prefix],
            cfg.data.top_image_prefix,
            frame_indices,
            "top",
        ),
        "wrist_image": _require_frames_from_map(
            episode_dir,
            frame_maps[cfg.data.wrist_image_prefix],
            cfg.data.wrist_image_prefix,
            frame_indices,
            "wrist",
        ),
    }
    return EpisodeRecord(
        episode_dir=episode_dir,
        frame_paths=frame_paths,
        state=state,
        actions=actions,
        task=_format_task(cfg, ann["row"], ann["column"]),
        source_episode=episode_dir.name,
        start_frame=ann["start"],
        end_frame=ann["end"],
        row=ann["row"],
        column=ann["column"],
    )


def _load_episode(episode_dir: Path, cfg: PipelineConfig) -> EpisodeRecord:
    if cfg.data.dataset_format == "company_steps":
        return _load_company_episode(episode_dir, cfg)
    if cfg.data.dataset_format == "tonglu_annotation":
        return _load_tonglu_episode(episode_dir, cfg)
    raise ValueError(
        f"Unsupported data.dataset_format={cfg.data.dataset_format!r}; "
        "use company_steps or tonglu_annotation"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="YAML config path.")
    p.add_argument("--overwrite", action="store_true", help="Delete existing dataset.")
    p.add_argument("--resume", action="store_true",
                   default=os.environ.get("RESUME", "0") == "1",
                   help="Continue an existing LeRobot dataset by skipping completed parquet episodes.")
    p.add_argument("--dry-run", action="store_true", help="Validate parsing without creating LeRobot files.")
    p.add_argument("--image-writer-threads", type=int, default=int(os.environ.get("IMAGE_WRITER_THREADS", "4")))
    p.add_argument("--image-writer-processes", type=int, default=int(os.environ.get("IMAGE_WRITER_PROCESSES", "2")))
    p.add_argument("--decode-workers", type=int, default=int(os.environ.get("DECODE_WORKERS", "0")),
                   help="Decode images with bounded prefetch. 0 = sequential, lowest memory.")
    p.add_argument("--decode-prefetch", type=int, default=int(os.environ.get("DECODE_PREFETCH", "4")),
                   help="Maximum prefetched frames per image key when decode-workers > 0.")
    p.add_argument("--progress-interval", type=int, default=int(os.environ.get("PROGRESS_INTERVAL", "10")),
                   help="Print progress every N converted episodes.")
    p.add_argument("--episode-start-index", type=int, default=int(os.environ.get("EPISODE_START_INDEX", "0")),
                   help="Start from this 0-based sorted raw episode index.")
    p.add_argument("--episode-end-index", type=int, default=os.environ.get("EPISODE_END_INDEX"),
                   help="Stop before this 0-based sorted raw episode index.")
    p.add_argument("--verbose-episodes", action="store_true",
                   default=os.environ.get("VERBOSE_EPISODES", "0") == "1",
                   help="Print a line before every episode conversion.")
    p.add_argument("--validate-images-only", action="store_true",
                   help="Decode selected raw images and exit without creating a LeRobot dataset.")
    p.add_argument("--repair-resume", action="store_true",
                   default=os.environ.get("REPAIR_RESUME", "0") == "1",
                   help=(
                       "Before --resume, truncate local LeRobot files/metadata to the consecutive "
                       "episode parquet prefix. Useful after deleting a bad tail episode."
                   ))
    p.add_argument("--copy-original-images", action="store_true",
                   default=os.environ.get("COPY_ORIGINAL_IMAGES", "0") == "1",
                   help=(
                       "Copy original image files into the LeRobot episode table instead of decoding "
                       "and re-encoding them. Fast path for raw JPEG datasets."
                   ))
    return p.parse_args()


def _iter_episode_dirs(cfg: PipelineConfig) -> list[Path]:
    raw_root = cfg.resolve_path(cfg.data.raw_root)
    episode_dirs = sorted((p for p in raw_root.iterdir() if p.is_dir()), key=_episode_sort_key)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {raw_root}")
    if cfg.data.max_episodes is not None:
        episode_dirs = episode_dirs[: cfg.data.max_episodes]
    return episode_dirs


def _image_shape(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return (height, width, 3)


def _read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _read_episode_images(record: EpisodeRecord, decode_workers: int) -> list[dict[str, np.ndarray]]:
    """Compatibility wrapper for callers that want a materialized episode.

    The main conversion loop uses _iter_episode_images() to avoid holding an
    entire episode's decoded image arrays in memory.
    """
    return list(_iter_episode_images(record, decode_workers, decode_prefetch=4))


def _iter_episode_images(
    record: EpisodeRecord,
    decode_workers: int,
    decode_prefetch: int,
):
    image_keys = list(record.frame_paths)

    if decode_workers <= 1:
        for t in range(len(record.state)):
            yield {key: _read_image(record.frame_paths[key][t]) for key in image_keys}
        return

    # Keep only a small sliding window of decoded frames in memory. A full
    # Tonglu episode can be hundreds of frames x 3 views, so materializing all
    # decoded images is an easy way to OOM inside Docker.
    max_pending_frames = max(1, decode_prefetch)
    with ThreadPoolExecutor(max_workers=decode_workers) as pool:
        pending: dict[int, dict[str, Any]] = {}

        def submit_frame(frame_idx: int) -> None:
            pending[frame_idx] = {
                key: pool.submit(_read_image, record.frame_paths[key][frame_idx])
                for key in image_keys
            }

        next_submit = 0
        while next_submit < len(record.state) and len(pending) < max_pending_frames:
            submit_frame(next_submit)
            next_submit += 1

        for t in range(len(record.state)):
            futures = pending.pop(t)
            frame = {key: future.result() for key, future in futures.items()}
            while next_submit < len(record.state) and len(pending) < max_pending_frames:
                submit_frame(next_submit)
                next_submit += 1
            yield frame


def _write_summary(cfg: PipelineConfig, summary: dict[str, Any]) -> Path:
    summary_dir = cfg.resolve_path("./artifacts/prepare")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{cfg.project.name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def _count_completed_lerobot_episodes(output_path: Path) -> int:
    """Count episodes that have a completed parquet file."""
    return len(sorted(output_path.glob("data/**/*.parquet")))


def _episode_index_from_path(path: Path) -> int | None:
    match = _LEROBOT_EPISODE_FILE_RE.match(path.name)
    return int(match.group("idx")) if match else None


def _consecutive_completed_lerobot_episodes(output_path: Path) -> int:
    indices = {
        idx
        for idx in (_episode_index_from_path(path) for path in output_path.glob("data/**/*.parquet"))
        if idx is not None
    }
    count = 0
    while count in indices:
        count += 1
    return count


def _filter_lerobot_jsonl_by_episode(path: Path, keep_episodes: int) -> int:
    if not path.exists():
        return 0

    kept_lines: list[str] = []
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError:
            kept_lines.append(line)
            continue
        episode_index = item.get("episode_index")
        if episode_index is None:
            kept_lines.append(line)
            continue
        if int(episode_index) < keep_episodes:
            kept_lines.append(json.dumps(item, ensure_ascii=False))
        else:
            removed += 1

    path.write_text(("\n".join(kept_lines) + "\n") if kept_lines else "", encoding="utf-8")
    return removed


def _read_lerobot_episode_lengths(path: Path, keep_episodes: int) -> list[int]:
    if not path.exists():
        return []

    lengths: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        episode_index = item.get("episode_index")
        if episode_index is None or int(episode_index) >= keep_episodes:
            continue
        length = item.get("length")
        if length is not None:
            lengths.append(int(length))
    return lengths


def _count_lerobot_metadata_episodes(output_path: Path) -> int:
    episodes_path = output_path / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        return 0

    count = 0
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "episode_index" in item:
            count += 1
    return count


def _update_lerobot_info_for_resume(output_path: Path, keep_episodes: int) -> None:
    info_path = output_path / "meta" / "info.json"
    if not info_path.exists():
        return

    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_episodes"] = keep_episodes
    lengths = _read_lerobot_episode_lengths(output_path / "meta" / "episodes.jsonl", keep_episodes)
    if lengths:
        info["total_frames"] = sum(lengths)
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def _repair_lerobot_resume_dataset(output_path: Path) -> int:
    """Rollback a local LeRobot dataset to its consecutive parquet prefix."""
    keep_episodes = _consecutive_completed_lerobot_episodes(output_path)
    removed_files = 0
    for path in output_path.rglob("*"):
        if not path.is_file():
            continue
        episode_index = _episode_index_from_path(path)
        if episode_index is not None and episode_index >= keep_episodes:
            path.unlink()
            removed_files += 1

    removed_meta = 0
    for name in ("episodes.jsonl", "episodes_stats.jsonl"):
        removed_meta += _filter_lerobot_jsonl_by_episode(output_path / "meta" / name, keep_episodes)
    _update_lerobot_info_for_resume(output_path, keep_episodes)

    print(
        f"Repair resume: kept episodes [0, {keep_episodes}), "
        f"removed {removed_files} episode file(s), {removed_meta} metadata row(s).",
        flush=True,
    )
    return keep_episodes


def _memory_summary() -> str:
    parts = []
    try:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        parts.append(f"maxrss={rss_mb:.0f}MB")
    except Exception:
        pass
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            current_mb = int(Path(path).read_text().strip()) / 1024.0 / 1024.0
            parts.append(f"cgroup={current_mb:.0f}MB")
            break
        except Exception:
            continue
    return " ".join(parts) if parts else "mem=unknown"


def _make_lerobot_frame_adder(dataset: Any):
    """Return a fast add_frame wrapper for the installed LeRobot version.

    Some versions expect ``dataset.add_frame(frame, task=...)`` while others
    read ``task`` from the frame dict. Keep both paths because training machines
    often have a different LeRobot revision than the dev machine.
    """
    signature = inspect.signature(dataset.add_frame)
    if "task" in signature.parameters:
        return lambda frame, task: dataset.add_frame(frame, task=task)
    return lambda frame, task: dataset.add_frame({**frame, "task": task})


def _install_copy_original_image_writer(dataset: Any, image_suffix: str = ".jpg") -> None:
    """Patch a LeRobotDataset instance to copy image files instead of encoding arrays."""
    original_save_image = dataset._save_image

    def get_image_file_path(episode_index: int, image_key: str, frame_index: int) -> Path:
        return (
            dataset.root
            / "images"
            / image_key
            / f"episode_{episode_index:06d}"
            / f"frame_{frame_index:06d}{image_suffix}"
        )

    def save_image(image: Any, fpath: Path) -> None:
        source = getattr(image, "filename", None)
        if source:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, fpath)
            return
        original_save_image(image, fpath)

    dataset._get_image_file_path = get_image_file_path
    dataset._save_image = save_image


def _open_episode_image_refs(record: EpisodeRecord, frame_index: int) -> dict[str, Image.Image]:
    return {key: Image.open(record.frame_paths[key][frame_index]) for key in record.frame_paths}


def _copied_image_suffix(records: list[EpisodeRecord]) -> str:
    for record in records:
        for paths in record.frame_paths.values():
            if paths:
                suffix = paths[0].suffix.lower()
                return ".jpg" if suffix == ".jpeg" else suffix
    return ".jpg"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    raw_root = cfg.resolve_path(cfg.data.raw_root)
    episode_dirs = _iter_episode_dirs(cfg)
    episode_end_index = int(args.episode_end_index) if args.episode_end_index is not None else len(episode_dirs)
    episode_dirs = episode_dirs[args.episode_start_index:episode_end_index]
    if not episode_dirs:
        raise ValueError(
            f"Selected episode range is empty: start={args.episode_start_index}, end={episode_end_index}"
        )

    records: list[EpisodeRecord] = []
    skipped: list[dict[str, str]] = []
    t_load = time.monotonic()
    for episode_dir in episode_dirs:
        try:
            records.append(_load_episode(episode_dir, cfg))
        except Exception as exc:
            skipped.append({"episode": episode_dir.name, "reason": str(exc)})

    if not records:
        raise RuntimeError(f"No valid episodes found under {raw_root}; skipped={skipped[:5]}")

    image_features = {
        key: {
            "dtype": "image",
            "shape": _image_shape(paths[0]),
            "names": ["height", "width", "channel"],
        }
        for key, paths in records[0].frame_paths.items()
    }

    total_frames = sum(len(r.state) for r in records)
    common_summary = {
        "config_path": str(cfg.config_path),
        "dataset_format": cfg.data.dataset_format,
        "raw_root": str(raw_root),
        "annotation_root": str(cfg.resolve_path(cfg.data.annotation_root)) if cfg.data.annotation_root else None,
        "repo_id": cfg.data.repo_id,
        "episodes": len(records),
        "episode_start_index": args.episode_start_index,
        "episode_end_index": episode_end_index,
        "frames": total_frames,
        "skipped": skipped,
        "load_metadata_seconds": round(time.monotonic() - t_load, 3),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "image_keys": sorted(records[0].frame_paths),
        "normalize_rx_to_2pi": cfg.data.normalize_rx_to_2pi,
        "gripper_action_source": cfg.data.gripper_action_source,
        "first_episode": {
            "source_episode": records[0].source_episode,
            "start_frame": records[0].start_frame,
            "end_frame": records[0].end_frame,
            "row": records[0].row,
            "column": records[0].column,
            "task": records[0].task,
            "state0": records[0].state[0].tolist(),
            "action0": records[0].actions[0].tolist(),
        },
        "project": asdict(cfg.project),
        "data": asdict(cfg.data),
    }

    if args.dry_run:
        summary_path = _write_summary(cfg, {**common_summary, "dry_run": True, "output_path": None})
        print(
            f"Dry-run OK: {len(records)} episodes, {total_frames} frames, "
            f"{len(skipped)} skipped. Summary: {summary_path}"
        )
        return

    if args.validate_images_only:
        t_validate = time.monotonic()
        checked_frames = 0
        for episode_idx, record in enumerate(records, start=1):
            print(
                f"validate [{episode_idx}/{len(records)}] source={record.source_episode} "
                f"frames={len(record.state)} {record.start_frame}-{record.end_frame} {_memory_summary()}",
                flush=True,
            )
            for _frame in _iter_episode_images(record, args.decode_workers, args.decode_prefetch):
                checked_frames += 1
        summary_path = _write_summary(
            cfg,
            {
                **common_summary,
                "dry_run": False,
                "validate_images_only": True,
                "checked_frames": checked_frames,
                "validate_seconds": round(time.monotonic() - t_validate, 3),
                "output_path": None,
            },
        )
        print(f"Image validation OK: {checked_frames} frames. Summary: {summary_path}")
        return

    configure_lerobot_home(cfg.resolve_path(cfg.data.hf_lerobot_home))
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[no-redef]

    import os as _os
    lerobot_home = Path(_os.environ["HF_LEROBOT_HOME"])
    output_path = lerobot_home / cfg.data.repo_id
    existing_episodes = 0
    if output_path.exists() and args.resume:
        if args.repair_resume:
            existing_episodes = _repair_lerobot_resume_dataset(output_path)
        else:
            existing_episodes = _count_completed_lerobot_episodes(output_path)
            consecutive_episodes = _consecutive_completed_lerobot_episodes(output_path)
            metadata_episodes = _count_lerobot_metadata_episodes(output_path)
            if consecutive_episodes != existing_episodes or metadata_episodes != existing_episodes:
                raise ValueError(
                    f"Existing dataset has {existing_episodes} parquet episode(s), "
                    f"{metadata_episodes} metadata episode row(s), and only the first "
                    f"{consecutive_episodes} parquet episode(s) are consecutive. Re-run with "
                    "REPAIR_RESUME=1 to truncate the broken tail before resuming."
                )
        if existing_episodes > len(records):
            raise ValueError(
                f"Existing dataset has {existing_episodes} completed episodes, "
                f"but selected raw range only has {len(records)} valid episodes."
            )
        print(
            f"Resume enabled: found {existing_episodes} completed episode parquet(s) in {output_path}; "
            f"skipping the same number of selected raw episodes.",
            flush=True,
        )
        records = records[existing_episodes:]
        if not records:
            summary_path = _write_summary(
                cfg,
                {
                    **common_summary,
                    "dry_run": False,
                    "resume": True,
                    "existing_episodes": existing_episodes,
                    "output_path": str(output_path),
                    "convert_seconds": 0.0,
                },
            )
            print(f"Nothing to do; all selected episodes are already complete. Summary: {summary_path}")
            return
    elif output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset already exists at {output_path}. Use --overwrite to replace.")
        shutil.rmtree(output_path)

    if args.resume and output_path.exists():
        dataset = LeRobotDataset(cfg.data.repo_id)
    else:
        dataset = LeRobotDataset.create(
            repo_id=cfg.data.repo_id,
            robot_type=cfg.data.robot_type,
            fps=cfg.data.fps,
            features={
                **image_features,
                "state": {"dtype": "float32", "shape": (STATE_DIM,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
            },
            image_writer_threads=0 if args.copy_original_images else args.image_writer_threads,
            image_writer_processes=0 if args.copy_original_images else args.image_writer_processes,
        )
    if args.copy_original_images:
        _install_copy_original_image_writer(dataset, _copied_image_suffix(records))
        print(
            "Copy original images enabled: using source image bytes instead of decoding/re-encoding.",
            flush=True,
        )
    add_lerobot_frame = _make_lerobot_frame_adder(dataset)

    t_convert = time.monotonic()
    converted_frames = 0
    for episode_idx, record in enumerate(records, start=1):
        if args.verbose_episodes:
            print(
                f"convert [{episode_idx}/{len(records)}] source={record.source_episode} "
                f"frames={len(record.state)} {record.start_frame}-{record.end_frame} {_memory_summary()}",
                flush=True,
            )
        if args.copy_original_images:
            for t in range(len(record.state)):
                frame = _open_episode_image_refs(record, t)
                try:
                    add_lerobot_frame({
                        **frame,
                        "state": record.state[t],
                        "actions": record.actions[t],
                    }, record.task)
                finally:
                    for image in frame.values():
                        image.close()
        else:
            for t, frame in enumerate(
                _iter_episode_images(record, args.decode_workers, args.decode_prefetch)
            ):
                add_lerobot_frame({
                    **frame,
                    "state": record.state[t],
                    "actions": record.actions[t],
                }, record.task)
        dataset.save_episode()
        converted_frames += len(record.state)
        if args.progress_interval > 0 and episode_idx % args.progress_interval == 0:
            elapsed = time.monotonic() - t_convert
            fps = converted_frames / max(elapsed, 1e-6)
            print(f"[{episode_idx}/{len(records)}] converted, {fps:.1f} frames/s, elapsed={elapsed:.1f}s")

    summary_path = _write_summary(
        cfg,
        {
            **common_summary,
            "dry_run": False,
            "output_path": str(output_path),
            "resume": args.resume,
            "existing_episodes": existing_episodes,
            "image_writer_threads": args.image_writer_threads,
            "image_writer_processes": args.image_writer_processes,
            "decode_workers": args.decode_workers,
            "decode_prefetch": args.decode_prefetch,
            "copy_original_images": args.copy_original_images,
            "convert_seconds": round(time.monotonic() - t_convert, 3),
        },
    )
    print(
        f"Converted {len(records)} episodes ({total_frames} frames) -> {output_path}. "
        f"Skipped {len(skipped)}. Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()