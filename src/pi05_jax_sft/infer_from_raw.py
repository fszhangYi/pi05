"""Offline π0.5 inference on a raw episode (steps.json + camera JPEGs).

Matches ACT ``infer_from_raw.py`` / true-robot ``serve`` input path: no LeRobot
dataset required. Uses ``Pi05Inference`` (same policy load as ``serve.py``) and
writes ACT-compatible infer JSON (``cartesian_abs`` chunks) for embody step 6.

Usage:
    python -m pi05_jax_sft.infer_from_raw \\
        --config configs/pi05_act_robot_smoke.yaml \\
        --checkpoint-dir artifacts/checkpoints/pi05_act_robot_smoke/convert_smoke \\
        --raw-root /root/autodl-tmp/act_robot/data/raw \\
        --episode 1 \\
        --infer-output-dir artifacts/infer/from_raw \\
        [--checkpoint-step 9999] [--max-frames 30]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from pi05_jax_sft.convert_company_dataset import (
    _apply_stride,
    _build_state_and_actions,
    _collect_frame_maps,
    _format_task,
    _load_steps,
    _read_annotation,
    _require_frames_from_map,
)
from pi05_jax_sft.npz_to_infer_json import _delta_chunk_to_abs
from pi05_jax_sft.project_config import load_config
from pi05_jax_sft.serve import Pi05Inference


def _slice_gt_chunk(
    gt_actions: np.ndarray,
    k: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Align GT action chunk with model output (same padding as ACT EpisodicDataset)."""
    valid_len = min(chunk_size, len(gt_actions) - k)
    gt_chunk = np.zeros((chunk_size, gt_actions.shape[1]), dtype=np.float32)
    if valid_len > 0:
        gt_chunk[:valid_len] = gt_actions[k : k + valid_len]
        if valid_len < chunk_size:
            gt_chunk[valid_len:] = gt_actions[k + valid_len - 1]
    is_pad = np.zeros(chunk_size, dtype=bool)
    if valid_len < chunk_size:
        is_pad[valid_len:] = True
    return gt_chunk, is_pad


def _chunk_mae(pred: np.ndarray, gt: np.ndarray, is_pad: np.ndarray) -> float:
    mask = ~is_pad
    if not mask.any():
        return float("nan")
    return float(np.abs(pred[mask] - gt[mask]).mean())


def _load_raw_episode(
    episode_dir: Path,
    cfg,
    *,
    stride: int = 1,
) -> dict[str, Any]:
    """Load one raw episode → state/actions/images + raw frame indices (no LeRobot)."""
    cart_all, grip_obs_all, grip_act_all = _load_steps(episode_dir)
    dataset_format = cfg.data.dataset_format
    row = column = None

    if dataset_format == "tonglu_annotation":
        if cfg.data.annotation_root is None:
            raise ValueError("data.annotation_root is required for dataset_format=tonglu_annotation")
        ann_path = cfg.resolve_path(cfg.data.annotation_root) / f"{episode_dir.name}.txt"
        if not ann_path.is_file():
            raise FileNotFoundError(f"annotation not found: {ann_path}")
        ann = _read_annotation(ann_path)
        if ann["end"] >= len(cart_all):
            raise ValueError(f"{episode_dir}: annotation end={ann['end']} >= timesteps={len(cart_all)}")
        frame_indices = _apply_stride(list(range(ann["start"], ann["end"] + 1)), stride)
        row, column = ann["row"], ann["column"]
        grip_act = None
    elif dataset_format == "company_steps":
        frame_indices = _apply_stride(list(range(len(cart_all))), stride)
        grip_act = grip_act_all[frame_indices] if grip_act_all is not None else None
    else:
        raise ValueError(
            f"Unsupported data.dataset_format={dataset_format!r}; "
            "use company_steps or tonglu_annotation"
        )

    if not frame_indices:
        raise ValueError(f"{episode_dir}: empty frame range after stride={stride}")

    cart = cart_all[frame_indices]
    grip_obs = grip_obs_all[frame_indices]
    # GT deltas match training converter; policy input is raw 7D (serve normalizes).
    _, actions = _build_state_and_actions(
        episode_dir,
        cart,
        grip_obs,
        normalize_rx=cfg.data.normalize_rx_to_2pi,
        gripper_action_source=cfg.data.gripper_action_source,
        grip_act=grip_act,
    )
    # Raw 7D proprio as the robot would send on the wire.
    state_raw = np.concatenate([cart, grip_obs[:, None]], axis=-1).astype(np.float32)

    cam_specs: list[tuple[str, str | None]] = [
        ("chest_image", cfg.data.chest_image_prefix),
        ("top_image", cfg.data.top_image_prefix),
        ("wrist_image", cfg.data.wrist_image_prefix),
    ]
    prefixes = [p for _, p in cam_specs if p]
    if not prefixes:
        raise ValueError("Need at least one image prefix (top/wrist/chest) in config")
    frame_maps = _collect_frame_maps(episode_dir, prefixes)
    frame_paths: dict[str, list[Path]] = {}
    for key, prefix in cam_specs:
        if not prefix:
            continue
        frame_paths[key] = _require_frames_from_map(
            episode_dir,
            frame_maps[prefix],
            prefix,
            frame_indices,
            key.replace("_image", ""),
        )

    return {
        "frame_indices": frame_indices,
        "state_raw": state_raw,
        "actions": actions,
        "frame_paths": frame_paths,
        "cart_all": cart_all,
        "task": _format_task(cfg, row, column),
        "source_episode": episode_dir.name,
    }


def _images_for_frame(frame_paths: dict[str, list[Path]], k: int) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for key, paths in frame_paths.items():
        out[key] = paths[k].read_bytes()
    return out


def infer_episode(
    inferencer: Pi05Inference,
    *,
    episode: int,
    raw_dir: Path,
    cfg,
    stride: int = 1,
    max_frames: int | None = None,
    task_prompt: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run inference on one raw episode; return {summary, frames} payload."""
    episode_dir = raw_dir / str(episode)
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"Episode dir not found: {episode_dir}")

    loaded = _load_raw_episode(episode_dir, cfg, stride=stride)
    frame_indices: list[int] = loaded["frame_indices"]
    state_raw: np.ndarray = loaded["state_raw"]
    gt_actions: np.ndarray = loaded["actions"]
    frame_paths: dict[str, list[Path]] = loaded["frame_paths"]
    cart_all: np.ndarray = loaded["cart_all"]
    task = task_prompt or loaded["task"]

    if max_frames is not None and max_frames > 0:
        frame_indices = frame_indices[:max_frames]
        state_raw = state_raw[:max_frames]
        gt_actions = gt_actions[:max_frames]
        frame_paths = {k: v[:max_frames] for k, v in frame_paths.items()}

    chunk_size = int(inferencer.chunk_size)
    results: list[dict] = []
    chunk0_maes: list[float] = []
    chunk_maes: list[float] = []

    if verbose:
        print(
            f"Episode {episode}  frames={len(frame_indices)}  "
            f"range=[{frame_indices[0]}, {frame_indices[-1]}]  "
            f"task={task!r}"
        )

    # Only pass cameras the policy expects.
    active = set(inferencer.active_image_keys)

    for k, raw_frame_idx in enumerate(frame_indices):
        image_bytes = {
            key: val
            for key, val in _images_for_frame(frame_paths, k).items()
            if key in active
        }
        if not image_bytes:
            raise RuntimeError(
                f"No active camera images for frame {raw_frame_idx}; "
                f"active={sorted(active)} available={sorted(frame_paths)}"
            )
        pred_delta = inferencer.infer_chunk(image_bytes, state_raw[k], task_prompt=task)
        gt_delta, is_pad = _slice_gt_chunk(gt_actions, k, chunk_size)

        # Embody step 6 expects cartesian_abs (compose SE(3) deltas onto raw poses).
        pred_abs = _delta_chunk_to_abs(pred_delta, cart_all, int(raw_frame_idx))
        gt_abs = _delta_chunk_to_abs(gt_delta, cart_all, int(raw_frame_idx))

        mae0 = float(np.abs(pred_abs[0] - gt_abs[0]).mean())
        mae_chunk = _chunk_mae(pred_abs, gt_abs, is_pad)
        chunk0_maes.append(mae0)
        chunk_maes.append(mae_chunk)
        results.append(
            {
                "raw_frame": int(raw_frame_idx),
                "qpos": state_raw[k].tolist(),
                "pred_chunk": pred_abs.tolist(),
                "gt_chunk": gt_abs.tolist(),
                "is_pad": is_pad.tolist(),
                "chunk0_mae": mae0,
                "chunk_mae": mae_chunk,
            }
        )
        if verbose and (k == 0 or (k + 1) % 20 == 0 or k + 1 == len(frame_indices)):
            print(f"  frame {k + 1}/{len(frame_indices)} raw={raw_frame_idx} chunk0_mae={mae0:.4f}")

    ckpt_path = getattr(inferencer, "_checkpoint_path", None)
    summary = {
        "episode": int(episode),
        "num_frames": len(frame_indices),
        "frame_count": len(frame_indices),
        "chunk_size": chunk_size,
        "frame_range": [frame_indices[0], frame_indices[-1]],
        "action_space": "cartesian_abs",
        "camera_keys": sorted(active),
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "repo_id": cfg.data.repo_id,
        "source": "pi05_infer_from_raw",
        "task": task,
        "mean_chunk0_mae": float(np.mean(chunk0_maes)) if chunk0_maes else float("nan"),
        "median_chunk0_mae": float(np.median(chunk0_maes)) if chunk0_maes else float("nan"),
        "mean_chunk_mae": float(np.nanmean(chunk_maes)) if chunk_maes else float("nan"),
        "median_chunk_mae": float(np.nanmedian(chunk_maes)) if chunk_maes else float("nan"),
    }
    if verbose:
        print(
            f"  mean chunk0_mae={summary['mean_chunk0_mae']:.4f}  "
            f"mean chunk_mae={summary['mean_chunk_mae']:.4f}"
        )
    return {"summary": summary, "frames": results}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Training YAML (cameras, norm_stats repo_id, task).")
    p.add_argument("--checkpoint-dir", required=True, help="Checkpoint run dir (contains step subdirs).")
    p.add_argument("--checkpoint-step", type=int, default=None, help="Step subdir; default = latest.")
    p.add_argument("--raw-root", default=None, help="Override data.raw_root.")
    p.add_argument("--annotation-root", default=None, help="Override data.annotation_root.")
    p.add_argument("--episode", type=int, required=True, help="Numeric raw episode folder name.")
    p.add_argument(
        "--infer-output-dir",
        type=Path,
        default=None,
        help="Write episode_<id>.json here (ACT infer JSON).",
    )
    p.add_argument("--output", type=Path, default=None, help="Optional explicit JSON path (overrides dir naming).")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None, help="Cap frames after slice/stride (debug).")
    p.add_argument("--action-horizon", type=int, default=None, help="Override model.action_horizon / chunk size.")
    p.add_argument("--task-prompt", default=None, help="Override episode task prompt.")
    p.add_argument("--hf-lerobot-home", default=None, help="Optional; only used to satisfy policy loader home.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)

    data_updates: dict[str, Any] = {}
    if args.raw_root:
        data_updates["raw_root"] = str(Path(args.raw_root).expanduser().resolve())
    if args.annotation_root:
        data_updates["annotation_root"] = str(Path(args.annotation_root).expanduser().resolve())
    if data_updates:
        cfg = replace(cfg, data=replace(cfg.data, **data_updates))

    raw_root = cfg.resolve_path(cfg.data.raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw-root not found: {raw_root}")

    lerobot_home = args.hf_lerobot_home
    if not lerobot_home:
        try:
            lerobot_home = str(cfg.resolve_path(cfg.data.hf_lerobot_home))
        except Exception:  # noqa: BLE001
            lerobot_home = None

    inferencer = Pi05Inference(
        config_path=str(Path(args.config).expanduser().resolve()),
        checkpoint_step=args.checkpoint_step,
        checkpoint_dir=str(Path(args.checkpoint_dir).expanduser().resolve()),
        num_steps_override=args.action_horizon,
        task_prompt=args.task_prompt,
        lerobot_home=lerobot_home,
    )
    # Stash resolved checkpoint path for summary (mirrors ACT helper).
    from pi05_jax_sft.runtime import latest_checkpoint_step

    ckpt_root = Path(args.checkpoint_dir).expanduser().resolve()
    step = args.checkpoint_step if args.checkpoint_step is not None else latest_checkpoint_step(ckpt_root)
    inferencer._checkpoint_path = str(ckpt_root / str(step))  # noqa: SLF001

    payload = infer_episode(
        inferencer,
        episode=int(args.episode),
        raw_dir=raw_root,
        cfg=cfg,
        stride=max(1, int(args.stride)),
        max_frames=args.max_frames,
        task_prompt=args.task_prompt,
        verbose=True,
    )
    summary = payload["summary"]
    print(
        f"\nMean chunk[0] MAE: {summary['mean_chunk0_mae']:.4f}  "
        f"median: {summary['median_chunk0_mae']:.4f}"
    )
    print(
        f"Mean full-chunk MAE (unpadded): {summary['mean_chunk_mae']:.4f}  "
        f"median: {summary['median_chunk_mae']:.4f}"
    )

    out_path: Path | None = None
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    elif args.infer_output_dir:
        out_dir = Path(args.infer_output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"episode_{int(args.episode)}.json"

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
