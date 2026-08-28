#!/usr/bin/env python3
"""pi0.5 inference server.

Speaks the same TCP wire protocol as act_robot/serve.py, so the existing
robot client works without modification.

Legacy wire protocol (client -> server per step):
    4B  wrist_image_length   (big-endian uint32)
    N B wrist JPEG bytes
    4B  extra_image_length   (big-endian uint32, received but ignored)
    M B extra image bytes    (ignored ¨C kept for protocol compatibility)
    28B robot_state          (7 ¡Á float32, big-endian) [x,y,z,rx,ry,rz,gripper]
    4B  refresh flag         (uint32; 1 = reset episode, 0 = continue)

Server ¡ú client per step:
    28B next_state_absolute  (7 ¡Á float32, big-endian)
                             = T_current @ T_predicted_action

If the policy was trained with an 8th action dimension for task status/progress,
set model.policy_action_dim=8 in the YAML config. The server keeps the wire
protocol unchanged: dims 0..6 drive the robot and dim 7 is logged as task_status.

Action chunking modes (choose at startup):
    default         ¨C query policy every chunk_size steps; replay chunk in sequence.
    --temporal-agg  ¨C query every step; aggregate overlapping chunks with exp-decay
                      weights (k=0.01, newest = highest weight). Smoother but slower.

Usage:
    python -m pi05_jax_sft.serve \\
        --config  configs/pi05_company_example.yaml \\
        --checkpoint-step 20000 \\
        --port 5000 \\
        [--task-prompt "..."] \\
        [--image-protocol legacy|three-view] \\
        [--temporal-agg] \\
        [--num-steps 10]          # override action chunk size
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import socket
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from pi05_jax_sft.pose_utils import normalize_rx_to_2pi

ROBOT_CONTROL_DIM = 7
TASK_STATUS_INDEX = 7
DEFAULT_RTC_PREFIX_POS_TOLERANCE = 0.01
DEFAULT_RTC_PREFIX_ROT_TOLERANCE = 0.08


@dataclasses.dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray
    model_actions: np.ndarray


# ---------------------------------------------------------------------------
# SE(3) helpers  (xyz-extrinsic Euler angles, same convention as training)
# ---------------------------------------------------------------------------

def _euler_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    rx, ry, rz = float(euler[0]), float(euler[1]), float(euler[2])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def _pose_to_se3(pose6: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pose6[:3]
    T[:3, :3] = _euler_xyz_to_matrix(pose6[3:6])
    return T


def _se3_to_pose(T: np.ndarray) -> np.ndarray:
    pose = np.zeros(6, dtype=np.float64)
    pose[:3] = T[:3, 3]
    pose[3:6] = _matrix_to_euler_xyz(T[:3, :3])
    return pose


def compose_pose(current_pose: np.ndarray, action: np.ndarray, *, normalize_rx: bool = False) -> np.ndarray:
    """Apply a local-frame SE(3) action to the current pose.

    T_next = T_current @ T_action
    Matches the training convention: action = inv(T_t) @ T_{t+1}.

    Args:
        current_pose: [x, y, z, rx, ry, rz, gripper]   float32/64
        action:       [dx, dy, dz, drx, dry, drz, gripper_cmd, ...]  float32/64
    Returns:
        next_pose:    [x, y, z, rx, ry, rz, gripper]   float32
    """
    T_cur = _pose_to_se3(np.asarray(current_pose, dtype=np.float64))
    T_act = _pose_to_se3(np.asarray(action[:6], dtype=np.float64))
    T_nxt = T_cur @ T_act
    result = np.zeros(7, dtype=np.float32)
    result[:6] = _se3_to_pose(T_nxt).astype(np.float32)
    if normalize_rx:
        result[:6] = normalize_rx_to_2pi(result[:6]).astype(np.float32)
    result[6] = float(action[6])   # absolute gripper command
    return result


def rollout_action_chunk(
    current_pose: np.ndarray,
    actions: np.ndarray,
    *,
    normalize_rx: bool = False,
) -> np.ndarray:
    """Convert step-relative actions into absolute target poses."""
    pose = np.asarray(current_pose, dtype=np.float32).copy()
    targets = []
    for action in actions:
        pose = compose_pose(pose, action, normalize_rx=normalize_rx)
        targets.append(pose)
    if not targets:
        return np.zeros((0, ROBOT_CONTROL_DIM), dtype=np.float32)
    return np.stack(targets, axis=0).astype(np.float32)


def _task_status(action: np.ndarray) -> float | None:
    if action.shape[-1] <= TASK_STATUS_INDEX:
        return None
    return float(np.clip(action[TASK_STATUS_INDEX], 0.0, 1.0))


def _angle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _pose_error(target_pose: np.ndarray, current_pose: np.ndarray) -> tuple[float, float]:
    pos_err = float(np.linalg.norm(target_pose[:3] - current_pose[:3]))
    rot_err = float(np.linalg.norm(_angle_delta(target_pose[3:6], current_pose[3:6])))
    return pos_err, rot_err


def _trim_reached_rtc_prefix(
    robot_state: np.ndarray,
    previous_targets: np.ndarray | None,
    prefix_start: int,
    prefix_end: int,
    *,
    pos_tolerance: float,
    rot_tolerance: float,
) -> tuple[int, int | None, float | None, float | None]:
    """Drop prefix actions whose absolute targets have already been reached.

    RTC requests are asynchronous: while the server is receiving images and
    running inference, the robot may continue executing the old queue.  If the
    client reports a stale prefix, applying that prefix again from the latest
    robot_state creates a small target jump at the chunk boundary.
    """
    if previous_targets is None or prefix_end <= prefix_start:
        return prefix_start, None, None, None

    prefix_start = max(0, min(prefix_start, len(previous_targets)))
    prefix_end = max(prefix_start, min(prefix_end, len(previous_targets)))
    if prefix_end <= prefix_start:
        return prefix_start, None, None, None

    best_idx: int | None = None
    best_pos: float | None = None
    best_rot: float | None = None
    best_score = float("inf")
    for idx in range(prefix_start, len(previous_targets)):
        if not np.all(np.isfinite(previous_targets[idx, :ROBOT_CONTROL_DIM])):
            continue
        pos_err, rot_err = _pose_error(previous_targets[idx], robot_state)
        score = pos_err / max(pos_tolerance, 1e-6) + rot_err / max(rot_tolerance, 1e-6)
        if score < best_score:
            best_score = score
            best_idx = idx
            best_pos = pos_err
            best_rot = rot_err

    if (
        best_idx is None
        or best_pos is None
        or best_rot is None
        or best_pos > pos_tolerance
        or best_rot > rot_tolerance
    ):
        return prefix_start, None, None, None

    return max(prefix_start, min(best_idx + 1, prefix_end)), best_idx, best_pos, best_rot


# ---------------------------------------------------------------------------
# Action chunk state (identical logic to act_robot/serve.py)
# ---------------------------------------------------------------------------

class ClientState:
    """Per-connection bookkeeping for action chunking / temporal aggregation."""

    def __init__(self, chunk_size: int, temporal_agg: bool, k: float = 0.01):
        self.chunk_size = chunk_size
        self.temporal_agg = temporal_agg
        self.k = k
        self.t = 0
        if temporal_agg:
            self.history: list[tuple[int, np.ndarray]] = []
        else:
            self.current_chunk: np.ndarray | None = None
            self.chunk_offset: int = chunk_size  # triggers re-query on first call

    def reset(self) -> None:
        self.t = 0
        if self.temporal_agg:
            self.history = []
        else:
            self.current_chunk = None
            self.chunk_offset = self.chunk_size

    def needs_query(self) -> bool:
        if self.temporal_agg:
            return True
        return self.chunk_offset >= self.chunk_size / 2

    def add_chunk(self, chunk: np.ndarray) -> None:
        """Store a newly predicted chunk (shape [chunk_size, action_dim])."""
        if self.temporal_agg:
            self.history.append((self.t, chunk.copy()))
            self.history = [(t0, c) for t0, c in self.history
                            if t0 + self.chunk_size > self.t]
        else:
            self.current_chunk = chunk.copy()
            self.chunk_offset = 0

    def get_action(self) -> np.ndarray:
        if self.temporal_agg:
            relevant = [
                (t0, chunk[self.t - t0])
                for t0, chunk in self.history
                if 0 <= self.t - t0 < self.chunk_size
            ]
            if not relevant:
                raise RuntimeError("No predictions cover the current timestep.")
            relevant.sort(key=lambda x: x[0])
            n = len(relevant)
            weights = np.exp(-self.k * np.arange(n - 1, -1, -1))
            weights /= weights.sum()
            return sum(w * a for w, (_, a) in zip(weights, relevant))
        else:
            if self.current_chunk is None:
                raise RuntimeError("No chunk available.")
            action = self.current_chunk[self.chunk_offset]
            self.chunk_offset += 1
            return action

    def step(self) -> None:
        self.t += 1


# ---------------------------------------------------------------------------
# pi0.5 model wrapper
# ---------------------------------------------------------------------------

class Pi05Inference:
    """Loads a trained pi0.5 checkpoint and provides chunk inference."""

    def __init__(
        self,
        config_path: str,
        checkpoint_step: int | None,
        num_steps_override: int | None,
        task_prompt: str | None,
    ):
        # --- import project modules (openpi path inserted by ensure_openpi_importable) ---
        from pi05_jax_sft.project_config import load_config
        from pi05_jax_sft.runtime import (
            build_train_config,
            configure_lerobot_home,
            ensure_openpi_importable,
            latest_checkpoint_step,
        )

        cfg = load_config(config_path)
        configure_lerobot_home(cfg.resolve_path(cfg.data.hf_lerobot_home))
        ensure_openpi_importable(cfg.project_root)

        from openpi.policies import policy_config as pc

        train_cfg = build_train_config(cfg)
        # object.__setattr__(train_cfg.model, "pytorch_compile_mode", None)
        object.__setattr__(train_cfg.model, "pytorch_compile_mode", "max-autotune-no-cudagraphs")
        # object.__setattr__(train_cfg.model, "pytorch_compile_mode", "max-autotune")
        checkpoint_root = train_cfg.checkpoint_dir
        step = checkpoint_step if checkpoint_step is not None else latest_checkpoint_step(checkpoint_root)
        checkpoint_dir = checkpoint_root / str(step)
        prompt = task_prompt or cfg.data.task_name

        print(f"Loading checkpoint: {checkpoint_dir}")
        self._policy = pc.create_trained_policy(
            train_cfg,
            checkpoint_dir,
            default_prompt=prompt,
        )

        self.chunk_size = num_steps_override if num_steps_override is not None else cfg.model.action_horizon
        self.task = prompt
        self.normalize_rx = cfg.data.normalize_rx_to_2pi
        self._lock = threading.Lock()

        # Warm up inference so the first real call is fast.
        print("Warming up policy inference ¡­")
        t0 = time.monotonic()
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_state = np.zeros(7, dtype=np.float32)
        self._infer_chunk_raw({"wrist_image": dummy_img}, dummy_state)
        print(f"Warm-up done in {time.monotonic() - t0:.1f} s")

    def _infer_chunk_raw(
        self,
        images: dict[str, np.ndarray],
        robot_state: np.ndarray,
        action_prefix: np.ndarray | None = None,
        task_prompt: str | None = None,
    ) -> ActionChunk:
        """Run a single policy query. Returns unnormalized actions plus model-space actions."""
        state = robot_state.astype(np.float32)
        if self.normalize_rx:
            state = normalize_rx_to_2pi(state).astype(np.float32)
        obs = {
            **images,
            "state": state,
            "prompt": task_prompt or self.task,
        }
        sample_kwargs = None
        if action_prefix is not None and len(action_prefix) > 0:
            sample_kwargs = {
                "action_prefix": np.asarray(action_prefix, dtype=np.float32),
                "prefix_lengths": np.asarray([len(action_prefix)], dtype=np.int64),
                "num_steps": 8
            }
        result = self._policy.infer(obs, sample_kwargs=sample_kwargs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] < ROBOT_CONTROL_DIM:
            raise ValueError(f"policy returned actions with invalid shape {actions.shape}; need [H, >=7]")
        model_actions = np.asarray(result.get("model_actions", actions), dtype=np.float32)
        # Clip to requested chunk_size in case action_horizon differs
        return ActionChunk(
            actions=actions[: self.chunk_size],
            model_actions=model_actions[: self.chunk_size],
        )

    def infer_chunk(
        self,
        image_bytes: dict[str, bytes],
        robot_state: np.ndarray,
        action_prefix: np.ndarray | None = None,
        task_prompt: str | None = None,
    ) -> ActionChunk:
        """Decode JPEG and run inference. Thread-safe."""
        images = {
            key: np.asarray(Image.open(io.BytesIO(value)).convert("RGB"), dtype=np.uint8)
            for key, value in image_bytes.items()
        }
        with self._lock:
            return self._infer_chunk_raw(
                images,
                robot_state,
                action_prefix=action_prefix,
                task_prompt=task_prompt,
            )


# ---------------------------------------------------------------------------
# Per-client TCP handler
# ---------------------------------------------------------------------------

def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        pkt = conn.recv(n - len(buf))
        if not pkt:
            return None
        buf += pkt
    return buf


def _recv_framed_image(conn: socket.socket) -> bytes | None:
    raw = _recv_exact(conn, 4)
    if raw is None:
        return None
    img_len = struct.unpack(">I", raw)[0]
    return _recv_exact(conn, img_len)


def handle_client(
    conn: socket.socket,
    addr: tuple,
    inferencer: Pi05Inference,
    temporal_agg: bool,
    image_protocol: str,
    response_protocol: str,
    rtc_send_steps: int,
    rtc_prefix_pos_tolerance: float,
    rtc_prefix_rot_tolerance: float,
) -> None:
    print(f"[{addr}] connected")
    state = ClientState(inferencer.chunk_size, temporal_agg)
    last_rtc_chunk: ActionChunk | None = None
    last_rtc_sent_targets: np.ndarray | None = None
    last_rtc_chunk_id = 0

    # Unified log dir (serve.py style)
    now = datetime.now()
    baseline_log_dir = Path("logs") / now.strftime('%Y%m%d') / now.strftime('%H%M%S')
    step = 0

    try:
        while True:
            step += 1

            # ============================================================
            # RTC protocol path  (lmm-rtc-chunk)
            # ============================================================
            print("step: ", step, response_protocol)
            if response_protocol == "lmm-rtc-chunk":
                if image_protocol == "three-view":
                    image_data: dict[str, bytes] = {}
                    for key in ("chest_image", "top_image", "wrist_image"):
                        payload = _recv_framed_image(conn)
                        if payload is None:
                            break
                        image_data[key] = payload
                    print("len(image_data) ", len(image_data))
                    if len(image_data) != 3:
                        break

                else:
                    wrist_data = _recv_framed_image(conn)
                    if wrist_data is None:
                        break
                    image_data = {"wrist_image": wrist_data}
                    raw = _recv_exact(conn, 4)
                    if raw is None:
                        break
                    extra_len = struct.unpack(">I", raw)[0]
                    if extra_len > 0 and _recv_exact(conn, extra_len) is None:
                        break

                request_text = ""
                raw = _recv_exact(conn, 4)
                if raw is None:
                    break
                text_len = struct.unpack(">I", raw)[0]
                raw = _recv_exact(conn, text_len)
                if raw is None:
                    break
                request_text = raw.decode("utf-8")
                print("request_text ", request_text)

                raw = _recv_exact(conn, 7 * 4)
                if raw is None:
                    break
                robot_state = np.array(struct.unpack(">7f", raw), dtype=np.float32)

                raw = _recv_exact(conn, 4)
                if raw is None:
                    break
                refresh = struct.unpack(">I", raw)[0]
                print("refresh ", refresh)
                if refresh:
                    state.reset()
                    last_rtc_chunk = None
                    last_rtc_sent_targets = None
                    last_rtc_chunk_id = 0
                    print(f"[{addr}] episode reset at step {step}")

                rtc_prefix_chunk_id = 0
                rtc_prefix_start = 0
                rtc_prefix_len = 0
                raw = _recv_exact(conn, 3 * 4)
                if raw is None:
                    break
                rtc_prefix_chunk_id, rtc_prefix_start, rtc_prefix_len = struct.unpack(">III", raw)
                print("rtc_prefix_chunk_id, rtc_prefix_start, rtc_prefix_len ", rtc_prefix_chunk_id, rtc_prefix_start, rtc_prefix_len)
                try:
                    action_prefix = None
                    prefix_len_used = 0
                    prefix_trim_msg = ""
                    if (
                        not refresh
                        and last_rtc_chunk is not None
                        and rtc_prefix_chunk_id == last_rtc_chunk_id
                        and rtc_prefix_len > 0
                    ):
                        prefix_start = max(0, min(int(rtc_prefix_start), len(last_rtc_chunk.model_actions)))
                        prefix_end = max(
                            prefix_start,
                            min(prefix_start + int(rtc_prefix_len), len(last_rtc_chunk.model_actions)),
                        )
                        original_prefix_start = prefix_start
                        prefix_start, reached_idx, reached_pos, reached_rot = _trim_reached_rtc_prefix(
                            robot_state,
                            last_rtc_sent_targets,
                            prefix_start,
                            prefix_end,
                            pos_tolerance=rtc_prefix_pos_tolerance,
                            rot_tolerance=rtc_prefix_rot_tolerance,
                        )
                        if reached_idx is not None:
                            skipped = prefix_start - original_prefix_start
                            prefix_trim_msg = (
                                f", rtc_prefix_skip={skipped}"
                                f"@{reached_idx}"
                                f"({reached_pos * 1000:.1f}mm/{reached_rot:.3f}rad)"
                            )
                        action_prefix = last_rtc_chunk.model_actions[prefix_start:prefix_end].copy()
                        prefix_len_used = len(action_prefix)

                    t_infer = time.monotonic()
                    print("infer....")
                    chunk = inferencer.infer_chunk(
                        image_data,
                        robot_state,
                        action_prefix=action_prefix,
                        task_prompt=request_text or None,
                    )
                    last_rtc_chunk_id += 1
                    last_rtc_chunk = chunk

                    send_start = prefix_len_used
                    send_count = min(rtc_send_steps, max(0, len(chunk.actions) - send_start))
                    control_actions = chunk.actions[send_start : send_start + send_count, :ROBOT_CONTROL_DIM]
                    chunk_targets = rollout_action_chunk(
                        robot_state,
                        chunk.actions[:, :ROBOT_CONTROL_DIM],
                        normalize_rx=inferencer.normalize_rx,
                    )
                    next_states = chunk_targets[send_start : send_start + send_count]
                    last_rtc_sent_targets = np.full_like(chunk_targets, np.nan, dtype=np.float32)
                    last_rtc_sent_targets[send_start : send_start + send_count] = next_states

                    infer_ms = (time.monotonic() - t_infer) * 1000
                    status_msg = ""
                    if chunk.actions.shape[-1] > TASK_STATUS_INDEX:
                        status = np.clip(chunk.actions[:, TASK_STATUS_INDEX], 0.0, 1.0)
                        status_msg = f", task_status={status[0]:.3f}->{status[-1]:.3f}"
                    prefix_msg = f", rtc_prefix={prefix_len_used}" if prefix_len_used else ""
                    print(
                        f"[{addr}] step={step} chunk={last_rtc_chunk_id} "
                        f"query OK ({infer_ms:.1f} ms{prefix_msg}{prefix_trim_msg}{status_msg}, "
                        f"send={send_start}:{send_start + send_count})"
                    )

                    step_dir = baseline_log_dir / f"step_{step:04d}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    for key, payload in image_data.items():
                        (step_dir / f"{key}.jpg").write_bytes(payload)
                    (step_dir / "robot_state.json").write_text(json.dumps(robot_state.tolist(), indent=2))
                    (step_dir / "actions.json").write_text(json.dumps(control_actions.tolist(), indent=2))
                    (step_dir / "policy_actions.json").write_text(json.dumps(chunk.actions.tolist(), indent=2))
                    (step_dir / "model_actions.json").write_text(json.dumps(chunk.model_actions.tolist(), indent=2))
                    (step_dir / "next_states.json").write_text(json.dumps(next_states.tolist(), indent=2))

                    conn.sendall(struct.pack(">IIII", last_rtc_chunk_id, send_start, send_count, prefix_len_used))
                    if send_count > 0:
                        conn.sendall(struct.pack(f">{send_count * ROBOT_CONTROL_DIM}f", *next_states.reshape(-1)))
                    conn.sendall(struct.pack(">f", 0.0))
                    conn.sendall(struct.pack(">I", 0))
                    processed_text = b""
                    conn.sendall(struct.pack(">I", len(processed_text)))
                    conn.sendall(processed_text)
                except Exception as exc:
                    import traceback
                    print(f"[{addr}] inference error at step {step}: {exc}")
                    traceback.print_exc()
                    break
                continue

            # ============================================================
            # Baseline protocol path  (action-only / lmm-client)
            # Matches serve.py exactly.
            # ============================================================
            raw_jpegs: dict[str, bytes] = {}
            recv_ok = True
            for cam_key in ('top', 'chest', 'wrist2'):
                raw = _recv_exact(conn, 4)
                if raw is None:
                    recv_ok = False
                    break
                jpeg_len = struct.unpack('>I', raw)[0]
                jpeg_bytes = _recv_exact(conn, jpeg_len)
                if jpeg_bytes is None:
                    recv_ok = False
                    break
                raw_jpegs[cam_key] = jpeg_bytes
            if not recv_ok:
                break

            # text instruction
            raw = _recv_exact(conn, 4)
            if raw is None:
                break
            text_len = struct.unpack('>I', raw)[0]
            task_prompt = None
            if text_len > 0:
                text_data = _recv_exact(conn, text_len)
                if text_data is None:
                    break
                task_prompt = text_data.decode('utf-8')

            # robot state
            raw = _recv_exact(conn, 7 * 4)
            if raw is None:
                break
            robot_state = np.array(struct.unpack('>7f', raw), dtype=np.float32)

            # episode reset on first step
            if step == 1:
                state.reset()

            try:
                if state.needs_query():
                    t_infer = time.monotonic()
                    image_data = {
                        "top_image": raw_jpegs["top"],
                        "chest_image": raw_jpegs["chest"],
                        "wrist_image": raw_jpegs["wrist2"],
                    }
                    chunk = inferencer.infer_chunk(image_data, robot_state, task_prompt=task_prompt)
                    state.add_chunk(chunk.actions)
                    infer_ms = (time.monotonic() - t_infer) * 1000
                    status_msg = ""
                    if chunk.actions.shape[-1] > TASK_STATUS_INDEX:
                        status = np.clip(chunk.actions[:, TASK_STATUS_INDEX], 0.0, 1.0)
                        status_msg = f", task_status={status[0]:.3f}->{status[-1]:.3f}"
                    print(f'[{addr}] step={step} query OK ({infer_ms:.1f} ms{status_msg})')

                next_state = robot_state
                while not state.needs_query():
                    raw_action = state.get_action()
                    if len(raw_action) >= 8:
                        motion_action = raw_action[:7]
                        term_flag = raw_action[7]
                    else:
                        motion_action = raw_action
                        term_flag = 0.0
                    next_state = compose_pose(next_state, motion_action, normalize_rx=inferencer.normalize_rx)
                    state.step()
            except Exception as exc:
                print(f'[{addr}] inference error at step {step}: {exc}')
                import traceback
                traceback.print_exc()
                break

            # --- debug logs (serve.py style) ---
            step_dir = baseline_log_dir / f'step_{step:04d}'
            step_dir.mkdir(parents=True, exist_ok=True)
            for cam, jpeg in zip(('top', 'chest', 'wrist2'), (raw_jpegs['top'], raw_jpegs['chest'], raw_jpegs['wrist2'])):
                (step_dir / f'{cam}.jpg').write_bytes(jpeg)
            (step_dir / 'robot_state.json').write_text(
                json.dumps(robot_state.tolist(), indent=2)
            )
            term_raw = float(raw_action[7]) if len(raw_action) >= 8 else 0.0
            (step_dir / 'next_state.json').write_text(
                json.dumps(next_state.tolist() + [term_raw], indent=2)
            )
            (step_dir / 'action.json').write_text(
                json.dumps(raw_action.tolist(), indent=2)
            )
            (step_dir / 'task_prompt.json').write_text(
                json.dumps({"task_prompt": task_prompt}, indent=2)
            )

            # --- output (serve.py baseline protocol) ---
            processed_text = 'success'
            processed_text_bytes = processed_text.encode('utf-8')
            reject_flag = 0
            conn.sendall(struct.pack('>7f', *next_state))
            conn.sendall(struct.pack('>f', term_flag))
            conn.sendall(struct.pack('>I', reject_flag))
            conn.sendall(struct.pack('>I', len(processed_text_bytes)))
            conn.sendall(processed_text_bytes)

    except Exception as exc:
        print(f'[{addr}] connection error: {exc}')
    finally:
        conn.close()
        print(f'[{addr}] disconnected (steps={step})')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="YAML config path.")
    p.add_argument("--checkpoint-step", type=int, default=None,
                   help="Checkpoint step to load (default: latest).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--num-steps", type=int, default=None,
                   help="Override action chunk size (default: model.action_horizon from config).")
    p.add_argument("--task-prompt", default=None,
                   help="Override the prompt used for live inference.")
    p.add_argument("--image-protocol", choices=["legacy", "three-view"], default="three-view",
                   help="legacy: wrist+ignored-extra protocol; three-view: chest, top, wrist framed images.")
    p.add_argument("--response-protocol", choices=["action-only", "lmm-client", "lmm-rtc-chunk"], default="lmm-rtc-chunk",
                   help="action-only: return one 7D pose; lmm-client: read text and return one 7D pose; "
                        "lmm-rtc-chunk: async RTC chunk protocol for lmm_client.py.")
    p.add_argument("--rtc-send-steps", type=int, default=5,
                   help="Number of non-prefix actions to return per lmm-rtc-chunk response.")
    p.add_argument("--rtc-prefix-pos-tolerance", type=float, default=DEFAULT_RTC_PREFIX_POS_TOLERANCE,
                   help="Position tolerance in meters for dropping already-reached RTC prefix targets.")
    p.add_argument("--rtc-prefix-rot-tolerance", type=float, default=DEFAULT_RTC_PREFIX_ROT_TOLERANCE,
                   help="Euler-angle tolerance in radians for dropping already-reached RTC prefix targets.")
    p.add_argument("--temporal-agg", action="store_true",
                   help="Query every step and aggregate overlapping chunks with exp-decay weights.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.rtc_send_steps <= 0:
        raise ValueError(f"--rtc-send-steps must be positive, got {args.rtc_send_steps}")
    if args.rtc_prefix_pos_tolerance < 0:
        raise ValueError(f"--rtc-prefix-pos-tolerance must be non-negative, got {args.rtc_prefix_pos_tolerance}")
    if args.rtc_prefix_rot_tolerance < 0:
        raise ValueError(f"--rtc-prefix-rot-tolerance must be non-negative, got {args.rtc_prefix_rot_tolerance}")
    if args.response_protocol == "lmm-rtc-chunk" and args.temporal_agg:
        raise ValueError("--response-protocol lmm-rtc-chunk cannot be combined with --temporal-agg")

    inferencer = Pi05Inference(
        config_path=args.config,
        checkpoint_step=args.checkpoint_step,
        num_steps_override=args.num_steps,
        task_prompt=args.task_prompt,
    )

    mode = ("temporal-agg" if args.temporal_agg
            else f"chunk-replay (chunk_size={inferencer.chunk_size})")
    print(f"Task prompt    : {inferencer.task}")
    print(f"Inference mode : {mode}")
    print(f"Image protocol : {args.image_protocol}")
    print(f"Reply protocol : {args.response_protocol}")
    if args.response_protocol == "lmm-rtc-chunk":
        print(f"RTC send steps : {args.rtc_send_steps}")
        print(f"RTC prefix tol : {args.rtc_prefix_pos_tolerance * 1000:.1f} mm, "
              f"{args.rtc_prefix_rot_tolerance:.3f} rad")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(5)
    print(f"Listening on {args.host}:{args.port}")

    try:
        while True:
            conn, addr = server_sock.accept()
            t = threading.Thread(
                target=handle_client,
                args=(
                    conn,
                    addr,
                    inferencer,
                    args.temporal_agg,
                    args.image_protocol,
                    args.response_protocol,
                    args.rtc_send_steps,
                    args.rtc_prefix_pos_tolerance,
                    args.rtc_prefix_rot_tolerance,
                ),
                daemon=True,
            )
            t.start()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()