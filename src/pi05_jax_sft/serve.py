#!/usr/bin/env python3
"""pi0.5 inference server.

Speaks the same TCP wire protocol as act_robot/serve.py baseline client,
so the existing robot client works without modification.

Per-frame input (big-endian):
    4B  top_len      + top JPEG
    4B  chest_len    + chest JPEG
    4B  wrist2_len   + wrist2 JPEG  -> mapped to camera "wrist"
    4B  text_len     + text UTF-8     (received but unused)
    28B robot_state  (7 x float32)

Per-frame output (big-endian):
    28B next_state   (7 x float32)
    4B  term_flag    (uint32)
    4B  reject_flag  (uint32)
    4B  text_len     + processed_text UTF-8

Episode reset happens when the client disconnects and reconnects.

Inference modes (mutually exclusive):
    default          chunk-replay: query every chunk_size steps, execute the
                     chunk in sequence (has discontinuities at chunk boundaries)
    --temporal-agg   query every step, exponentially-weighted average over
                     overlapping chunks (smoothest, blended)
    --always-first   query every step, always use chunk[0] (greedy 1-step
                     lookahead; eliminates boundary jumps, no blending)

Usage:
    python -m pi05_jax_sft.serve \\
        --config  configs/pi05_tonglu0602_example.yaml \\
        --checkpoint-dir /home/ubuntu/hww/pi05_jax_sft/artifacts/checkpoints/pi05_tonglu0602_lora/8gpu_lora \\
        --checkpoint-step 24000 \\
        --port 5000 \\
        [--task-prompt "..."] \\
        [--temporal-agg | --always-first]
"""
from __future__ import annotations

import argparse
import io
import json
import socket
import struct
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from pi05_jax_sft.pose_utils import normalize_rx_to_2pi


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
        action:       [dx, dy, dz, drx, dry, drz, gripper_cmd]  float32/64
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


# ---------------------------------------------------------------------------
# Client episode state (identical logic to act_robot/serve.py)
# ---------------------------------------------------------------------------

class ClientState:
    """Per-connection state. Three inference modes (chunk_replay / temporal_agg /
    always_first) — see module docstring."""

    def __init__(self, chunk_size: int, mode: str = 'chunk_replay', k: float = 0.01):
        assert mode in ('chunk_replay', 'temporal_agg', 'always_first')
        self.chunk_size = chunk_size
        self.mode = mode
        self.k = k
        self.t = 0

        if mode == 'temporal_agg':
            self.history: list[tuple[int, np.ndarray]] = []
        else:
            self.current_chunk: np.ndarray | None = None
            self.chunk_offset: int = chunk_size  # force re-query on first call

    def reset(self) -> None:
        self.t = 0
        if self.mode == 'temporal_agg':
            self.history = []
        else:
            self.current_chunk = None
            self.chunk_offset = self.chunk_size

    def needs_query(self) -> bool:
        if self.mode in ('temporal_agg', 'always_first'):
            return True
        return self.chunk_offset >= self.chunk_size / 2 ## add by huweiwen

    def add_chunk(self, chunk: np.ndarray) -> None:
        if self.mode == 'temporal_agg':
            self.history.append((self.t, chunk.copy()))
            self.history = [(t0, c) for t0, c in self.history
                            if t0 + self.chunk_size > self.t]
        else:
            self.current_chunk = chunk.copy()
            self.chunk_offset = 0

    def get_action(self) -> np.ndarray:
        if self.mode == 'temporal_agg':
            relevant = [
                (t0, chunk[self.t - t0])
                for t0, chunk in self.history
                if 0 <= self.t - t0 < self.chunk_size
            ]
            if not relevant:
                raise RuntimeError('No predictions cover the current timestep.')
            relevant.sort(key=lambda x: x[0])
            n = len(relevant)
            weights = np.exp(-self.k * np.arange(n - 1, -1, -1))
            weights /= weights.sum()
            action = sum(w * a for w, (_, a) in zip(weights, relevant))
            return action
        if self.current_chunk is None:
            raise RuntimeError('No chunk available.')
        if self.mode == 'always_first':
            return self.current_chunk[0]
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
        checkpoint_dir: str | None,
        num_steps_override: int | None,
        task_prompt: str | None,
        lerobot_home: str | None,
    ):
        from pi05_jax_sft.project_config import load_config
        from pi05_jax_sft.runtime import (
            build_train_config,
            configure_lerobot_home,
            ensure_openpi_importable,
            latest_checkpoint_step,
        )

        cfg = load_config(config_path)
        _home = Path(lerobot_home) if lerobot_home else cfg.resolve_path(cfg.data.hf_lerobot_home)
        try:
            configure_lerobot_home(_home)
        except PermissionError:
            fallback = Path(tempfile.gettempdir()) / "lerobot"
            print(f"Warning: cannot create {_home}, using fallback {fallback}")
            configure_lerobot_home(fallback)
        ensure_openpi_importable(cfg.project_root)

        from openpi.policies import policy_config as pc

        train_cfg = build_train_config(cfg)
        # object.__setattr__(train_cfg.model, "pytorch_compile_mode", None)
        object.__setattr__(train_cfg.model, "pytorch_compile_mode", "max-autotune-no-cudagraphs")

        if checkpoint_dir is not None:
            checkpoint_root = Path(checkpoint_dir)
        else:
            checkpoint_root = train_cfg.checkpoint_dir

        step = checkpoint_step if checkpoint_step is not None else latest_checkpoint_step(checkpoint_root)
        checkpoint_path = checkpoint_root / str(step)
        prompt = task_prompt or cfg.data.task_name

        print(f"Loading checkpoint: {checkpoint_path}")
        self._policy = pc.create_trained_policy(
            train_cfg,
            checkpoint_path,
            default_prompt=prompt,
        )

        self.chunk_size = num_steps_override if num_steps_override is not None else cfg.model.action_horizon
        self.task = prompt
        self.normalize_rx = cfg.data.normalize_rx_to_2pi
        self._lock = threading.Lock()

        # Determine active cameras from config (protocol still receives all)
        self.camera_map = [
            ("top", "top_image", "top_image_prefix"),
            ("chest", "chest_image", "chest_image_prefix"),
            ("wrist2", "wrist_image", "wrist_image_prefix"),
        ]
        self.active_image_keys = [
            model_key
            for _, model_key, prefix_field in self.camera_map
            if getattr(cfg.data, prefix_field, None) is not None
        ]
        print(f"Active camera inputs: {self.active_image_keys}")

        # Warm up JAX JIT with a dummy inference so the first real call is fast.
        print("Warming up JAX JIT compilation (this may take ~30 s) …")
        t0 = time.monotonic()
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_state = np.zeros(7, dtype=np.float32)
        dummy_images = {k: dummy_img for k in self.active_image_keys}
        self._infer_chunk_raw(dummy_images, dummy_state)
        print(f"Warm-up done in {time.monotonic() - t0:.1f} s")

    def _infer_chunk_raw(self, images: dict[str, np.ndarray], robot_state: np.ndarray, *, task_prompt: str | None = None) -> np.ndarray:
        """Run a single policy query. Returns [chunk_size, 7] float32."""
        state = robot_state.astype(np.float32)
        if self.normalize_rx:
            state = normalize_rx_to_2pi(state).astype(np.float32)
        obs = {
            **images,
            "state": state,
            "prompt": task_prompt if task_prompt is not None else self.task,
        }
        result = self._policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)  # [action_horizon, 7]
        # Clip to requested chunk_size in case action_horizon differs
        return actions[: self.chunk_size]

    def infer_chunk(self, image_bytes: dict[str, bytes], robot_state: np.ndarray, task_prompt: str | None = None) -> np.ndarray:
        """Decode JPEG and run inference. Thread-safe. Returns [chunk_size, 7]."""
        images = {
            key: np.asarray(Image.open(io.BytesIO(value)).convert("RGB"), dtype=np.uint8)
            for key, value in image_bytes.items()
        }
        with self._lock:
            return self._infer_chunk_raw(images, robot_state, task_prompt=task_prompt)


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def recv_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = b''
    while len(buf) < n:
        pkt = conn.recv(n - len(buf))
        if not pkt:
            return None
        buf += pkt
    return buf


def handle_client(
    conn: socket.socket,
    addr: tuple,
    inferencer: Pi05Inference,
    mode: str,
) -> None:
    """Baseline client handler. Protocol matches act_robot/serve.py."""
    print(f'[{addr}] connected  (pi0.5 three-view)')

    client_state = ClientState(inferencer.chunk_size, mode=mode)
    now = datetime.now()
    date_str = now.strftime('%Y%m%d')
    time_str = now.strftime('%H%M%S')
    log_dir = Path('logs') / date_str / time_str
    log_dir.mkdir(parents=True, exist_ok=True)

    step = 0

    def _ts():
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    try:
        while True:
            step += 1
            t_recv_start = _ts()
            infer_meta = None

            # --- 3 camera JPEGs: top, chest, wrist2 ---
            raw_jpegs: dict[str, bytes] = {}
            recv_ok = True
            for cam_key in ('top', 'chest', 'wrist2'):
                raw = recv_exact(conn, 4)
                if raw is None:
                    recv_ok = False
                    break
                jpeg_len = struct.unpack('>I', raw)[0]
                jpeg_bytes = recv_exact(conn, jpeg_len)
                if jpeg_bytes is None:
                    recv_ok = False
                    break
                raw_jpegs[cam_key] = jpeg_bytes
            if not recv_ok:
                print("0.. not recv_ok")
                break
            # print("receive text prompt")
            # --- text instruction (used as task prompt) ---
            raw = recv_exact(conn, 4)
            if raw is None:
                break
            text_len = struct.unpack('>I', raw)[0]
            task_prompt = None
            if text_len > 0:
                text_data = recv_exact(conn, text_len)
                if text_data is None:
                    print("text_data is None")
                    break
                task_prompt = text_data.decode('utf-8')
            
            # print("receive robot state")
            # --- robot state ---
            raw = recv_exact(conn, 7 * 4)
            if raw is None:
                print("robot data is none")
                break
            # print("receive done")
            robot_state = np.array(struct.unpack('>7f', raw), dtype=np.float32)

            # --- new connection = new episode ---
            if step == 1:
                client_state.reset()

            # print("infer.....")
            try:
                if client_state.needs_query():
                    t_infer = time.monotonic()
                    t_infer_str = _ts()
                    # Map protocol camera names to model observation keys
                    image_data = {
                        model_key: raw_jpegs[protocol_name]
                        for protocol_name, model_key, _ in inferencer.camera_map
                        if model_key in inferencer.active_image_keys
                    }
                    chunk = inferencer.infer_chunk(image_data, robot_state, task_prompt=task_prompt)
                    client_state.add_chunk(chunk)
                    t_infer_end = time.monotonic()
                    t_infer_end_str = _ts()
                    infer_ms = (t_infer_end - t_infer) * 1000
                    print(f'[{addr}] step={step} query OK ({infer_ms:.1f} ms)')
                    infer_meta = {
                        "start_ts": t_infer_str,
                        "end_ts": t_infer_end_str,
                        "duration_ms": infer_ms,
                    }

                next_state = robot_state
                while not client_state.needs_query():
                    raw_action = client_state.get_action()
                    if len(raw_action) >= 8:
                        motion_action = raw_action[:7]
                        term_flag = raw_action[7] #int(raw_action[7] > 0.95)
                    else:
                        motion_action = raw_action
                        term_flag = 0.0
                    next_state = compose_pose(next_state, motion_action, normalize_rx=inferencer.normalize_rx)
                    client_state.step()
            except Exception as exc:
                print(f'[{addr}] inference error at step {step}: {exc}')
                import traceback
                traceback.print_exc()
                break

            # --- debug logs ---
            step_dir = log_dir / f'step_{step:04d}'
            step_dir.mkdir(exist_ok=True)
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

            # --- output matching act_robot baseline protocol ---
            processed_text = 'success'
            processed_text_bytes = processed_text.encode('utf-8')
            reject_flag = 0
            conn.sendall(struct.pack('>7f', *next_state))
            conn.sendall(struct.pack('>f', term_flag))
            conn.sendall(struct.pack('>I', reject_flag))
            conn.sendall(struct.pack('>I', len(processed_text_bytes)))
            conn.sendall(processed_text_bytes)
            t_send_end = _ts()
            step_meta = {
                "recv_start_ts": t_recv_start,
                "send_end_ts": t_send_end,
            }
            if infer_meta is not None:
                step_meta["infer"] = infer_meta
            (step_dir / "meta.json").write_text(json.dumps(step_meta, indent=2))
    except Exception as exc:
        print(f'[{addr}] connection error: {exc}')
    finally:
        conn.close()
        print(f'[{addr}] disconnected  (steps={step})')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="YAML config path.")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Override checkpoint root directory (default: derived from config).")
    p.add_argument("--lerobot-home", default=None,
                   help="Override HF_LEROBOT_HOME directory (default: from config).")
    p.add_argument("--checkpoint-step", type=int, default=None,
                   help="Checkpoint step to load (default: latest).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--num-steps", type=int, default=None,
                   help="Override action chunk size (default: model.action_horizon from config).")
    p.add_argument("--task-prompt", default=None,
                   help="Override the prompt used for live inference.")
    inf_mode_group = p.add_mutually_exclusive_group()
    inf_mode_group.add_argument("--temporal-agg", action='store_true',
                                help="Query every step and aggregate overlapping chunks with exp-decay weights.")
    inf_mode_group.add_argument("--always-first", action='store_true',
                                help="Query every step, always use chunk[0] (greedy 1-step lookahead).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    inferencer = Pi05Inference(
        config_path=args.config,
        checkpoint_step=args.checkpoint_step,
        checkpoint_dir=args.checkpoint_dir,
        num_steps_override=args.num_steps,
        task_prompt=args.task_prompt,
        lerobot_home=args.lerobot_home,
    )

    if args.temporal_agg:
        mode = 'temporal_agg'
    elif args.always_first:
        mode = 'always_first'
    else:
        mode = 'chunk_replay'

    mode_label = {
        'temporal_agg': 'temporal-agg',
        'always_first': 'always-first (per-step, chunk[0] only)',
        'chunk_replay': f'chunk-replay (chunk_size={inferencer.chunk_size})',
    }[mode]
    print(f"Task prompt    : {inferencer.task}")
    print(f"Inference mode : {mode_label}")

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
                args=(conn, addr, inferencer, mode),
                daemon=True,
            )
            t.start()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()
