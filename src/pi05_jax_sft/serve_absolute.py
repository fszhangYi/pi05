#!/usr/bin/env python3
"""pi0.5 absolute-pose inference server.

This server uses the same TCP protocols as ``pi05_jax_sft.serve``, but treats
policy action dimensions 0..6 as absolute robot targets:

    [x, y, z, rx, ry, rz, gripper]

No SE(3) composition or action integration is performed. Use this entry point
only with a checkpoint trained on absolute Cartesian pose actions.

RTC logs are written under ``logs/YYYY-MM-DD/HHMMSS/step_NNNN``.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from pi05_jax_sft.pose_utils import normalize_rx_to_2pi
from pi05_jax_sft.serve_rtc import (
    DEFAULT_RTC_PREFIX_POS_TOLERANCE,
    DEFAULT_RTC_PREFIX_ROT_TOLERANCE,
    ROBOT_CONTROL_DIM,
    TASK_STATUS_INDEX,
    ActionChunk,
    ClientState,
    Pi05Inference,
    _recv_exact,
    _recv_framed_image,
    _task_status,
    _trim_reached_rtc_prefix,
)


def _absolute_targets(actions: np.ndarray, *, normalize_rx: bool) -> np.ndarray:
    targets = np.asarray(actions[:, :ROBOT_CONTROL_DIM], dtype=np.float32).copy()
    if normalize_rx and len(targets) > 0:
        targets[:, :6] = normalize_rx_to_2pi(targets[:, :6]).astype(np.float32)
    return targets


def _write_json(path: Path, value: np.ndarray | dict) -> None:
    payload = value.tolist() if isinstance(value, np.ndarray) else value
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _new_log_dir(root: Path) -> Path:
    now = datetime.now()
    date_dir = root / now.strftime("%Y-%m-%d")
    base = date_dir / now.strftime("%H%M%S")
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = date_dir / f"{base.name}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _save_step(
    log_dir: Path,
    step: int,
    image_data: dict[str, bytes],
    robot_state: np.ndarray,
    actions: np.ndarray,
    policy_actions: np.ndarray,
    model_actions: np.ndarray,
    next_states: np.ndarray,
    metadata: dict,
) -> None:
    step_dir = log_dir / f"step_{step:04d}"
    step_dir.mkdir(exist_ok=True)
    for key, payload in image_data.items():
        (step_dir / f"{key}.jpg").write_bytes(payload)
    _write_json(step_dir / "robot_state.json", robot_state)
    _write_json(step_dir / "actions.json", actions)
    _write_json(step_dir / "policy_actions.json", policy_actions)
    _write_json(step_dir / "model_actions.json", model_actions)
    _write_json(step_dir / "next_states.json", next_states)
    _write_json(step_dir / "metadata.json", metadata)


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
    log_root: Path,
) -> None:
    print(f"[{addr}] connected")
    state = ClientState(inferencer.chunk_size, temporal_agg)
    last_rtc_chunk: ActionChunk | None = None
    last_rtc_sent_targets: np.ndarray | None = None
    last_rtc_chunk_id = 0
    log_dir = _new_log_dir(log_root)
    print(f"[{addr}] log directory: {log_dir}")
    step = 0

    try:
        while True:
            step += 1

            if image_protocol == "three-view":
                image_data: dict[str, bytes] = {}
                for key in ("chest_image", "top_image", "wrist_image"):
                    payload = _recv_framed_image(conn)
                    if payload is None:
                        break
                    image_data[key] = payload
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
            if response_protocol in ("lmm-client", "lmm-rtc-chunk"):
                raw = _recv_exact(conn, 4)
                if raw is None:
                    break
                text_len = struct.unpack(">I", raw)[0]
                raw = _recv_exact(conn, text_len)
                if raw is None:
                    break
                request_text = raw.decode("utf-8")

            raw = _recv_exact(conn, ROBOT_CONTROL_DIM * 4)
            if raw is None:
                break
            robot_state = np.asarray(struct.unpack(">7f", raw), dtype=np.float32)

            raw = _recv_exact(conn, 4)
            if raw is None:
                break
            refresh = struct.unpack(">I", raw)[0]
            if refresh:
                state.reset()
                last_rtc_chunk = None
                last_rtc_sent_targets = None
                last_rtc_chunk_id = 0
                print(f"[{addr}] episode reset at step {step}")

            rtc_prefix_chunk_id = 0
            rtc_prefix_start = 0
            rtc_prefix_len = 0
            if response_protocol == "lmm-rtc-chunk":
                raw = _recv_exact(conn, 3 * 4)
                if raw is None:
                    break
                rtc_prefix_chunk_id, rtc_prefix_start, rtc_prefix_len = struct.unpack(">III", raw)

            try:
                if response_protocol == "lmm-rtc-chunk":
                    action_prefix = None
                    prefix_len_used = 0
                    prefix_skip = 0
                    reached_idx: int | None = None
                    reached_pos: float | None = None
                    reached_rot: float | None = None
                    if (
                        not refresh
                        and last_rtc_chunk is not None
                        and rtc_prefix_chunk_id == last_rtc_chunk_id
                        and rtc_prefix_len > 0
                    ):
                        prefix_start = max(
                            0,
                            min(int(rtc_prefix_start), len(last_rtc_chunk.model_actions)),
                        )
                        prefix_end = max(
                            prefix_start,
                            min(
                                prefix_start + int(rtc_prefix_len),
                                len(last_rtc_chunk.model_actions),
                            ),
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
                        prefix_skip = prefix_start - original_prefix_start
                        action_prefix = last_rtc_chunk.model_actions[prefix_start:prefix_end].copy()
                        prefix_len_used = len(action_prefix)

                    t_infer = time.monotonic()
                    chunk = inferencer.infer_chunk(
                        image_data,
                        robot_state,
                        action_prefix=action_prefix,
                        task_prompt=request_text or None,
                    )
                    infer_ms = (time.monotonic() - t_infer) * 1000
                    last_rtc_chunk_id += 1
                    last_rtc_chunk = chunk

                    send_start = prefix_len_used
                    send_count = min(rtc_send_steps, max(0, len(chunk.actions) - send_start))
                    next_states = _absolute_targets(
                        chunk.actions[send_start : send_start + send_count],
                        normalize_rx=inferencer.normalize_rx,
                    )
                    control_actions = next_states.copy()
                    last_rtc_sent_targets = np.full(
                        (len(chunk.actions), ROBOT_CONTROL_DIM),
                        np.nan,
                        dtype=np.float32,
                    )
                    last_rtc_sent_targets[send_start : send_start + send_count] = next_states

                    trim_msg = ""
                    if reached_idx is not None and reached_pos is not None and reached_rot is not None:
                        trim_msg = (
                            f", rtc_prefix_skip={prefix_skip}@{reached_idx}"
                            f"({reached_pos * 1000:.1f}mm/{reached_rot:.3f}rad)"
                        )
                    prefix_msg = f", rtc_prefix={prefix_len_used}" if prefix_len_used else ""
                    print(
                        f"[{addr}] step={step} chunk={last_rtc_chunk_id} "
                        f"query OK ({infer_ms:.1f} ms{prefix_msg}{trim_msg}, "
                        f"send={send_start}:{send_start + send_count})"
                    )

                    _save_step(
                        log_dir,
                        step,
                        image_data,
                        robot_state,
                        control_actions,
                        chunk.actions,
                        chunk.model_actions,
                        next_states,
                        {
                            "action_representation": "absolute_pose",
                            "chunk_id": last_rtc_chunk_id,
                            "send_start": send_start,
                            "send_count": send_count,
                            "prefix_len_requested": int(rtc_prefix_len),
                            "prefix_len_used": prefix_len_used,
                            "prefix_skip": prefix_skip,
                            "infer_ms": infer_ms,
                            "refresh": int(refresh),
                            "request_text": request_text,
                        },
                    )

                    conn.sendall(
                        struct.pack(
                            ">IIII",
                            last_rtc_chunk_id,
                            send_start,
                            send_count,
                            prefix_len_used,
                        )
                    )
                    if send_count > 0:
                        conn.sendall(
                            struct.pack(
                                f">{send_count * ROBOT_CONTROL_DIM}f",
                                *next_states.reshape(-1),
                            )
                        )
                    conn.sendall(struct.pack(">I", 0))
                    conn.sendall(struct.pack(">I", 0))
                    conn.sendall(struct.pack(">I", 0))
                    continue

                if state.needs_query():
                    t_infer = time.monotonic()
                    chunk = inferencer.infer_chunk(
                        image_data,
                        robot_state,
                        task_prompt=request_text or None,
                    )
                    state.add_chunk(chunk.actions)
                    infer_ms = (time.monotonic() - t_infer) * 1000
                    print(f"[{addr}] step={step} query OK ({infer_ms:.1f} ms)")

                action = state.get_action()
                next_state = _absolute_targets(
                    action[None, :],
                    normalize_rx=inferencer.normalize_rx,
                )[0]
                task_status = _task_status(action)
                state.step()

                _save_step(
                    log_dir,
                    step,
                    image_data,
                    robot_state,
                    next_state[None, :],
                    action[None, :],
                    action[None, :],
                    next_state[None, :],
                    {
                        "action_representation": "absolute_pose",
                        "task_status": task_status,
                        "refresh": int(refresh),
                        "request_text": request_text,
                    },
                )

                conn.sendall(struct.pack(">7f", *next_state))
                if response_protocol == "lmm-client":
                    conn.sendall(struct.pack(">I", 0))
                    conn.sendall(struct.pack(">I", 0))
                    conn.sendall(struct.pack(">I", 0))
            except Exception as exc:
                import traceback

                print(f"[{addr}] inference error at step {step}: {exc}")
                traceback.print_exc()
                break
    except Exception as exc:
        print(f"[{addr}] connection error: {exc}")
    finally:
        conn.close()
        print(f"[{addr}] disconnected (steps={step})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--task-prompt", default=None)
    parser.add_argument("--image-protocol", choices=["legacy", "three-view"], default="legacy")
    parser.add_argument(
        "--response-protocol",
        choices=["action-only", "lmm-client", "lmm-rtc-chunk"],
        default="action-only",
    )
    parser.add_argument("--rtc-send-steps", type=int, default=5)
    parser.add_argument(
        "--rtc-prefix-pos-tolerance",
        type=float,
        default=DEFAULT_RTC_PREFIX_POS_TOLERANCE,
    )
    parser.add_argument(
        "--rtc-prefix-rot-tolerance",
        type=float,
        default=DEFAULT_RTC_PREFIX_ROT_TOLERANCE,
    )
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--temporal-agg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rtc_send_steps <= 0:
        raise ValueError(f"--rtc-send-steps must be positive, got {args.rtc_send_steps}")
    if args.rtc_prefix_pos_tolerance < 0 or args.rtc_prefix_rot_tolerance < 0:
        raise ValueError("RTC prefix tolerances must be non-negative")
    if args.response_protocol == "lmm-rtc-chunk" and args.temporal_agg:
        raise ValueError("--response-protocol lmm-rtc-chunk cannot be combined with --temporal-agg")

    inferencer = Pi05Inference(
        config_path=args.config,
        checkpoint_step=args.checkpoint_step,
        num_steps_override=args.num_steps,
        task_prompt=args.task_prompt,
    )

    print("Action format   : absolute pose [x,y,z,rx,ry,rz,gripper]")
    print(f"Task prompt     : {inferencer.task}")
    print(f"Image protocol  : {args.image_protocol}")
    print(f"Reply protocol  : {args.response_protocol}")
    print(f"Log root        : {args.log_root}")
    if args.response_protocol == "lmm-rtc-chunk":
        print(f"RTC send steps  : {args.rtc_send_steps}")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(5)
    print(f"Listening on {args.host}:{args.port}")

    try:
        while True:
            conn, addr = server_sock.accept()
            thread = threading.Thread(
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
                    args.log_root,
                ),
                daemon=True,
            )
            thread.start()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()

