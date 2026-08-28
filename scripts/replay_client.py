#!/usr/bin/env python3
"""Replay client: reads saved robot logs and sends frames to a serve.py server.

Usage:
    python scripts/replay_client.py
    python scripts/replay_client.py --log-dir logs/20260616/144703 --port 6000
    python scripts/replay_client.py --text "pick red cube"
"""
import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path


def send_all(sock: socket.socket, data: bytes) -> None:
    """Send all bytes on a blocking socket."""
    total = 0
    while total < len(data):
        sent = sock.send(data[total:])
        if sent == 0:
            raise ConnectionError("Socket closed while sending")
        total += sent


def recv_all(sock: socket.socket, n: int) -> bytes:
    """Exactly receive n bytes from a blocking socket."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"Socket closed while receiving (got {len(buf)}/{n})")
        buf += chunk
    return buf


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay robot logs to pi0.5 inference server")
    parser.add_argument("--log-dir", type=Path, default=Path("/home/ubuntu/hww/pi05_jax_sft/logs/20260616/144703"),
                        help="Directory containing step_XXXX subdirectories")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=6000, help="Server port")
    parser.add_argument("--text", default="", help="Task prompt text sent each frame (default: empty)")
    parser.add_argument("--fps", type=float, default=15.0, help="Playback frame rate")
    parser.add_argument("--max-steps", type=int, default=0, help="Max frames to send (0 = all)")
    args = parser.parse_args()

    if not args.log_dir.exists():
        print(f"Error: log directory not found: {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    # Sort step directories: step_0001, step_0002, ...
    step_dirs = sorted(args.log_dir.glob("step_*"))
    if not step_dirs:
        print(f"Error: no step_* directories in {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    if args.max_steps > 0:
        step_dirs = step_dirs[:args.max_steps]

    text_bytes = args.text.encode("utf-8")
    frame_delay = 1.0 / args.fps if args.fps > 0 else 0.0

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to {args.host}:{args.port} ...")
    sock.connect((args.host, args.port))
    print("Connected.\n")

    try:
        for step_dir in step_dirs:
            step_name = step_dir.name

            # --- read source images ---
            top_jpg = (step_dir / "top.jpg").read_bytes()
            chest_jpg = (step_dir / "chest.jpg").read_bytes()
            wrist_jpg = (step_dir / "wrist2.jpg").read_bytes()

            # --- read robot state ---
            robot_state = json.loads((step_dir / "robot_state.json").read_text())
            if len(robot_state) != 7:
                print(f"[{step_name}] warning: robot_state length {len(robot_state)}, expected 7")
            state_bytes = struct.pack(">7f", *robot_state)

            # --- send one frame ---
            send_all(sock, struct.pack(">I", len(top_jpg)))
            send_all(sock, top_jpg)
            send_all(sock, struct.pack(">I", len(chest_jpg)))
            send_all(sock, chest_jpg)
            send_all(sock, struct.pack(">I", len(wrist_jpg)))
            send_all(sock, wrist_jpg)
            send_all(sock, struct.pack(">I", len(text_bytes)))
            send_all(sock, text_bytes)
            send_all(sock, state_bytes)

            # --- receive response ---
            next_state = struct.unpack(">7f", recv_all(sock, 28))
            term_flag = struct.unpack(">I", recv_all(sock, 4))[0]
            reject_flag = struct.unpack(">I", recv_all(sock, 4))[0]
            resp_text_len = struct.unpack(">I", recv_all(sock, 4))[0]
            resp_text = recv_all(sock, resp_text_len).decode("utf-8") if resp_text_len > 0 else ""

            print(
                f"{step_name}  |  "
                f"next=({next_state[0]:+.4f},{next_state[1]:+.4f},{next_state[2]:+.4f})  "
                f"term={term_flag}  reject={reject_flag}  text={resp_text!r}"
            )

            if frame_delay > 0:
                time.sleep(frame_delay)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"\nError: {exc}")
    finally:
        sock.close()
        print("Disconnected.")


if __name__ == "__main__":
    main()
