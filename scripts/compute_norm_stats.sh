#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "$SCRIPT_DIR/common_env.sh"

CONFIG_PATH="${1:-$ROOT_DIR/configs/pi05_act_robot_smoke.yaml}"
USE_VENV="${USE_VENV:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

pi05_activate_python

"$PYTHON_CMD" -m pi05_jax_sft.compute_norm_stats --config "$CONFIG_PATH"
