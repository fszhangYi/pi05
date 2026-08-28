#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "$SCRIPT_DIR/common_env.sh"

CONFIG_PATH="${1:-$ROOT_DIR/configs/pi05_act_robot_smoke.yaml}"
CHECKPOINT_STEP="${2:-}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
USE_VENV="${USE_VENV:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

pi05_activate_python

if [[ -n "$CHECKPOINT_STEP" ]]; then
  "$PYTHON_CMD" -m pi05_jax_sft.evaluate_checkpoint --config "$CONFIG_PATH" --checkpoint-step "$CHECKPOINT_STEP" --sample-index "$SAMPLE_INDEX"
else
  "$PYTHON_CMD" -m pi05_jax_sft.evaluate_checkpoint --config "$CONFIG_PATH" --sample-index "$SAMPLE_INDEX"
fi
