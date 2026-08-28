#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/pi05_company_example.yaml}"
CHECKPOINT_STEP="${2:-}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
USE_VENV="${USE_VENV:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if [[ "$USE_VENV" == "1" ]]; then
  source "$VENV_DIR/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="$PYTHON_BIN"
fi

if [[ -n "$CHECKPOINT_STEP" ]]; then
  "$PYTHON_CMD" -m pi05_jax_sft.evaluate_checkpoint --config "$CONFIG_PATH" --checkpoint-step "$CHECKPOINT_STEP" --sample-index "$SAMPLE_INDEX"
else
  "$PYTHON_CMD" -m pi05_jax_sft.evaluate_checkpoint --config "$CONFIG_PATH" --sample-index "$SAMPLE_INDEX"
fi
