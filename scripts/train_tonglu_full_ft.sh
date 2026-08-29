#!/usr/bin/env bash
# Tonglu mlu_full_ft* PyTorch full fine-tune (matches hww serve-used recipes).
#
# Usage:
#   bash scripts/train_tonglu_full_ft.sh
#   bash scripts/train_tonglu_full_ft.sh configs/pi05_tonglu0630_full_ft_abs_pose.yaml
#   bash scripts/train_tonglu_full_ft.sh configs/pi05_tonglu0602_full_ft_filter.yaml
#
# Requires: ≥ training.fsdp_devices GPUs (default 8), ./checkpoints/pi05_base_pytorch/model.safetensors
# This script does NOT downscale for single-GPU AutoDL.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/pi05_tonglu0630_full_ft_two_view.yaml}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
USE_VENV="${USE_VENV:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/lerobot:${PYTHONPATH:-}"

if [[ "$USE_VENV" == "1" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="$PYTHON_BIN"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config not found: $CONFIG_PATH" >&2
  exit 1
fi

NPROC="$("$PYTHON_CMD" - <<PY
import yaml
cfg = yaml.safe_load(open("$CONFIG_PATH"))
print(int(cfg.get("training", {}).get("fsdp_devices", 8)))
PY
)"

if [[ -n "${NPROC_OVERRIDE:-}" ]]; then
  echo "WARNING: NPROC_OVERRIDE=${NPROC_OVERRIDE} (config fsdp_devices=${NPROC}). Reproduce expects ${NPROC}."
  NPROC="$NPROC_OVERRIDE"
fi

GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  # nvidia-smi may be present but non-executable (AutoDL / container); never abort here.
  GPU_COUNT="$( { nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true; } | wc -l | tr -d ' ' )"
fi
GPU_COUNT="${GPU_COUNT:-0}"

echo "Config     : $CONFIG_PATH"
echo "Python     : $PYTHON_CMD"
echo "nproc      : $NPROC (from training.fsdp_devices)"
echo "GPUs seen  : $GPU_COUNT"
echo "Backend    : PyTorch openpi train_pytorch via torchrun"
echo
echo "torchrun cmdline: torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} -m pi05_jax_sft.train_pytorch --config ${CONFIG_PATH}"
echo

if [[ "$GPU_COUNT" -lt "$NPROC" ]]; then
  echo "ERROR: need ≥${NPROC} GPUs for Tonglu full FT; found ${GPU_COUNT}." >&2
  echo "       Place base at checkpoints/pi05_base_pytorch/ and use an 8×~80GB box." >&2
  echo "       For 32GB smoke LoRA use: bash scripts/train_8gpu.sh configs/pi05_act_robot_smoke.yaml" >&2
  exit 1
fi

# Print-only / preflight on rank logic lives in the module; run torchrun.
exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  -m pi05_jax_sft.train_pytorch \
  --config "$CONFIG_PATH"
