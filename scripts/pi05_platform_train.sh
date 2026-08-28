#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/pi05_company_example.yaml}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
USE_VENV="${USE_VENV:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if [[ -z "${DL_HOSTS_LIST:-}" ]]; then
  GPU_COUNT="$(nvidia-smi --list-gpus | wc -l)"
  HOST_LIST="127.0.0.1:${GPU_COUNT}"
else
  HOST_LIST="${DL_HOSTS_LIST}"
fi

MACHINE_NODE="$(echo "${HOST_LIST}" | awk -F' ' '{print NF}')"
if [[ "${MACHINE_NODE}" -gt 1 ]]; then
  echo "Official openpi JAX training does not support multi-node training."
  echo "Detected DL_HOSTS_LIST=${HOST_LIST}"
  exit 1
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "$USE_VENV" == "1" ]]; then
  source "$VENV_DIR/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="$PYTHON_BIN"
fi

echo "Running single-node JAX training"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "DL_HOSTS_LIST=${HOST_LIST}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"

"$PYTHON_CMD" -m pi05_jax_sft.train --config "$CONFIG_PATH"
