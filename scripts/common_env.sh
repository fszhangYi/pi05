#!/usr/bin/env bash
# Shared environment for pi05 scripts. Source from other bash scripts:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

if [[ -n "${PI05_COMMON_ENV_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
PI05_COMMON_ENV_LOADED=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/lerobot:$ROOT_DIR/external/openpi/src:$ROOT_DIR/external/openpi/packages/openpi-client/src:${PYTHONPATH:-}"

# Keep HuggingFace / pip / tempfile off the AutoDL system overlay (~30G).
# Prefer an explicit data-disk root when present; otherwise use the project tree
# (which should already live on the large disk, e.g. ~/autodl-tmp/pi05).
_PI05_DATA_DISK="${PI05_DATA_DISK:-/root/autodl-tmp}"
if [[ -d "$_PI05_DATA_DISK" && -w "$_PI05_DATA_DISK" ]]; then
  _PI05_CACHE_ROOT="${PI05_CACHE_ROOT:-$_PI05_DATA_DISK/pi05-cache}"
else
  _PI05_CACHE_ROOT="${PI05_CACHE_ROOT:-$ROOT_DIR/.cache}"
fi
unset _PI05_DATA_DISK

mkdir -p \
  "$_PI05_CACHE_ROOT/hf/datasets" \
  "$_PI05_CACHE_ROOT/tmp" \
  "$_PI05_CACHE_ROOT/pip"

export HF_HOME="${HF_HOME:-$_PI05_CACHE_ROOT/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TMPDIR="${TMPDIR:-$_PI05_CACHE_ROOT/tmp}"
export TEMP="${TEMP:-$TMPDIR}"
export TMP="${TMP:-$TMPDIR}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$_PI05_CACHE_ROOT/pip}"
# openpi tokenizer download helper also respects this for local caches
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$_PI05_CACHE_ROOT/openpi}"
mkdir -p "$OPENPI_DATA_HOME"

unset _PI05_CACHE_ROOT

pi05_activate_python() {
  local venv_dir="${VENV_DIR:-$ROOT_DIR/.venv}"
  local use_venv="${USE_VENV:-0}"
  local python_bin="${PYTHON_BIN:-python3.12}"

  if [[ "$use_venv" == "1" ]]; then
    # shellcheck disable=SC1090
    source "$venv_dir/bin/activate"
    PYTHON_CMD="python"
  else
    PYTHON_CMD="$python_bin"
  fi
  export PYTHON_CMD
}
