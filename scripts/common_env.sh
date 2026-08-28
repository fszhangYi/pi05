#!/usr/bin/env bash
# Shared environment for pi05 scripts. Source from other bash scripts:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

if [[ -n "${PI05_COMMON_ENV_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
PI05_COMMON_ENV_LOADED=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/lerobot:$ROOT_DIR/external/openpi/src:$ROOT_DIR/external/openpi/packages/openpi-client/src:${PYTHONPATH:-}"

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
