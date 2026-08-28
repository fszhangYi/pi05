#!/usr/bin/env bash
# Usage:
#   USE_VENV=0 PYTHON_BIN=python3.12 bash scripts/setup_env.sh
#   USE_VENV=1 PYTHON_BIN=python3.12 bash scripts/setup_env.sh  # creates .venv
#
# To point at an internal pip mirror, export PIP_INDEX_URL and PIP_TRUSTED_HOST
# before running this script.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
USE_VENV="${USE_VENV:-0}"
OPENPI_DIR="${OPENPI_DIR:-$ROOT_DIR/external/openpi}"
LEROBOT_SRC_DIR="${LEROBOT_SRC_DIR:-$ROOT_DIR/third_party/lerobot}"
LEROBOT_PACKAGE="${LEROBOT_PACKAGE:-lerobot}"

if [[ ! -d "$OPENPI_DIR/src/openpi" ]]; then
  echo "ERROR: openpi source not found at $OPENPI_DIR"
  exit 1
fi

if [[ "$USE_VENV" == "1" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  PYTHON_CMD="python"
else
  echo "USE_VENV=0: installing into current Python environment ($PYTHON_BIN)"
  PYTHON_CMD="$PYTHON_BIN"
fi

"$PYTHON_CMD" -m pip install --upgrade pip setuptools wheel hatchling
"$PYTHON_CMD" -m pip install -r "$ROOT_DIR/requirements/openpi_jax.txt"
"$PYTHON_CMD" -m pip install "$OPENPI_DIR/packages/openpi-client"

if [[ -d "$LEROBOT_SRC_DIR" ]]; then
  "$PYTHON_CMD" -m pip install --no-deps "$LEROBOT_SRC_DIR"
else
  "$PYTHON_CMD" -m pip install "$LEROBOT_PACKAGE"
fi

"$PYTHON_CMD" -m pip install --no-deps "$OPENPI_DIR"
"$PYTHON_CMD" -m pip install --no-deps "$ROOT_DIR"

echo
echo "Environment ready."
[[ "$USE_VENV" == "1" ]] && echo "Activate with: source $VENV_DIR/bin/activate"
