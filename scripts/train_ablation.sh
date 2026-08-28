#!/usr/bin/env bash
# Serial ablation training: run multiple configs one after another on this machine.
#
# Usage:
#   bash scripts/train_ablation.sh
#
# Edit ALL_CONFIGS below to select which configs to train.
# Each config trains to completion before the next one starts.
#
# Environment variables:
#   CONFIG_DIR      path to ablation config directory (default: configs/ablation_se3)
#   USE_VENV=1      activate .venv before training
#   PYTHON_BIN      Python executable (default: python3.12)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${CONFIG_DIR:-$ROOT_DIR/configs/ablation_se3}"
LOG_DIR="$ROOT_DIR/logs/ablation"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
USE_VENV="${USE_VENV:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT_DIR}/external/openpi/src:${PYTHONPATH}"
export PYTHONPATH="${ROOT_DIR}/external/openpi/packages/openpi-client/src:${PYTHONPATH}"

if [[ "$USE_VENV" == "1" ]]; then
    source "$VENV_DIR/bin/activate"
    PYTHON_CMD="python"
else
    PYTHON_CMD="$PYTHON_BIN"
fi

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Configs to train — edit this list
# ---------------------------------------------------------------------------
ALL_CONFIGS=(
    "cam_100_15_se3"
    "cam_100_0_se3"
    "cam_120_15_se3"
    "cam_120_30_se3"
    "cam_20_0_se3"
    "cam_50_0_se3"
)

TOTAL=${#ALL_CONFIGS[@]}

echo "========================================="
echo "  Ablation  —  serial training"
echo "  $TOTAL configs  |  $(date)"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

SUCCESSFUL=()
FAILED=()
START_TIME=$(date +%s)

for i in "${!ALL_CONFIGS[@]}"; do
    cam_config="${ALL_CONFIGS[$i]}"
    config_file="$CONFIG_DIR/${cam_config}.yaml"
    job_num=$((i + 1))
    exp_name="wrist_only_${cam_config}"
    log_file="$LOG_DIR/${cam_config}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo "  [$job_num/$TOTAL] $cam_config"
    echo "  start: $(date)"
    echo "------------------------------------------------------------"

    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: config not found: $config_file"
        FAILED+=("$cam_config")
        continue
    fi

    # Override exp_name; restore it no matter what happens
    original_exp_name="$(grep "^  exp_name:" "$config_file" | sed 's/.*exp_name: *//')"
    sed -i "s/^  exp_name:.*/  exp_name: ${exp_name}/" "$config_file"

    set +e
    "$PYTHON_CMD" -m pi05_jax_sft.train --config "$config_file" 2>&1 | tee "$log_file"
    exit_code=${PIPESTATUS[0]}
    set -e

    sed -i "s/^  exp_name:.*/  exp_name: ${original_exp_name}/" "$config_file"

    if [[ $exit_code -eq 0 ]]; then
        echo "✓ $cam_config done"
        SUCCESSFUL+=("$cam_config")
    else
        echo "✗ $cam_config failed (exit $exit_code)"
        FAILED+=("$cam_config")
    fi

    if (( job_num < TOTAL )); then
        sleep 10
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "========================================="
echo "  Summary  |  ${HOURS}h ${MINUTES}m"
echo "========================================="
echo "Successful (${#SUCCESSFUL[@]}):"
for c in "${SUCCESSFUL[@]}"; do echo "  ✓ $c"; done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed (${#FAILED[@]}):"
    for c in "${FAILED[@]}"; do echo "  ✗ $c"; done
    echo "Logs: $LOG_DIR"
    exit 1
fi

echo ""
echo "All $TOTAL configs trained successfully."
echo "Logs: $LOG_DIR"
