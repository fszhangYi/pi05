#!/usr/bin/env bash
# Serial Training Script for All 6 Ablation Experiments
# Each config trains sequentially with 10s sleep between
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$ROOT_DIR/configs/ablation_se3"

# Memory settings for 8x A100
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Ensure all 8 GPUs are visible
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# List of camera configurations
declare -a CAMERA_CONFIGS=(
    "cam_100_15_se3"
    "cam_100_0_se3"
    "cam_120_15_se3"
    "cam_120_30_se3"
    "cam_20_0_se3"
    "cam_50_0_se3"
)

echo "========================================="
echo "  Ablation Experiments - Serial Training"
echo "  6 Camera Configurations"
echo "========================================="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,index,memory.total,memory.free --format=csv || true
echo ""

# Setup environment
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export PYTHONPATH="${ROOT_DIR}/external/openpi/src:${PYTHONPATH}"
export PYTHONPATH="${ROOT_DIR}/external/openpi/packages/openpi-client/src:${PYTHONPATH}"

echo ""
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Track stats
declare -a SUCCESSFUL=()
declare -a FAILED=()
START_TIME=$(date +%s)

# Train each config sequentially
for i in "${!CAMERA_CONFIGS[@]}"; do
    cam_config="${CAMERA_CONFIGS[$i]}"
    config_file="$CONFIG_DIR/${cam_config}.yaml"
    config_num=$((i + 1))
    total_configs=${#CAMERA_CONFIGS[@]}

    echo ""
    echo "============================================================"
    echo "  [$config_num/$total_configs] Training: $cam_config"
    echo "  Config: $config_file"
    echo "  Start time: $(date)"
    echo "============================================================"
    echo ""

    if [[ ! -f "$config_file" ]]; then
        echo "Error: Config file not found: $config_file"
        FAILED+=("$cam_config")
        continue
    fi

    # Save original exp_name and set unique exp_name for this camera
    original_exp_name=$(grep "^  exp_name:" "$config_file" | sed 's/.*exp_name: *//')
    new_exp_name="wrist_only_${cam_config}"

    echo "Setting exp_name to: $new_exp_name (original: $original_exp_name)"
    sed -i "s/^  exp_name:.*/  exp_name: ${new_exp_name}/" "$config_file"

    # Run training
    if python3 -m pi05_jax_sft.train --config "$config_file" 2>&1 | tee "$ROOT_DIR/logs/${cam_config}_train.log"; then
        echo ""
        echo "Restoring original exp_name: $original_exp_name"
        sed -i "s/^  exp_name:.*/  exp_name: ${original_exp_name}/" "$config_file"
        echo ""
        echo "✓ Training completed: $cam_config"
        SUCCESSFUL+=("$cam_config")
    else
        echo ""
        echo "Restoring original exp_name: $original_exp_name"
        sed -i "s/^  exp_name:.*/  exp_name: ${original_exp_name}/" "$config_file"
        echo ""
        echo "✗ Training failed: $cam_config"
        FAILED+=("$cam_config")
    fi

      # Sleep 10s before next training (except after the last one)
    if [[ $config_num -lt $total_configs ]]; then
        echo ""
        echo "Sleeping 10 seconds before next training..."
        sleep 10
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

# Summary
echo ""
echo "========================================="
echo "  Training Summary"
echo "========================================="
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Successful (${#SUCCESSFUL[@]}):"
for cam in "${SUCCESSFUL[@]}"; do
    echo "  ✓ $cam"
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "Failed (${#FAILED[@]}):"
    for cam in "${FAILED[@]}"; do
        echo "  ✗ $cam"
    done
    exit 1
else
    echo ""
    echo "All experiments trained successfully!"
    echo ""
    echo "Checkpoints: $ROOT_DIR/artifacts/ablation_se3/checkpoints/ablation_sim_se3/"
    echo ""
    for cam in "${CAMERA_CONFIGS[@]}"; do
        echo "  - wrist_only_${cam}/"
    done
fi