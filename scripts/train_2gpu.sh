#!/usr/bin/env bash
# 2 卡训练启动脚本 (全参或 LoRA)
#
# 用法:
#   bash scripts/train_2gpu.sh configs/pi05_company_example.yaml      # 全参
#   bash scripts/train_2gpu.sh configs/pi05_lora_example.yaml         # LoRA
#
# 注意: 2 卡时需要将 config 中的超参调整为:
#   fsdp_devices: 2
#   batch_size: 32       (保持 per_gpu=16)
#   num_train_steps: 40000   (全参) / 60000 (LoRA)  保持与 8 卡相近的 epoch 数
#
# 本脚本会在启动前校验 fsdp_devices <= 可见 GPU 数, 避免 JAX 初始化失败.
#
# 环境变量:
#   CUDA_VISIBLE_DEVICES=0,1   指定使用哪两张卡 (默认不限制)
#   USE_VENV=1                 使用 .venv (默认 0)
#   PYTHON_BIN=...             Python 可执行文件 (默认 python3.12)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/pi05_company_example.yaml}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
USE_VENV="${USE_VENV:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

# Use platform (cudaMalloc) allocator instead of BFC to avoid fragmentation during
# FSDP init, which briefly replicates the full model on every GPU simultaneously.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "$USE_VENV" == "1" ]]; then
  source "$VENV_DIR/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="$PYTHON_BIN"
fi

# ---- 读取 config 中的 fsdp_devices 并校验 ----
FSDP_DEVICES=$("$PYTHON_CMD" -c "
import yaml, sys
cfg = yaml.safe_load(open('$CONFIG_PATH'))
print(cfg.get('training', {}).get('fsdp_devices', 8))
")
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

echo "Config       : $CONFIG_PATH"
echo "fsdp_devices : $FSDP_DEVICES"
echo "Visible GPUs : $GPU_COUNT"

if (( FSDP_DEVICES > GPU_COUNT )); then
  echo "ERROR: config fsdp_devices=$FSDP_DEVICES > available GPUs=$GPU_COUNT"
  echo "       请将 config 的 fsdp_devices 改为 $GPU_COUNT 以下，或增加可见 GPU"
  exit 1
fi
echo

"$PYTHON_CMD" -m pi05_jax_sft.train --config "$CONFIG_PATH"
