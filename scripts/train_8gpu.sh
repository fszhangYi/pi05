#!/usr/bin/env bash
# 8 卡全参微调 / 8 卡 LoRA 启动脚本
#
# 用法:
#   bash scripts/train_8gpu.sh configs/pi05_company_example.yaml      # 全参
#   bash scripts/train_8gpu.sh configs/pi05_lora_example.yaml         # LoRA
#
# 环境变量:
#   USE_VENV=1        使用 .venv 虚拟环境 (默认 0)
#   PYTHON_BIN=...    指定 Python 可执行文件 (默认 python3.12)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/pi05_tonglu0602_example.yaml}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
USE_VENV="${USE_VENV:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
export PYTHONPATH="/data1/pi05_jax_sft/lerobot:/data1/pi05_jax_sft/src"

# JAX 显存管理
# platform allocator avoids BFC fragmentation during FSDP init (full model replicated per GPU)
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "$USE_VENV" == "1" ]]; then
  source "$VENV_DIR/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="$PYTHON_BIN"
fi

echo "Config  : $CONFIG_PATH"
echo "Python  : $PYTHON_CMD"
echo "GPUs    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l) detected"
echo

"$PYTHON_CMD" -m pi05_jax_sft.train --config "$CONFIG_PATH"
