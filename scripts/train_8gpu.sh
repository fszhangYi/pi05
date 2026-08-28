#!/usr/bin/env bash
# 8 卡全参微调 / 8 卡 LoRA 启动脚本
#
# 用法:
#   bash scripts/train_8gpu.sh configs/pi05_act_robot_smoke.yaml
#
# 环境变量:
#   USE_VENV=1        使用 .venv 虚拟环境 (默认 0)
#   PYTHON_BIN=...    指定 Python 可执行文件 (默认 python3.12)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common_env.sh
source "$SCRIPT_DIR/common_env.sh"

CONFIG_PATH="${1:-$ROOT_DIR/configs/pi05_act_robot_smoke.yaml}"
USE_VENV="${USE_VENV:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

pi05_activate_python

FSDP_DEVICES=$("$PYTHON_CMD" -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG_PATH'))
print(cfg.get('training', {}).get('fsdp_devices', 8))
")
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

echo "Config       : $CONFIG_PATH"
echo "Python       : $PYTHON_CMD"
echo "fsdp_devices : $FSDP_DEVICES"
echo "Visible GPUs : $GPU_COUNT"
echo "PYTHONPATH   : $PYTHONPATH"
echo

if (( FSDP_DEVICES > GPU_COUNT )); then
  echo "ERROR: config fsdp_devices=$FSDP_DEVICES > available GPUs=$GPU_COUNT"
  exit 1
fi

"$PYTHON_CMD" -m pi05_jax_sft.train --config "$CONFIG_PATH"
