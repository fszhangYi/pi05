#!/usr/bin/env bash
# 启动 pi0.5 推理服务端
#
# 用法:
#   bash scripts/serve.sh configs/pi05_company_example.yaml [checkpoint_step]
#
#   # 用最新 checkpoint，端口 5000
#   bash scripts/serve.sh configs/pi05_company_example.yaml
#
#   # 用指定 step
#   bash scripts/serve.sh configs/pi05_company_example.yaml 20000
#
#   # temporal aggregation 模式
#   TEMPORAL_AGG=1 bash scripts/serve.sh configs/pi05_company_example.yaml
#
# 环境变量:
#   PORT=5000              监听端口 (默认 5000)
#   TEMPORAL_AGG=1         启用 temporal aggregation (默认 chunk-replay)
#   NUM_STEPS=10           覆盖 action chunk 大小
#   TASK_PROMPT=...        覆盖实时推理 prompt
#   IMAGE_PROTOCOL=legacy  legacy 或 three-view
#   USE_VENV=1             使用 .venv (默认 0)
#   PYTHON_BIN=python3.12
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/external/openpi/src:$ROOT_DIR/external/openpi/packages/openpi-client/src:${PYTHONPATH:-}"
export OPENPI_DATA_HOME="/home/ubuntu/hww/pi05_jax_sft/checkpoints"
CONFIG_PATH="${1:-$ROOT_DIR/configs/pi05_tonglu0602_mlu_example_terminal_abs_pose_filter.yaml}"
CHECKPOINT_STEP="${2:-10000}"
PORT="${PORT:-8000}"
TEMPORAL_AGG="${TEMPORAL_AGG:-0}"
NUM_STEPS="${NUM_STEPS:-}"
TASK_PROMPT="${TASK_PROMPT:-}"
IMAGE_PROTOCOL="${IMAGE_PROTOCOL:-three-view}"
USE_VENV="${USE_VENV:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "$USE_VENV" == "1" ]]; then
  source "$ROOT_DIR/.venv/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="$PYTHON_BIN"
fi

CMD=("$PYTHON_CMD" -m pi05_jax_sft.serve_absolute --config "$CONFIG_PATH" --port "$PORT")

[[ -n "$CHECKPOINT_STEP" ]] && CMD+=(--checkpoint-step "$CHECKPOINT_STEP")
# [[ "$TEMPORAL_AGG" == "1" ]] && CMD+=(--temporal-agg)
# [[ -n "$NUM_STEPS" ]] && CMD+=(--num-steps "$NUM_STEPS")
# [[ -n "$TASK_PROMPT" ]] && CMD+=(--task-prompt "$TASK_PROMPT")

CMD+=(--image-protocol three-view --response-protocol lmm-rtc-chunk --rtc-send-steps 8)

echo "Config  : $CONFIG_PATH"
echo "Port    : $PORT"
echo "Images  : $IMAGE_PROTOCOL"
echo "Mode    : $( [[ "$TEMPORAL_AGG" == "1" ]] && echo temporal-agg || echo chunk-replay )"
echo

"${CMD[@]}"
