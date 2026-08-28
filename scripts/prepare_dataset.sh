#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/pi05_company_example.yaml}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
USE_VENV="${USE_VENV:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
OVERWRITE_FLAG="${OVERWRITE_FLAG:---overwrite}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
IMAGE_WRITER_THREADS="${IMAGE_WRITER_THREADS:-4}"
IMAGE_WRITER_PROCESSES="${IMAGE_WRITER_PROCESSES:-2}"
DECODE_WORKERS="${DECODE_WORKERS:-0}"
DECODE_PREFETCH="${DECODE_PREFETCH:-4}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-10}"
EPISODE_START_INDEX="${EPISODE_START_INDEX:-0}"
EPISODE_END_INDEX="${EPISODE_END_INDEX:-}"
VERBOSE_EPISODES="${VERBOSE_EPISODES:-0}"
VALIDATE_IMAGES_ONLY="${VALIDATE_IMAGES_ONLY:-0}"
REPAIR_RESUME="${REPAIR_RESUME:-0}"
COPY_ORIGINAL_IMAGES="${COPY_ORIGINAL_IMAGES:-1}"
IMAGE_LINK_MODE="${IMAGE_LINK_MODE:-hardlink}"

if [[ "$USE_VENV" == "1" ]]; then
  source "$VENV_DIR/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="$PYTHON_BIN"
fi

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/lerobot:${PYTHONPATH:-}"

CMD=(
  "$PYTHON_CMD" -m pi05_jax_sft.convert_company_dataset
  --config "$CONFIG_PATH"
  --image-writer-threads "$IMAGE_WRITER_THREADS"
  --image-writer-processes "$IMAGE_WRITER_PROCESSES"
  --decode-workers "$DECODE_WORKERS"
  --decode-prefetch "$DECODE_PREFETCH"
  --progress-interval "$PROGRESS_INTERVAL"
  --episode-start-index "$EPISODE_START_INDEX"
)
[[ -n "$EPISODE_END_INDEX" ]] && CMD+=(--episode-end-index "$EPISODE_END_INDEX")
[[ "$VERBOSE_EPISODES" == "1" ]] && CMD+=(--verbose-episodes)
[[ "$VALIDATE_IMAGES_ONLY" == "1" ]] && CMD+=(--validate-images-only)
[[ "$REPAIR_RESUME" == "1" ]] && CMD+=(--repair-resume)
[[ "$COPY_ORIGINAL_IMAGES" == "1" ]] && CMD+=(--copy-original-images --image-link-mode "$IMAGE_LINK_MODE")
[[ "$RESUME" == "1" ]] && CMD+=(--resume)
[[ "$DRY_RUN" == "1" ]] && CMD+=(--dry-run)
[[ "$DRY_RUN" != "1" && "$RESUME" != "1" ]] && CMD+=($OVERWRITE_FLAG)

"${CMD[@]}"

