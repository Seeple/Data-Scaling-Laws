#!/usr/bin/env bash
# Ordinary DP training, adapted from manage_table.sh / hang_chinese_knot.sh.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

TELEOP_DATASET_PATH="${TELEOP_DATASET_PATH:-${DATA_ROOT}/dataset/teleop.zarr.zip}"
NUM_EPOCHS="${NUM_EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"
MODEL_PRETRAINED="${MODEL_PRETRAINED:-true}"

recipe_launch pretrain \
    --dataset "${TELEOP_DATASET_PATH}" \
    --set task.dataset.use_ratio=1.0 \
    --set "task.dataset.val_ratio=${VAL_RATIO:-0.1}" \
    "$@"
