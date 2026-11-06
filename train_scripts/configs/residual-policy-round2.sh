#!/usr/bin/env bash
# Later-round full-chunk replacement residual: replay + previous residual weights.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

export RESIDUAL_DATASET_PATH="${RESIDUAL_DATASET_PATH:-${DATA_ROOT}/dataset/residual-round2.zarr.zip}"
export REPLAY_RESIDUAL_DATASET_PATH="${REPLAY_RESIDUAL_DATASET_PATH:-${DATA_ROOT}/dataset/residual-round1.zarr.zip}"
export INIT_RESIDUAL_CKPT="${INIT_RESIDUAL_CKPT:-${DATA_ROOT}/ckpts/residual-round1.ckpt}"
export INIT_EMA_MIN_DECAY="${INIT_EMA_MIN_DECAY:-0.99}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29512}"

exec bash "${RECIPE_DIR}/residual-policy.sh" "$@"
