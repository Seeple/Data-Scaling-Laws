#!/usr/bin/env bash
# Human-gated DAgger; defaults follow the original pipeline.sh switches.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

TELEOP_DATASET_PATH="${TELEOP_DATASET_PATH:-${DATA_ROOT}/dataset/teleop.zarr.zip}"
HITL_DATASET_PATH="${HITL_DATASET_PATH:-${DATA_ROOT}/dataset/hitl.zarr.zip}"
FINETUNE_CKPT="${FINETUNE_CKPT:-${DATA_ROOT}/ckpts/base.ckpt}"
NUM_EPOCHS="${NUM_EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29502}"
MODEL_PRETRAINED="${MODEL_PRETRAINED:-false}"

online_args=()
if ! recipe_has_option --online-dataset "$@"; then
    online_args+=(--online-dataset "${HITL_DATASET_PATH}")
fi
ratio_args=()
if [[ -n "${RLPD_RATIO:-}" ]]; then
    ratio_args+=(--set "task.dataset.rlpd_ratio=${RLPD_RATIO}")
fi

recipe_launch dagger \
    --teleop-dataset "${TELEOP_DATASET_PATH}" \
    "${online_args[@]}" \
    --checkpoint "${FINETUNE_CKPT}" \
    "${ratio_args[@]}" \
    --set "task.dataset.hitl_only_tag=${HITL_ONLY_TAG:-true}" \
    --set "task.dataset.hitl_action_mask=${HITL_ACTION_MASK:-true}" \
    --set "task.dataset.hitl_contiguous_action_mask=${HITL_CONTIGUOUS_ACTION_MASK:-false}" \
    --set "task.dataset.hitl_invalid_tail_padding=${HITL_INVALID_TAIL_PADDING:-false}" \
    --set "task.dataset.hitl_disable_downsample=${HITL_DISABLE_DOWNSAMPLE:-true}" \
    --set "task.dataset.hitl_downsample_multiplier=${HITL_DOWNSAMPLE_MULTIPLIER:-2}" \
    --set "task.dataset.lowdim_obs_normalizer_source=${LOWDIM_OBS_NORMALIZER_SOURCE:-mixed}" \
    --set "training.freeze_encoder_epochs=${FREEZE_ENCODER_EPOCHS:-3}" \
    --set "task.dataset.val_ratio=${VAL_RATIO:-0.1}" \
    "$@"
