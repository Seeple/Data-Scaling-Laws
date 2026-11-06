#!/usr/bin/env bash
# Full online-rollout weighting, adapted from sirius_weighted_finetune.sh.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

TELEOP_DATASET_PATH="${TELEOP_DATASET_PATH:-${DATA_ROOT}/dataset/teleop.zarr.zip}"
HITL_DATASET_PATH="${HITL_DATASET_PATH:-${DATA_ROOT}/dataset/hitl.zarr.zip}"
FINETUNE_CKPT="${FINETUNE_CKPT:-${DATA_ROOT}/ckpts/base.ckpt}"
NUM_EPOCHS="${NUM_EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29520}"
MODEL_PRETRAINED="${MODEL_PRETRAINED:-false}"

online_args=()
if ! recipe_has_option --online-dataset "$@"; then
    online_args+=(--online-dataset "${HITL_DATASET_PATH}")
fi
ratio_args=()
if [[ -n "${RLPD_RATIO:-}" ]]; then
    ratio_args+=(--set "task.dataset.rlpd_ratio=${RLPD_RATIO}")
fi

recipe_launch sirius \
    --teleop-dataset "${TELEOP_DATASET_PATH}" \
    "${online_args[@]}" \
    --checkpoint "${FINETUNE_CKPT}" \
    "${ratio_args[@]}" \
    --set "task.dataset.sirius_target_intervention_fraction=${SIRIUS_TARGET_INTERVENTION_FRACTION:-0.5}" \
    --set "task.dataset.sirius_target_pre_intervention_fraction=${SIRIUS_TARGET_PRE_INTERVENTION_FRACTION:-0.0}" \
    --set "task.dataset.sirius_pre_intervention_seconds=${SIRIUS_PRE_INTERVENTION_SECONDS:-1.0}" \
    --set "task.dataset.sirius_tag_debounce_enabled=${SIRIUS_TAG_DEBOUNCE_ENABLED:-true}" \
    --set "task.dataset.sirius_merge_policy_gaps_seconds=${SIRIUS_MERGE_POLICY_GAPS_SECONDS:-0.5}" \
    --set "task.dataset.sirius_min_intervention_seconds=${SIRIUS_MIN_INTERVENTION_SECONDS:-0.3}" \
    --set "task.dataset.sirius_max_importance_weight=${SIRIUS_MAX_IMPORTANCE_WEIGHT:-null}" \
    --set "task.dataset.val_ratio=${VAL_RATIO:-0.1}" \
    "$@"
