#!/usr/bin/env bash
# Full-chunk residual; recorded_chunk matches the current ARISE deployment.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

BASE_POLICY_CKPT="${BASE_POLICY_CKPT:-${DATA_ROOT}/ckpts/base.ckpt}"
RESIDUAL_DATASET_PATH="${RESIDUAL_DATASET_PATH:-${DATA_ROOT}/dataset/residual-round1.zarr.zip}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-${NUM_WORKERS}}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29511}"

warm_start_args=()
if [[ -n "${INIT_RESIDUAL_CKPT:-}" ]] && ! recipe_has_option --init-residual "$@" && ! recipe_has_option --resume "$@"; then
    warm_start_args+=(--init-residual "${INIT_RESIDUAL_CKPT}")
fi
fraction_args=()
if [[ -n "${CORRECTION_FRACTION:-}" ]]; then
    fraction_args+=(--set "task.dataset.correction_fraction=${CORRECTION_FRACTION}")
fi
replay_args=()
if [[ -n "${REPLAY_RESIDUAL_DATASET_PATH:-}" ]] && ! recipe_has_option --replay-dataset "$@"; then
    replay_args+=(--replay-dataset "${REPLAY_RESIDUAL_DATASET_PATH}")
fi

recipe_launch residual \
    --dataset "${RESIDUAL_DATASET_PATH}" \
    --base-checkpoint "${BASE_POLICY_CKPT}" \
    --variant "${VARIANT:-mixed_no_gate}" \
    --action-alignment "${ACTION_ALIGNMENT:-recorded_chunk}" \
    "${warm_start_args[@]}" "${replay_args[@]}" "${fraction_args[@]}" \
    --set "task.dataset.val_ratio=${VAL_RATIO:-0.15}" \
    --set "task.dataset.human_fraction_within_nonzero=${HUMAN_FRACTION_WITHIN_NONZERO:-null}" \
    --set "base_policy.encoder_mode=${ENCODER_MODE:-eval}" \
    --set "policy.model.condition_on_base_action=${CONDITION_ON_BASE_ACTION:-true}" \
    --set "policy.correction_loss_weight=${CORRECTION_LOSS_WEIGHT:-1.0}" \
    --set "policy.gate_loss_weight=${GATE_LOSS_WEIGHT:-1.0}" \
    --set "validation.human_loss_weight=${VAL_HUMAN_LOSS_WEIGHT:-1.0}" \
    --set "validation.accepted_rollout_loss_weight=${VAL_ACCEPTED_ROLLOUT_LOSS_WEIGHT:-1.0}" \
    --set "validation.zero_loss_weight=${VAL_ZERO_LOSS_WEIGHT:-1.0}" \
    --set "training.checkpoint_every=${CHECKPOINT_EVERY:-10}" \
    --set "training.init_residual_state_key=${INIT_RESIDUAL_STATE_KEY:-auto}" \
    --set "training.init_use_checkpoint_normalizer=${INIT_USE_CHECKPOINT_NORMALIZER:-true}" \
    --set "training.reparameterize_output_for_new_normalizer=${REPARAMETERIZE_OUTPUT_FOR_NEW_NORMALIZER:-true}" \
    --set "training.init_ema_min_decay=${INIT_EMA_MIN_DECAY:-null}" \
    "$@"
