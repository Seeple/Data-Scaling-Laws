#!/usr/bin/env bash
set -euo pipefail

# Chunk-level active-plan residual training. The historical filename remains
# for compatibility; set TASK_NAME for other tasks.
#
# Required:
#   BASE_POLICY_CKPT=/path/to/base.ckpt
#   RESIDUAL_DATASET_PATH=/path/to/residual_events.zarr.zip
#
# Variants:
#   VARIANT=correction_only  # capacity/smoke test; no zero supervision or gate
#   VARIANT=mixed_no_gate    # correction + zero attempts, balanced 1:1
#   VARIANT=mixed_gate       # same data plus explicit chunk-level gate
#   ACTION_ALIGNMENT=recorded_chunk | active_suffix_left_aligned
#
# Example:
#   BASE_POLICY_CKPT=../data/ckpts/manage_table/teleop/base.ckpt \
#   RESIDUAL_DATASET_PATH=../data/dataset/manage_table/residual.zarr.zip \
#   VARIANT=mixed_gate bash manage_table_residual.sh
#
# Hang Chinese Knot:
#   TASK_NAME=hang_chinese_knot VARIANT=mixed_gate \
#     bash manage_table_residual.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

: "${BASE_POLICY_CKPT:?Set BASE_POLICY_CKPT to the frozen base DP checkpoint}"
: "${RESIDUAL_DATASET_PATH:?Set RESIDUAL_DATASET_PATH to residual Zarr}"

VARIANT="${VARIANT:-mixed_no_gate}"
TASK_NAME="${TASK_NAME:-manage_table}"
GPU_LIST="${GPU_LIST:-}"
NUM_PROCESSES="${NUM_PROCESSES:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29511}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
CORRECTION_FRACTION="${CORRECTION_FRACTION:-0.5}"
VAL_RATIO="${VAL_RATIO:-0.15}"
# Set an explicit Hydra list (for example '[2,13,18,21]') to keep
# correction-only / mixed / gated ablations on exactly the same episodes.
VAL_EPISODE_INDICES="${VAL_EPISODE_INDICES:-null}"
USE_WANDB="${USE_WANDB:-true}"
ENCODER_MODE="${ENCODER_MODE:-eval}"
CONDITION_ON_BASE_ACTION="${CONDITION_ON_BASE_ACTION:-true}"
# Set to recorded_chunk or active_suffix_left_aligned. "auto" keeps backward
# compatibility, but explicit values are recommended for real experiments.
ACTION_ALIGNMENT="${ACTION_ALIGNMENT:-auto}"
CORRECTION_LOSS_WEIGHT="${CORRECTION_LOSS_WEIGHT:-1.0}"
GATE_LOSS_WEIGHT="${GATE_LOSS_WEIGHT:-1.0}"

if [ -n "${GPU_LIST}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
fi

case "${VARIANT}" in
  correction_only)
    SAMPLE_MODE="correction_only"
    CORRECTION_FRACTION_ARG="null"
    ZERO_LOSS_WEIGHT="0.0"
    GATE_ENABLED="false"
    ;;
  mixed_no_gate)
    SAMPLE_MODE="all"
    CORRECTION_FRACTION_ARG="${CORRECTION_FRACTION}"
    ZERO_LOSS_WEIGHT="1.0"
    GATE_ENABLED="false"
    ;;
  mixed_gate)
    SAMPLE_MODE="all"
    CORRECTION_FRACTION_ARG="${CORRECTION_FRACTION}"
    ZERO_LOSS_WEIGHT="1.0"
    GATE_ENABLED="true"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}; use correction_only, mixed_no_gate or mixed_gate" >&2
    exit 2
    ;;
esac

logging_time="$(date '+%d-%H.%M.%S')"
now_date="$(date '+%Y.%m.%d')"
run_dir="data/outputs/${now_date}/${logging_time: -8}_${TASK_NAME}_residual_${VARIANT}"
mkdir -p "${run_dir}"

ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${SCRIPT_DIR}/../accelerate_config.yaml}"
ACCELERATE_ARGS=()
if [ -n "${NUM_PROCESSES}" ]; then
  ACCELERATE_ARGS+=(--num_processes "${NUM_PROCESSES}")
fi

export BASE_POLICY_CKPT RESIDUAL_DATASET_PATH
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

accelerate launch \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --config_file "${ACCELERATE_CONFIG_FILE}" \
  "${ACCELERATE_ARGS[@]}" \
  --mixed_precision "${MIXED_PRECISION}" \
  ../train.py \
  --config-name=train_chunk_residual_mlp_workspace \
  hydra.run.dir="${run_dir}" \
  task.dataset.sample_mode="${SAMPLE_MODE}" \
  task.dataset.correction_fraction="${CORRECTION_FRACTION_ARG}" \
  task.dataset.val_ratio="${VAL_RATIO}" \
  task.dataset.val_episode_indices="${VAL_EPISODE_INDICES}" \
  task.dataset.expected_action_alignment="${ACTION_ALIGNMENT}" \
  policy.model.gate_enabled="${GATE_ENABLED}" \
  policy.model.condition_on_base_action="${CONDITION_ON_BASE_ACTION}" \
  policy.zero_loss_weight="${ZERO_LOSS_WEIGHT}" \
  policy.correction_loss_weight="${CORRECTION_LOSS_WEIGHT}" \
  policy.gate_loss_weight="${GATE_LOSS_WEIGHT}" \
  base_policy.encoder_mode="${ENCODER_MODE}" \
  training.num_epochs="${NUM_EPOCHS}" \
  dataloader.batch_size="${BATCH_SIZE}" \
  dataloader.num_workers="${NUM_WORKERS}" \
  dataloader.persistent_workers="$([ "${NUM_WORKERS}" -gt 0 ] && echo true || echo false)" \
  val_dataloader.num_workers="${NUM_WORKERS}" \
  val_dataloader.persistent_workers="$([ "${NUM_WORKERS}" -gt 0 ] && echo true || echo false)" \
  optimizer.lr="${LEARNING_RATE}" \
  logging.use_wandb="${USE_WANDB}" \
  logging.name="${logging_time}_${TASK_NAME}_residual_${VARIANT}" \
  2>&1 | tee "${run_dir}/train.log"
