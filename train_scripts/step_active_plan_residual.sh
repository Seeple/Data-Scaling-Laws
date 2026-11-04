#!/usr/bin/env bash
set -euo pipefail

# Dense step-level active-plan residual training.
#
# Required:
#   BASE_POLICY_CKPT=/absolute/base.ckpt
#   RESIDUAL_DATASET_PATH=/absolute/dense_h5.zarr.zip
#
# Main controls:
#   RESIDUAL_HORIZON=1|5
#   VARIANT=correction_only|mixed_no_gate|mixed_gate
#   TASK_NAME=cable_routing|hang_chinese_knot
#
# The Zarr horizon and RESIDUAL_HORIZON are checked at startup. Dense windows
# from one edit/attempt are family-balanced, and only the best two checkpoints
# by val_composite_objective are retained.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

: "${BASE_POLICY_CKPT:?Set BASE_POLICY_CKPT to the frozen base DP checkpoint}"
: "${RESIDUAL_DATASET_PATH:?Set RESIDUAL_DATASET_PATH to dense residual Zarr}"

VARIANT="${VARIANT:-mixed_no_gate}"
TASK_NAME="${TASK_NAME:-manage_table}"
RESIDUAL_HORIZON="${RESIDUAL_HORIZON:-5}"
GPU_LIST="${GPU_LIST:-}"
NUM_PROCESSES="${NUM_PROCESSES:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29521}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
CORRECTION_FRACTION="${CORRECTION_FRACTION:-0.5}"
VAL_RATIO="${VAL_RATIO:-0.15}"
VAL_EPISODE_INDICES="${VAL_EPISODE_INDICES:-null}"
USE_WANDB="${USE_WANDB:-true}"
ENCODER_MODE="${ENCODER_MODE:-eval}"
CONDITION_ON_BASE_ACTION="${CONDITION_ON_BASE_ACTION:-true}"
CORRECTION_LOSS_WEIGHT="${CORRECTION_LOSS_WEIGHT:-1.0}"
GATE_LOSS_WEIGHT="${GATE_LOSS_WEIGHT:-1.0}"
REPLAY_RESIDUAL_DATASET_PATH="${REPLAY_RESIDUAL_DATASET_PATH:-}"
DATASET_MIXTURE_WEIGHTS="${DATASET_MIXTURE_WEIGHTS:-[0.5,0.5]}"
VAL_EPISODE_INDICES_BY_DATASET="${VAL_EPISODE_INDICES_BY_DATASET:-null}"
INIT_RESIDUAL_CKPT="${INIT_RESIDUAL_CKPT:-}"
INIT_RESIDUAL_STATE_KEY="${INIT_RESIDUAL_STATE_KEY:-auto}"
INIT_USE_CHECKPOINT_NORMALIZER="${INIT_USE_CHECKPOINT_NORMALIZER:-true}"
HUMAN_FRACTION_WITHIN_NONZERO="${HUMAN_FRACTION_WITHIN_NONZERO:-null}"

if [ "${RESIDUAL_HORIZON}" != "1" ] && [ "${RESIDUAL_HORIZON}" != "5" ]; then
  echo "RESIDUAL_HORIZON must be 1 or 5" >&2
  exit 2
fi
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
    echo "Unknown VARIANT=${VARIANT}" >&2
    exit 2
    ;;
esac

logging_time="$(date '+%d-%H.%M.%S')"
now_date="$(date '+%Y.%m.%d')"
run_dir="data/outputs/${now_date}/${logging_time: -8}_${TASK_NAME}_step_residual_h${RESIDUAL_HORIZON}_${VARIANT}"
mkdir -p "${run_dir}"

ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${SCRIPT_DIR}/../accelerate_config.yaml}"
ACCELERATE_ARGS=()
if [ -n "${NUM_PROCESSES}" ]; then
  ACCELERATE_ARGS+=(--num_processes "${NUM_PROCESSES}")
fi

export BASE_POLICY_CKPT RESIDUAL_DATASET_PATH RESIDUAL_HORIZON
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

ROUND2_ARGS=()
DATASET_VAL_EPISODE_INDICES="${VAL_EPISODE_INDICES}"
if [ -n "${REPLAY_RESIDUAL_DATASET_PATH}" ]; then
  DATASET_VAL_EPISODE_INDICES="null"
  ROUND2_ARGS+=(
    'task.dataset._target_=diffusion_policy.dataset.umi_residual_dataset.UmiResidualDatasetCollection'
    "+task.dataset.dataset_paths=[${REPLAY_RESIDUAL_DATASET_PATH},${RESIDUAL_DATASET_PATH}]"
    "+task.dataset.dataset_mixture_weights=${DATASET_MIXTURE_WEIGHTS}"
    "+task.dataset.val_episode_indices_by_dataset=${VAL_EPISODE_INDICES_BY_DATASET}"
  )
fi
if [ -n "${INIT_RESIDUAL_CKPT}" ]; then
  ROUND2_ARGS+=(
    "training.init_residual_checkpoint=${INIT_RESIDUAL_CKPT}"
    "training.init_residual_state_key=${INIT_RESIDUAL_STATE_KEY}"
    "training.init_use_checkpoint_normalizer=${INIT_USE_CHECKPOINT_NORMALIZER}"
  )
fi

accelerate launch \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --config_file "${ACCELERATE_CONFIG_FILE}" \
  "${ACCELERATE_ARGS[@]}" \
  --mixed_precision "${MIXED_PRECISION}" \
  ../train.py \
  --config-name=train_step_active_plan_residual_mlp_workspace \
  "${ROUND2_ARGS[@]}" \
  hydra.run.dir="${run_dir}" \
  task.dataset.sample_mode="${SAMPLE_MODE}" \
  task.dataset.correction_fraction="${CORRECTION_FRACTION_ARG}" \
  task.dataset.val_ratio="${VAL_RATIO}" \
  task.dataset.val_episode_indices="${DATASET_VAL_EPISODE_INDICES}" \
  task.dataset.human_fraction_within_nonzero="${HUMAN_FRACTION_WITHIN_NONZERO}" \
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
  logging.name="${logging_time}_${TASK_NAME}_step_residual_h${RESIDUAL_HORIZON}_${VARIANT}" \
  2>&1 | tee "${run_dir}/train.log"
