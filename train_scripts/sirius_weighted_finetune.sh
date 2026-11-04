#!/usr/bin/env bash
set -euo pipefail

# Generic 15 Hz Sirius-style weighted finetuning entry point.
#
# Required:
#   TASK_NAME=hang_chinese_knot \
#   TELEOP_DATASET_PATH=/abs/expert.zarr.zip \
#   HITL_DATASET_PATH=/abs/full_vrrtc_rollout.zarr.zip \
#   FINETUNE_CKPT=/abs/base.ckpt \
#   bash train_scripts/sirius_weighted_finetune.sh
#
# Optional multi-GPU example:
#   GPU_LIST=0,1 NUM_PROCESSES=2 MAIN_PROCESS_PORT=29521 ...
#
# The HITL input must retain both policy (hitl_tag=0) and takeover
# (hitl_tag=1) frames. This entry deliberately disables all correction-only
# filters and masks; Sirius assigns a loss weight to every action token.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${TASK_NAME:?Set TASK_NAME}"
: "${TELEOP_DATASET_PATH:?Set TELEOP_DATASET_PATH}"
: "${HITL_DATASET_PATH:?Set HITL_DATASET_PATH}"
: "${FINETUNE_CKPT:?Set FINETUNE_CKPT}"

for required_path in \
    "${TELEOP_DATASET_PATH}" \
    "${HITL_DATASET_PATH}" \
    "${FINETUNE_CKPT}"; do
    if [ ! -f "${required_path}" ]; then
        echo "Required input does not exist: ${required_path}" >&2
        exit 2
    fi
done

GPU_LIST="${GPU_LIST:-}"
NUM_PROCESSES="${NUM_PROCESSES:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29520}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${REPO_ROOT}/accelerate_config.yaml}"
if [ -n "${GPU_LIST}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
fi
ACCELERATE_ARGS=()
if [ -n "${NUM_PROCESSES}" ]; then
    ACCELERATE_ARGS+=(--num_processes "${NUM_PROCESSES}")
fi

NUM_EPOCHS="${NUM_EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-500}"
RLPD_RATIO="${RLPD_RATIO:-[0.5,0.5]}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SIRIUS_TARGET_INTERVENTION_FRACTION="${SIRIUS_TARGET_INTERVENTION_FRACTION:-0.5}"
SIRIUS_TARGET_PRE_INTERVENTION_FRACTION="${SIRIUS_TARGET_PRE_INTERVENTION_FRACTION:-0.0}"
SIRIUS_PRE_INTERVENTION_SECONDS="${SIRIUS_PRE_INTERVENTION_SECONDS:-1.0}"
SIRIUS_TAG_DEBOUNCE_ENABLED="${SIRIUS_TAG_DEBOUNCE_ENABLED:-true}"
SIRIUS_MERGE_POLICY_GAPS_SECONDS="${SIRIUS_MERGE_POLICY_GAPS_SECONDS:-0.5}"
SIRIUS_MIN_INTERVENTION_SECONDS="${SIRIUS_MIN_INTERVENTION_SECONDS:-0.3}"
SIRIUS_MAX_IMPORTANCE_WEIGHT="${SIRIUS_MAX_IMPORTANCE_WEIGHT:-null}"

MODEL_PRETRAINED="${MODEL_PRETRAINED:-false}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-umi}"
export WANDB_MODE

CACHE_DIR="${CACHE_DIR:-/home/fangyuan/ssd/umi_cache}"
TMPDIR="${TMPDIR:-/home/fangyuan/ssd/tmp}"
mkdir -p "${CACHE_DIR}" "${TMPDIR}"
export TMPDIR TMP="${TMPDIR}" TEMP="${TMPDIR}"

STAMP="$(date '+%Y.%m.%d-%H.%M.%S')"
RUN_NAME="${RUN_NAME:-${TASK_NAME}_sirius_weighted_iter1}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/data/outputs/${STAMP}_${RUN_NAME}}"
mkdir -p "${RUN_DIR}"

export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export ACCELERATE_LOG_LEVEL=info

echo "Sirius run directory: ${RUN_DIR}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES:-accelerate default}"

cd "${REPO_ROOT}"
accelerate launch \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --config_file "${ACCELERATE_CONFIG_FILE}" \
    "${ACCELERATE_ARGS[@]}" \
    --mixed_precision bf16 \
    train.py \
    --config-name=train_diffusion_unet_timm_umi_dagger_workspace \
    hydra.run.dir="${RUN_DIR}" \
    hydra.sweep.dir="${RUN_DIR}" \
    multi_run.run_dir="${RUN_DIR}" \
    multi_run.wandb_name_base="${STAMP}_${RUN_NAME}" \
    task.teleop_dataset_path="${TELEOP_DATASET_PATH}" \
    task.hitl_dataset_path="${HITL_DATASET_PATH}" \
    task.dataset.rlpd_ratio="${RLPD_RATIO}" \
    task.dataset.hitl_only_tag=false \
    task.dataset.hitl_require_full_action_tag=false \
    task.dataset.hitl_action_mask=false \
    task.dataset.hitl_contiguous_action_mask=false \
    task.dataset.hitl_invalid_tail_padding=false \
    task.dataset.hitl_min_valid_steps_enabled=false \
    task.dataset.hitl_skip_rising_edge=false \
    task.dataset.hitl_treat_segments_as_episodes=false \
    task.dataset.hitl_disable_downsample=false \
    task.dataset.sirius_weighting_enabled=true \
    task.dataset.sirius_source_fps=15.0 \
    task.dataset.sirius_target_intervention_fraction="${SIRIUS_TARGET_INTERVENTION_FRACTION}" \
    task.dataset.sirius_target_pre_intervention_fraction="${SIRIUS_TARGET_PRE_INTERVENTION_FRACTION}" \
    task.dataset.sirius_pre_intervention_seconds="${SIRIUS_PRE_INTERVENTION_SECONDS}" \
    task.dataset.sirius_tag_debounce_enabled="${SIRIUS_TAG_DEBOUNCE_ENABLED}" \
    task.dataset.sirius_merge_policy_gaps_seconds="${SIRIUS_MERGE_POLICY_GAPS_SECONDS}" \
    task.dataset.sirius_min_intervention_seconds="${SIRIUS_MIN_INTERVENTION_SECONDS}" \
    task.dataset.sirius_max_importance_weight="${SIRIUS_MAX_IMPORTANCE_WEIGHT}" \
    task.dataset.action_normalizer_source=teleop \
    task.dataset.lowdim_obs_normalizer_source=teleop \
    task.dataset.cache_dir="${CACHE_DIR}" \
    task.dataset.val_ratio="${VAL_RATIO}" \
    task.obs_down_sample_steps=3 \
    task.action_down_sample_steps=3 \
    task.dataset_frequeny=15 \
    training.finetune_from="${FINETUNE_CKPT}" \
    training.reset_steps_on_finetune=true \
    training.num_epochs="${NUM_EPOCHS}" \
    training.lr_warmup_steps="${LR_WARMUP_STEPS}" \
    training.freeze_encoder_on_finetune=true \
    training.freeze_encoder_epochs=3 \
    training.rollout_every=1000 \
    training.sample_every=5 \
    training.checkpoint_every=10 \
    optimizer.lr="${LEARNING_RATE}" \
    dataloader.batch_size="${BATCH_SIZE}" \
    dataloader.num_workers="${NUM_WORKERS}" \
    dataloader.persistent_workers=true \
    val_dataloader.num_workers="${VAL_NUM_WORKERS}" \
    val_dataloader.persistent_workers=true \
    policy.obs_encoder.model_name=vit_large_patch14_dinov2.lvd142m \
    policy.obs_encoder.pretrained="${MODEL_PRETRAINED}" \
    checkpoint.topk.k=2 \
    logging.project="${WANDB_PROJECT}" \
    logging.use_wandb="${USE_WANDB}" \
    logging.name="${STAMP}_${RUN_NAME}" \
    2>&1 | tee "${RUN_DIR}/train.log"
