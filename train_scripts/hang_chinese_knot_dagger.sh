task_name="hang_chinese_knot_dagger_iter2_fixed_position_lr5e-5_rlpd0.5_downsample6_vrrtc_hitl"
logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")
run_dir="data/outputs/${now_date}/${now_seconds}"
mkdir -p "${run_dir}"
echo "${run_dir}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${SCRIPT_DIR}/../accelerate_config.yaml}"

# Multi-GPU settings (optional)
# Example:
#   GPU_LIST="0,1,2,3" NUM_PROCESSES=4 bash hang_chinese_knot_dagger.sh
GPU_LIST="${GPU_LIST:-}"
NUM_PROCESSES="${NUM_PROCESSES:-}"
if [ -n "${GPU_LIST}" ]; then
	export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
fi
ACCELERATE_ARGS=()
if [ -n "${NUM_PROCESSES}" ]; then
	ACCELERATE_ARGS+=(--num_processes "${NUM_PROCESSES}")
fi

# Hugging Face cache / endpoint (optional)
# Example:
#   HF_ENDPOINT=https://hf-mirror.com HF_HOME=/home/fangyuan/hf_cache \
#   MODEL_PRETRAINED=false bash hang_chinese_knot_dagger.sh

# HITL ablation switches (all new features are opt-in).
#
# Contiguous-prefix mask:
#   HITL_ONLY_TAG=true HITL_ACTION_MASK=true \
#   HITL_CONTIGUOUS_ACTION_MASK=true bash hang_chinese_knot_dagger.sh
#
# Invalid-tail padding (also requires the contiguous-prefix mask):
#   HITL_ONLY_TAG=true HITL_ACTION_MASK=true \
#   HITL_CONTIGUOUS_ACTION_MASK=true HITL_INVALID_TAIL_PADDING=true \
#   bash hang_chinese_knot_dagger.sh
#
# Require at least five continuous human action positions per HITL anchor:
#   HITL_ONLY_TAG=true HITL_ACTION_MASK=true \
#   HITL_CONTIGUOUS_ACTION_MASK=true HITL_MIN_VALID_STEPS_ENABLED=true \
#   HITL_MIN_VALID_STEPS=5 bash hang_chinese_knot_dagger.sh
#
# Separate HITL observation/action stride multipliers. The historical
# HITL_DISABLE_DOWNSAMPLE gate must remain true; this example keeps obs at
# stride 3 (x1) while changing action to stride 6 (x2):
#   HITL_DISABLE_DOWNSAMPLE=true \
#   HITL_SEPARATE_DOWNSAMPLE_MULTIPLIERS=true \
#   HITL_OBS_DOWNSAMPLE_MULTIPLIER=1 \
#   HITL_ACTION_DOWNSAMPLE_MULTIPLIER=2 bash hang_chinese_knot_dagger.sh
HF_ENDPOINT="${HF_ENDPOINT:-}"
HF_HOME="${HF_HOME:-/home/fangyuan/hf_cache}"
MODEL_PRETRAINED="${MODEL_PRETRAINED:-}"
HF_OFFLINE="${HF_OFFLINE:-}"
WANDB_MODE="${WANDB_MODE:-}"
USE_WANDB="${USE_WANDB:-}"
ONLY_CAMERA_OBS="${ONLY_CAMERA_OBS:-}"
HITL_DISABLE_DOWNSAMPLE="${HITL_DISABLE_DOWNSAMPLE:-}"
HITL_DOWNSAMPLE_MULTIPLIER="${HITL_DOWNSAMPLE_MULTIPLIER:-}"
HITL_SEPARATE_DOWNSAMPLE_MULTIPLIERS="${HITL_SEPARATE_DOWNSAMPLE_MULTIPLIERS:-}"
HITL_OBS_DOWNSAMPLE_MULTIPLIER="${HITL_OBS_DOWNSAMPLE_MULTIPLIER:-}"
HITL_ACTION_DOWNSAMPLE_MULTIPLIER="${HITL_ACTION_DOWNSAMPLE_MULTIPLIER:-}"
HITL_ONLY_TAG="${HITL_ONLY_TAG:-}"
HITL_REQUIRE_FULL_ACTION_TAG="${HITL_REQUIRE_FULL_ACTION_TAG:-}"
HITL_ACTION_MASK="${HITL_ACTION_MASK:-}"
HITL_CONTIGUOUS_ACTION_MASK="${HITL_CONTIGUOUS_ACTION_MASK:-}"
HITL_INVALID_TAIL_PADDING="${HITL_INVALID_TAIL_PADDING:-}"
HITL_MIN_VALID_STEPS_ENABLED="${HITL_MIN_VALID_STEPS_ENABLED:-}"
HITL_MIN_VALID_STEPS="${HITL_MIN_VALID_STEPS:-}"
HITL_SKIP_RISING_EDGE="${HITL_SKIP_RISING_EDGE:-}"
HITL_SKIP_RISING_EDGE_STEPS="${HITL_SKIP_RISING_EDGE_STEPS:-}"
HITL_TREAT_SEGMENTS_AS_EPISODES="${HITL_TREAT_SEGMENTS_AS_EPISODES:-}"
LOWDIM_OBS_NORMALIZER_SOURCE="${LOWDIM_OBS_NORMALIZER_SOURCE:-}"
ONLINE_DATASET_PATHS="${ONLINE_DATASET_PATHS:-}"
RLPD_RATIO="${RLPD_RATIO:-[0.5,0.5]}"
if [ -n "${HF_ENDPOINT}" ]; then
	export HF_ENDPOINT
fi
if [ -n "${HF_HOME}" ]; then
	export HF_HOME
	mkdir -p "${HF_HOME}"
fi
if [ -n "${HF_OFFLINE}" ]; then
	export HF_HUB_OFFLINE="${HF_OFFLINE}"
	export TRANSFORMERS_OFFLINE="${HF_OFFLINE}"
fi
if [ -n "${WANDB_MODE}" ]; then
	export WANDB_MODE
fi
HYDRA_ARGS=()
if [ -n "${MODEL_PRETRAINED}" ]; then
	HYDRA_ARGS+=("policy.obs_encoder.pretrained=${MODEL_PRETRAINED}")
fi
if [ -n "${USE_WANDB}" ]; then
	HYDRA_ARGS+=("logging.use_wandb=${USE_WANDB}")
fi
if [ -n "${HITL_DISABLE_DOWNSAMPLE}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_disable_downsample=${HITL_DISABLE_DOWNSAMPLE}")
fi
if [ -n "${HITL_DOWNSAMPLE_MULTIPLIER}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_downsample_multiplier=${HITL_DOWNSAMPLE_MULTIPLIER}")
fi
if [ -n "${HITL_SEPARATE_DOWNSAMPLE_MULTIPLIERS}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_separate_downsample_multipliers=${HITL_SEPARATE_DOWNSAMPLE_MULTIPLIERS}")
fi
if [ -n "${HITL_OBS_DOWNSAMPLE_MULTIPLIER}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_obs_downsample_multiplier=${HITL_OBS_DOWNSAMPLE_MULTIPLIER}")
fi
if [ -n "${HITL_ACTION_DOWNSAMPLE_MULTIPLIER}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_action_downsample_multiplier=${HITL_ACTION_DOWNSAMPLE_MULTIPLIER}")
fi
if [ -n "${HITL_ONLY_TAG}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_only_tag=${HITL_ONLY_TAG}")
fi
if [ -n "${HITL_REQUIRE_FULL_ACTION_TAG}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_require_full_action_tag=${HITL_REQUIRE_FULL_ACTION_TAG}")
fi
if [ -n "${HITL_ACTION_MASK}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_action_mask=${HITL_ACTION_MASK}")
fi
if [ -n "${HITL_CONTIGUOUS_ACTION_MASK}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_contiguous_action_mask=${HITL_CONTIGUOUS_ACTION_MASK}")
fi
if [ -n "${HITL_INVALID_TAIL_PADDING}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_invalid_tail_padding=${HITL_INVALID_TAIL_PADDING}")
fi
if [ -n "${HITL_MIN_VALID_STEPS_ENABLED}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_min_valid_steps_enabled=${HITL_MIN_VALID_STEPS_ENABLED}")
fi
if [ -n "${HITL_MIN_VALID_STEPS}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_min_valid_steps=${HITL_MIN_VALID_STEPS}")
fi
if [ -n "${HITL_SKIP_RISING_EDGE}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_skip_rising_edge=${HITL_SKIP_RISING_EDGE}")
fi
if [ -n "${HITL_SKIP_RISING_EDGE_STEPS}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_skip_rising_edge_steps=${HITL_SKIP_RISING_EDGE_STEPS}")
fi
if [ -n "${HITL_TREAT_SEGMENTS_AS_EPISODES}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_treat_segments_as_episodes=${HITL_TREAT_SEGMENTS_AS_EPISODES}")
fi
if [ -n "${LOWDIM_OBS_NORMALIZER_SOURCE}" ]; then
	HYDRA_ARGS+=("task.dataset.lowdim_obs_normalizer_source=${LOWDIM_OBS_NORMALIZER_SOURCE}")
fi
if [ -n "${ONLINE_DATASET_PATHS}" ]; then
	ONLINE_DATASET_PATHS_ARG="${ONLINE_DATASET_PATHS}"
	case "${ONLINE_DATASET_PATHS_ARG}" in
		\[*\]) ;;
		*) ONLINE_DATASET_PATHS_ARG="[${ONLINE_DATASET_PATHS_ARG}]" ;;
	esac
	HYDRA_ARGS+=("task.online_dataset_paths=${ONLINE_DATASET_PATHS_ARG}")
fi
if [ -n "${RLPD_RATIO}" ]; then
	RLPD_RATIO_ARG="${RLPD_RATIO}"
	case "${RLPD_RATIO_ARG}" in
		\[*\]) ;;
		*) RLPD_RATIO_ARG="[${RLPD_RATIO_ARG}]" ;;
	esac
	HYDRA_ARGS+=("task.dataset.rlpd_ratio=${RLPD_RATIO_ARG}")
fi
if [ -n "${ONLY_CAMERA_OBS}" ]; then
	HYDRA_ARGS+=("task.ignore_proprioception=${ONLY_CAMERA_OBS}")
fi

# Zarr cache (LMDB) on local SSD to reduce transient read errors
CACHE_DIR="/home/fangyuan/ssd/umi_cache"
mkdir -p "${CACHE_DIR}"

# Temp dir for multiprocessing / shared memory to avoid /tmp space issues
TMPDIR="${TMPDIR:-/home/fangyuan/ssd/tmp}"
mkdir -p "${TMPDIR}"
export TMPDIR TMP TEMP="${TMPDIR}"


# Enable full tracebacks and worker crash visibility
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export ACCELERATE_LOG_LEVEL=info
export TORCH_DATALOADER_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
# export FINETUNE_CKPT="/home/fangyuan/project/Data-Scaling-Laws/train_scripts/data/ckpts/hang_chinese_knot_raw_teleop_4_21.ckpt"
export FINETUNE_CKPT="/home/fangyuan/project/Data-Scaling-Laws/train_scripts/data/ckpts/hang_chinese_knot_dagger_w_vrrtc_4_23.ckpt"

# Optional: set FINETUNE_CKPT=/path/to/checkpoint.ckpt to enable finetuning
finetune_ckpt="${FINETUNE_CKPT:-}"
finetune_args=()
if [ -n "${finetune_ckpt}" ]; then
	finetune_args=("training.finetune_from=${finetune_ckpt}" "training.reset_steps_on_finetune=True")
fi

# launch training
accelerate launch --main_process_port 29514 --config_file "${ACCELERATE_CONFIG_FILE}" "${ACCELERATE_ARGS[@]}" --mixed_precision 'bf16' ../train.py \
	--config-name=train_diffusion_unet_timm_umi_dagger_workspace \
	multi_run.run_dir=${run_dir} multi_run.wandb_name_base=${logging_time} hydra.run.dir=${run_dir} hydra.sweep.dir=${run_dir} \
	task.teleop_dataset_path=../data/dataset/hang_chinese_knot/teleop_data/hang_chinese_knot_raw_2.zarr.zip \
	task.hitl_dataset_path=../data/dataset/hang_chinese_knot/hitl_data/vr_rtc_hitl/hang_chinese_knot_vrhitl_iter2_fixed_position_4_29.zarr.zip \
	training.num_epochs=200 \
	dataloader.batch_size=32 \
	dataloader.num_workers=8 \
	dataloader.persistent_workers=True \
	val_dataloader.num_workers=4 \
	val_dataloader.persistent_workers=True \
	optimizer.lr=5e-5 \
	training.lr_warmup_steps=500 \
	logging.name="${logging_time}_${task_name}_repro" \
	policy.obs_encoder.model_name='vit_large_patch14_dinov2.lvd142m' \
	task.dataset.use_ratio=1.0 \
	task.dataset.val_ratio=0.1 \
	task.dataset.cache_dir=${CACHE_DIR} \
	training.gradient_accumulate_every=1 \
	training.rollout_every=1000 \
	logging.use_wandb=True \
	training.freeze_encoder_on_finetune=True \
	training.freeze_encoder_epochs=3 \
	"${HYDRA_ARGS[@]}" \
	"${finetune_args[@]}" \
	2>&1 | tee ${run_dir}/debug_workers.log
