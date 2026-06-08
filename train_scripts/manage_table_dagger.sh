task_name="manage_table_lr5e-5_rlpd0.5_vrrtc_hitl_single_pattern_sanity_iter2_check_only_human_downsample6"
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
#   GPU_LIST="0,1,2,3" NUM_PROCESSES=4 bash manage_table_dagger.sh
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
#   MODEL_PRETRAINED=false bash manage_table_dagger.sh

# Whether disable downsampling for HITL data (set to false by default)
# Example:
#   HITL_DISABLE_DOWNSAMPLE=true HITL_DOWNSAMPLE_MULTIPLIER=2 bash manage_table_dagger.sh
HF_ENDPOINT="${HF_ENDPOINT:-}"
HF_HOME="${HF_HOME:-/home/fangyuan/hf_cache}"
MODEL_PRETRAINED="${MODEL_PRETRAINED:-}"
HF_OFFLINE="${HF_OFFLINE:-}"
WANDB_MODE="${WANDB_MODE:-}"
USE_WANDB="${USE_WANDB:-}"
ONLY_CAMERA_OBS="${ONLY_CAMERA_OBS:-}"
HITL_DISABLE_DOWNSAMPLE="${HITL_DISABLE_DOWNSAMPLE:-}"
HITL_DOWNSAMPLE_MULTIPLIER="${HITL_DOWNSAMPLE_MULTIPLIER:-}"
HITL_ONLY_TAG="${HITL_ONLY_TAG:-}"
HITL_REQUIRE_FULL_ACTION_TAG="${HITL_REQUIRE_FULL_ACTION_TAG:-}"
HITL_ACTION_MASK="${HITL_ACTION_MASK:-}"
HITL_SKIP_RISING_EDGE="${HITL_SKIP_RISING_EDGE:-}"
HITL_SKIP_RISING_EDGE_STEPS="${HITL_SKIP_RISING_EDGE_STEPS:-}"
HITL_TREAT_SEGMENTS_AS_EPISODES="${HITL_TREAT_SEGMENTS_AS_EPISODES:-}"
LOWDIM_OBS_NORMALIZER_SOURCE="${LOWDIM_OBS_NORMALIZER_SOURCE:-}"
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
if [ -n "${HITL_ONLY_TAG}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_only_tag=${HITL_ONLY_TAG}")
fi
if [ -n "${HITL_REQUIRE_FULL_ACTION_TAG}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_require_full_action_tag=${HITL_REQUIRE_FULL_ACTION_TAG}")
fi
if [ -n "${HITL_ACTION_MASK}" ]; then
	HYDRA_ARGS+=("task.dataset.hitl_action_mask=${HITL_ACTION_MASK}")
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
export FINETUNE_CKPT="/home/fangyuan/project/Data-Scaling-Laws/train_scripts/data/ckpts/manage_table_dagger_w_single_pattern_iter1_only_human_downsample6.ckpt"

# Optional: set FINETUNE_CKPT=/path/to/checkpoint.ckpt to enable finetuning
finetune_ckpt="${FINETUNE_CKPT:-}"
finetune_args=()
if [ -n "${finetune_ckpt}" ]; then
	finetune_args=("training.finetune_from=${finetune_ckpt}" "training.reset_steps_on_finetune=True")
fi

# launch training
accelerate launch --main_process_port 29501 --config_file "${ACCELERATE_CONFIG_FILE}" "${ACCELERATE_ARGS[@]}" --mixed_precision 'bf16' ../train.py \
	--config-name=train_diffusion_unet_timm_umi_dagger_workspace \
	multi_run.run_dir=${run_dir} multi_run.wandb_name_base=${logging_time} hydra.run.dir=${run_dir} hydra.sweep.dir=${run_dir} \
	task.teleop_dataset_path=../data/dataset/manage_table/teleop_data/manage_table_raw_4_28.zarr.zip \
	task.hitl_dataset_path=../data/dataset/manage_table/hitl_data/vrrtc_hitl/manage_table_vrhitl_single_pattern_iter2.zarr.zip \
	training.num_epochs=300 \
	dataloader.batch_size=64 \
	dataloader.num_workers=16 \
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
	task.dataset.hitl_prob=0.5 \
	logging.use_wandb=True \
	training.freeze_encoder_on_finetune=True \
	training.freeze_encoder_epochs=3 \
	"${HYDRA_ARGS[@]}" \
	"${finetune_args[@]}" \
	2>&1 | tee ${run_dir}/debug_workers.log
