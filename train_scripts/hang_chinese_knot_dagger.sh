task_name="hang_chinese_knot_dagger"
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
HF_ENDPOINT="${HF_ENDPOINT:-}"
HF_HOME="${HF_HOME:-/home/fangyuan/hf_cache}"
MODEL_PRETRAINED="${MODEL_PRETRAINED:-}"
HF_OFFLINE="${HF_OFFLINE:-}"
WANDB_MODE="${WANDB_MODE:-}"
USE_WANDB="${USE_WANDB:-}"
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

# Zarr cache (LMDB) on local SSD to reduce transient read errors
CACHE_DIR="/home/fangyuan/ssd/umi_cache"
mkdir -p "${CACHE_DIR}"

# Enable full tracebacks and worker crash visibility
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export ACCELERATE_LOG_LEVEL=info
export TORCH_DATALOADER_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export FINETUNE_CKPT="/home/fangyuan/project_lab/SuperInference/third_party/data-scaling-laws/train_scripts/data/outputs/2026.03.31/16.16.44/checkpoints/latest.ckpt"

# Optional: set FINETUNE_CKPT=/path/to/checkpoint.ckpt to enable finetuning
finetune_ckpt="${FINETUNE_CKPT:-}"
finetune_args=()
if [ -n "${finetune_ckpt}" ]; then
	finetune_args=("training.finetune_from=${finetune_ckpt}" "training.reset_steps_on_finetune=True")
fi

# launch training
# disable mixed precision here for more stable training
accelerate launch --config_file "${ACCELERATE_CONFIG_FILE}" "${ACCELERATE_ARGS[@]}" --mixed_precision 'bf16' ../train.py \
	--config-name=train_diffusion_unet_timm_umi_dagger_workspace \
	multi_run.run_dir=${run_dir} multi_run.wandb_name_base=${logging_time} hydra.run.dir=${run_dir} hydra.sweep.dir=${run_dir} \
	task.teleop_dataset_path=../data/dataset/hang_chinese_knot/teleop_data/hang_chinese_knot_raw.zarr.zip \
	task.hitl_dataset_path=../data/dataset/hang_chinese_knot/hitl_data/vr_rtc_hitl/hang_chinese_knot_vrhitl_1.zarr.zip \
	training.num_epochs=100 \
	dataloader.batch_size=8 \
	dataloader.num_workers=4 \
	dataloader.persistent_workers=False \
	val_dataloader.num_workers=2 \
	optimizer.lr=1e-5 \
	training.lr_warmup_steps=500 \
	val_dataloader.persistent_workers=False \
	logging.name="${logging_time}_${task_name}_repro" \
	policy.obs_encoder.model_name='vit_large_patch14_dinov2.lvd142m' \
	task.dataset.use_ratio=1.0 \
	task.dataset.val_ratio=0.1 \
	task.dataset.cache_dir=${CACHE_DIR} \
	training.gradient_accumulate_every=2 \
	training.rollout_every=201 \
	task.dataset.hitl_prob=0.5 \
	logging.use_wandb=True \
	training.freeze_encoder_on_finetune=True \
	training.freeze_encoder_epochs=3 \
	"${HYDRA_ARGS[@]}" \
	"${finetune_args[@]}" \
	2>&1 | tee ${run_dir}/debug_workers.log
