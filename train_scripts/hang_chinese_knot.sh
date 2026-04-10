# task_name="hang_chinese_knot"
# logging_time=$(date "+%d-%H.%M.%S")
# now_seconds="${logging_time: -8}"
# now_date=$(date "+%Y.%m.%d")
# run_dir="data/outputs/${now_date}/${now_seconds}"
# echo ${run_dir}

# # python ../train.py \
# accelerate launch --mixed_precision 'bf16' ../train.py \
# --config-name=train_diffusion_unet_timm_umi_workspace \
# multi_run.run_dir=${run_dir} multi_run.wandb_name_base=${logging_time} hydra.run.dir=${run_dir} hydra.sweep.dir=${run_dir} \
# task.dataset_path=../data/dataset/${task_name}/teleop_data/hang_chinese_knot_raw.zarr.zip \
# training.num_epochs=100 \
# dataloader.batch_size=8 \
# dataloader.num_workers=4 \
# val_dataloader.num_workers=4 \
# logging.name="${logging_time}_${task_name}" \
# policy.obs_encoder.model_name='vit_large_patch14_dinov2.lvd142m' \
# task.dataset.use_ratio=1.0 \
# task.dataset.val_ratio=0.1 \
# training.gradient_accumulate_every=2 \
# training.rollout_every=101

task_name="hang_chinese_knot"
logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")
run_dir="data/outputs/${now_date}/${now_seconds}"
mkdir -p "${run_dir}"
echo "${run_dir}"

# Zarr cache (LMDB) on local SSD to reduce transient read errors
CACHE_DIR="/mnt/ssd/umi_cache"
mkdir -p "${CACHE_DIR}"

export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export ACCELERATE_LOG_LEVEL=info
export TORCH_DATALOADER_DEBUG=INFO   # optional: more verbose worker errors

accelerate launch --mixed_precision 'bf16' ../train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  multi_run.run_dir=${run_dir} multi_run.wandb_name_base=${logging_time} hydra.run.dir=${run_dir} hydra.sweep.dir=${run_dir} \
  task.dataset_path=../data/dataset/${task_name}/teleop_data/hang_chinese_knot_raw.zarr.zip \
  training.num_epochs=100 \
  dataloader.batch_size=8 \
  dataloader.num_workers=4 \
  dataloader.persistent_workers=False \
  val_dataloader.num_workers=4 \
  val_dataloader.persistent_workers=False \
  logging.name="${logging_time}_${task_name}_repro" \
  policy.obs_encoder.model_name='vit_large_patch14_dinov2.lvd142m' \
  task.dataset.use_ratio=1.0 \
  task.dataset.val_ratio=0.1 \
  task.dataset.cache_dir=${CACHE_DIR} \
  training.gradient_accumulate_every=2 \
  training.rollout_every=101 \
  2>&1 | tee ${run_dir}/debug_workers.log