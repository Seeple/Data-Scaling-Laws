task_name="pick_cup"
logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")
run_dir="data/outputs/${now_date}/${now_seconds}"
echo ${run_dir}

# accelerate launch --mixed_precision 'bf16' ../train.py \
python ../train.py \
--config-name=train_diffusion_unet_timm_umi_workspace \
multi_run.run_dir=${run_dir} multi_run.wandb_name_base=${logging_time} hydra.run.dir=${run_dir} hydra.sweep.dir=${run_dir} \
task.dataset_path=../data/dataset/${task_name}/dataset.zarr.zip \
training.num_epochs=100 \
dataloader.batch_size=16 \
dataloader.num_workers=8 \
val_dataloader.num_workers=8 \
logging.name="${logging_time}_${task_name}" \
policy.obs_encoder.model_name='vit_large_patch14_dinov2.lvd142m' \
task.dataset.use_ratio=1.0 \
task.dataset.val_ratio=0.1 