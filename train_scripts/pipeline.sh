# For first iteration DAgger (default)
cd ./train_scripts

HF_ENDPOINT=https://hf-mirror.com HF_HOME=/home/fangyuan/hf_cache \
HITL_ONLY_TAG=true \ 
HITL_ACTION_MASK=true \
HITL_DISABLE_DOWNSAMPLE=true HITL_DOWNSAMPLE_MULTIPLIER=2 \
GPU_LIST="0,1,2,3" NUM_PROCESSES=4 \
bash manage_table_dagger.sh

# For second iteration DAgger (with RLPD, multiple online buffers)
cd ./train_scripts

HF_ENDPOINT=https://hf-mirror.com HF_HOME=/home/fangyuan/hf_cache \
HITL_ONLY_TAG=true \
HITL_ACTION_MASK=true \
HITL_DISABLE_DOWNSAMPLE=true HITL_DOWNSAMPLE_MULTIPLIER=2 \
ONLINE_DATASET_PATHS="../data/dataset/manage_table/hitl_data/iter1.zarr.zip,../data/dataset/manage_table/hitl_data/iter2.zarr.zip" \
RLPD_RATIO="[0.4,0.2,0.4]" \
GPU_LIST="4,5,6,7" NUM_PROCESSES=4 \
bash manage_table_dagger.sh