export DATASET_PATH=/home/disk1/inhee/hypernerf
export EXPERIMENT_PATH=/home/disk1/inhee/result/hypernerf
python eval.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local.gin

