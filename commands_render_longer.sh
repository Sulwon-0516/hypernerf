export DATASET_PATH=/home/disk1/inhee/hypernerf
export EXPERIMENT_PATH=/home/disk1/inhee/result/hypernerf_longer
python render.py \
    --base_folder $EXPERIMENT_PATH \
    --gin_bindings="data_dir='$DATASET_PATH'" \
    --gin_configs configs/test_local.gin \
    --camera_latent /home/disk1/inhee/view_dnerf.txt \
    --video_dir /home/disk1/inhee/result/renders/hyper_100k_nv_render_v1_230130.mp4 \
    --camera_traj /home/disk1/inhee/hypernerf/camera-paths/nerfstudio-camera

# python render.py \
#     --base_folder $EXPERIMENT_PATH \
#     --gin_bindings="data_dir='$DATASET_PATH'" \
#     --gin_configs configs/test_local.gin \
#     --camera_latent /home/disk1/inhee/view_dnerf.txt \
#     --video_dir /home/disk1/inhee/result/renders/hyper_100k.mp4 \
#     --camera_traj /home/disk1/inhee/hypernerf/camera-paths/nerfstudio-camera

