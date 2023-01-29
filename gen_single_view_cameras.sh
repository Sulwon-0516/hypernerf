CAM_DIR=/home/disk1/inhee/hypernerf/camera/00000.json
GEN_DIR=/home/disk1/inhee/hypernerf/camera-paths/nerfstudio-camera-fixed

mkdir $GEN_DIR
for i in {00001..00100}
do
   cp $CAM_DIR $GEN_DIR/${i}.json
done

