#!/bin/bash

# Edit these paths. The volume mounting below assumes your datasets are e.g. data/Baraja, data/KITTI etc.
DATA_PATH="/mnt/big-data/darren/data"
CODE_PATH="/mnt/big-data/darren/code/VCN"

GPU_ID="0,1,2"

ENVS="  --env=NVIDIA_VISIBLE_DEVICES=$GPU_ID
        --env=CUDA_VISIBLE_DEVICES=$GPU_ID
        --env=NVIDIA_DRIVER_CAPABILITIES=all"

# Modify these paths as necessary to mount the data
VOLUMES="       --volume=$DATA_PATH/shapenet/VC:/VCN/data/VC
                --volume=$DATA_PATH/shapenet:/VCN/data/shapenet
                --volume=$CODE_PATH:/VCN
                --volume=$DATA_PATH/kitti:/VCN/data/KITTI
                --volume=$DATA_PATH/nuscenes:/VCN/data/nuscenes
                --volume=$DATA_PATH/waymo:/VCN/data/waymo
                --volume=$DATA_PATH/baraja:/VCN/data/baraja"

# Setup visualization for point cloud demos
VISUAL="        --env=DISPLAY
                --env=QT_X11_NO_MITSHM=1
                --volume=/tmp/.X11-unix:/tmp/.X11-unix"

# Start docker image
xhost +local:docker

docker  run -d -it --rm \
$VOLUMES \
$ENVS \
$VISUAL \
--privileged \
--runtime=nvidia \
--gpus $GPU_ID \
--net=host \
--shm-size=16G \
--workdir=/VCN \
darrenjkt/pointr:v1.0