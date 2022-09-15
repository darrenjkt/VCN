#!/bin/bash

# Edit these paths
DATA_PATH="/user/data"
CODE_PATH="/user/code/VCN"
GPU_ID="0"

ENVS="  --env=NVIDIA_VISIBLE_DEVICES=$GPU_ID
        --env=CUDA_VISIBLE_DEVICES=$GPU_ID
        --env=NVIDIA_DRIVER_CAPABILITIES=all"

# Modify these paths as necessary to mount the data
VOLUMES="       --volume=$DATA_PATH:/VCN/data
                --volume=$CODE_PATH:/VCN"

# Setup environment for pop-up visualization of point clouds
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
darrenjkt/vcn:1.0
