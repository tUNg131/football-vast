#!/bin/bash

# Update package list and install Git
# pip install h5py lightning tensorboard gdown

# Download data
# gdown 1BsysF9kXg0WWWQ7RYufecsiOGLyZRxbD --folder --output /workspace/data

# git clone
# git clone https://github.com/tUNg131/football-vast.git /workspace/src

python /workspace/src/main.py train \
    --train-path /workspace/data/train.hdf5 \
    --val-path /workspace/data/val.hdf5 \
    --batch-size 128 \
    --version test \
    --max-epochs 1 \
    --fast-dev-run \
    --precision transformer-engine
    /

# tensorboard --logdir /workspace/tb_logs --port=8008

# ssh -L 8008:127.0.0.1:8008 -fN -p 51324 root@213.181.122.2