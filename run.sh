#!/bin/bash

# Update package list and install Git
# pip install h5py lightning tensorboard gdown git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Download data
# gdown 1BsysF9kXg0WWWQ7RYufecsiOGLyZRxbD --folder --output /workspace/data

# git clone
# git clone https://github.com/tUNg131/football-vast.git /workspace/src

python /workspace/src/main.py train \
    --train-path /workspace/data/train.hdf5 \
    --val-path /workspace/data/val.hdf5 \
    --batch-size 128 \
    --version random30_lr0d0015_lamb4_4x \
    --random random30 \
    --max-epochs 40 \
    --precision 16-mixed \
    --fast-dev-run

# tensorboard --logdir /workspace/tb_logs --port=8008

# ssh -L 8008:127.0.0.1:8008 -fN -p 51324 root@213.181.122.2

python /workspace/src/main.py train --train-path /workspace/data/train.hdf5 --val-path /workspace/data/val.hdf5 --batch-size 128 --version random30_lr0d0015_lamb4_4x --random random30 --max-epochs 40 --precision 16-mixed --fast-dev-run