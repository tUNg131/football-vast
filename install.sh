#!/bin/bash

# Update package list and install Git
# pip install h5py lightning tensorboard

python /workspace/main.py \
    --train-path ./data/train.hdf5 \
    --val-path ./data/val.hdf5 \
    --batch-size=1 --version test \
    --max-epochs 1 \
    --fast-dev-run \
    --precision transformer-engine
    /

# tensorboard --logdir /workspace/tb_logs --port=8008

# ssh -L 8008:127.0.0.1:8008 -fN -p 51324 root@213.181.122.2