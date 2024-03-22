python train.py \
    --benchmark \
    --masking-strategy vchunk 15 \
    --precision 16-mixed \
    --training-noise-std 0.05 \
    --min-epochs 15 \
    --max-epochs 20 \
    --train-path /workspace/data/train.hdf5 \
    --val-path /workspace/data/val.hdf5 \
    --test-path /workspace/data/test.hdf5 \
    --train-batch-size 168 \
    --val-batch-size 1024

python train.py \
    --benchmark \
    --masking-strategy random 60 \
    --precision 16-mixed \
    --training-noise-std 0.05 \
    --min-epochs 15 \
    --max-epochs 20 \
    --train-path /workspace/data/train.hdf5 \
    --val-path /workspace/data/val.hdf5 \
    --test-path /workspace/data/test.hdf5 \
    --train-batch-size 168 \
    --val-batch-size 1024