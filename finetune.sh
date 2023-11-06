#! /usr/bin/bash

for dim in 1024 0512 0256 0128 0064 0032 0016 0008 0004 0002
do
    # Make output directory and move checkpoint 'meru_vit_s.pth' there.
    mkdir -p output/meru_vit_small_$dim
    echo meru_vit_s.pth > output/meru_vit_small_$dim/last_checkpoint.txt
    cp checkpoints/meru_vit_s.pth output/meru_vit_small_$dim/meru_vit_s.pth

    # Train model.
    python scripts/train.py --config configs/train_meru_vit_s.py --proj-layer-only $dim --output-dir ./output/meru_vit_small_$dim --resume --num-gpus 2 --save train.num_iterations=130000
done