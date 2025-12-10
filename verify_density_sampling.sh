#!/bin/bash
source /home/fyb/envs/torch-cuda/bin/activate
cd /home/fyb/mydir/rf-detr

python experiements/scripts/train_density_sampling.py \
    2>&1 | tee experiements/results/debug_density_sampling/log.txt
