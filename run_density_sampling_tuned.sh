#!/bin/bash
source /home/fyb/envs/torch-cuda/bin/activate
cd /home/fyb/mydir/rf-detr

python experiements/scripts/run_density_sampling_tuned.py \
    2>&1 | tee experiements/results/density_sampling_modulation_tuned/log.txt
