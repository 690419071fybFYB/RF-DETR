#!/bin/bash
source /home/fyb/envs/torch-cuda/bin/activate
cd /home/fyb/mydir/rf-detr

python experiements/scripts/run_contrastive_density.py \
    2>&1 | tee experiements/results/contrastive_density/log.txt
