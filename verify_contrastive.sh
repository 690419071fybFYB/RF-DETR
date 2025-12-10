#!/bin/bash
source /home/fyb/envs/torch-cuda/bin/activate
cd /home/fyb/mydir/rf-detr

python experiements/scripts/verify_contrastive_loss.py \
    2>&1 | tee experiements/results/verify_contrastive/log.txt
