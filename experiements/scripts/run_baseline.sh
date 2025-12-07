#!/bin/bash
# Baseline training script with output logging

# Create output directory
mkdir -p /home/fyb/mydir/rf-detr/experiements/results/baseline

# Run training and save output to log file
nohup /home/fyb/envs/torch-cuda/bin/python /home/fyb/mydir/rf-detr/experiements/scripts/train_baseline.py \
    > /home/fyb/mydir/rf-detr/experiements/results/baseline/training.log 2>&1 &

# Get the process ID
PID=$!

echo "Training started with PID: $PID"
echo "View training log with: tail -f /home/fyb/mydir/rf-detr/experiements/results/baseline/training.log"
echo "Check process status with: ps -p $PID"
