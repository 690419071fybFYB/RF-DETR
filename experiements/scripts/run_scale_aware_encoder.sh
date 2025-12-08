#!/bin/bash
# Run Scale-Aware Encoder Training
# Usage: ./experiements/scripts/run_scale_aware_encoder.sh

set -e

# Activate virtual environment
source ~/envs/torch-cuda/bin/activate

# Navigate to project root
cd /home/fyb/mydir/rf-detr

# Define output and log directory
LOG_DIR="/home/fyb/mydir/rf-detr/experiements/results/scale_aware_encoder"
mkdir -p "$LOG_DIR"

echo "Starting Scale-Aware Encoder Training in background..."
echo "Logs will be saved to $LOG_DIR/train.log"

# Run in background with nohup, redirecting stdout and stderr to log file
nohup python experiements/scripts/train_scale_aware_encoder.py > "$LOG_DIR/train.log" 2>&1 &

echo "Training started. PID: $!"
