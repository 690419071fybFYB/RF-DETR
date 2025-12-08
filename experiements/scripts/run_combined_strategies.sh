#!/bin/bash
# Run Combined Strategies Training (Density Init + Scale-Aware Encoder)
# Usage: ./experiements/scripts/run_combined_strategies.sh

set -e

# Activate virtual environment
source ~/envs/torch-cuda/bin/activate

# Navigate to project root
cd /home/fyb/mydir/rf-detr

# Define output and log directory
LOG_DIR="/home/fyb/mydir/rf-detr/experiements/results/combined_strategies"
mkdir -p "$LOG_DIR"

echo "Starting Combined Strategies Training (Density + Scale-Aware) in background..."
echo "Logs will be saved to $LOG_DIR/train.log"

# Run in background with nohup, redirecting stdout and stderr to log file
nohup python experiements/scripts/train_combined_strategies.py > "$LOG_DIR/train.log" 2>&1 &

echo "Training started. PID: $!"
