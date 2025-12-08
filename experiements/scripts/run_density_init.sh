#!/bin/bash
# Run Density-Guided Query Initialization Training
# Usage: ./experiements/scripts/run_density_init.sh

set -e

# Activate virtual environment
source ~/envs/torch-cuda/bin/activate

# Navigate to project root
cd /home/fyb/mydir/rf-detr

# Define output and log directory
LOG_DIR="/home/fyb/mydir/rf-detr/experiements/results/density_init"
mkdir -p "$LOG_DIR"

echo "Starting Density-Guided Query Initialization Training in background..."
echo "Logs will be saved to $LOG_DIR/train.log"

# Run in background with nohup, redirecting stdout and stderr to log file
nohup python experiements/scripts/train_density_init.py > "$LOG_DIR/train.log" 2>&1 &

echo "Training started. PID: $!"
