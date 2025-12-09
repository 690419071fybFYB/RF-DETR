#!/bin/bash
# Run Spectral Density Loss (Full 12-Epoch Experiment)
# Usage: ./run_spectral_loss.sh

set -e

# Activate virtual environment
source ~/envs/torch-cuda/bin/activate

# Navigate to project root
cd /home/fyb/mydir/rf-detr

# Define output and log directory
OUTPUT_DIR="/home/fyb/mydir/rf-detr/experiements/results/spectral_density_loss"
mkdir -p "$OUTPUT_DIR"

echo "Starting Spectral Density Loss Full Experiment..."
echo "Output Directory: $OUTPUT_DIR"

nohup python experiements/scripts/run_spectral_loss_full.py > "$OUTPUT_DIR/train.log" 2>&1 &

echo "Experiment started. PID: $!"
echo "Logs: $OUTPUT_DIR/train.log"
