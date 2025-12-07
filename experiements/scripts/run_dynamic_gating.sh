#!/bin/bash
# Run Dynamic Multi-Scale Gating training in background
# Usage: ./experiements/scripts/run_dynamic_gating.sh

set -e

# Activate virtual environment
source ~/envs/torch-cuda/bin/activate

# Create output directory
OUTPUT_DIR="/home/fyb/mydir/rf-detr/experiements/results/e2_dynamic_multiscale_gating"
mkdir -p "$OUTPUT_DIR"

# Navigate to project root
cd /home/fyb/mydir/rf-detr

# Run training in background with logging
echo "Starting Dynamic Multi-Scale Gating training..."
echo "This experiment builds on Scale-Aware Query Grouping (e1)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $OUTPUT_DIR/training.log"

nohup python experiements/scripts/train_dynamic_gating.py \
    > "$OUTPUT_DIR/training.log" 2>&1 &

echo "Training started in background with PID: $!"
echo "To monitor: tail -f $OUTPUT_DIR/training.log"
