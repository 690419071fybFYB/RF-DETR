#!/bin/bash
# Run Scale-Aware Query Grouping training in background
# Usage: ./experiements/scripts/run_scale_aware_query_grouping.sh

set -e

# Activate virtual environment
source ~/envs/torch-cuda/bin/activate

# Create output directory
OUTPUT_DIR="/home/fyb/mydir/rf-detr/experiements/results/e1_scale_aware_query_grouping"
mkdir -p "$OUTPUT_DIR"

# Navigate to project root
cd /home/fyb/mydir/rf-detr

# Run training in background with logging
echo "Starting Scale-Aware Query Grouping training..."
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $OUTPUT_DIR/training.log"

nohup python experiements/scripts/train_scale_aware_query_grouping.py \
    > "$OUTPUT_DIR/training.log" 2>&1 &

echo "Training started in background with PID: $!"
echo "To monitor: tail -f $OUTPUT_DIR/training.log"
