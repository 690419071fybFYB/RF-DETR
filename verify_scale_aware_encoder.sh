#!/bin/bash
# Verify Scale-Aware Encoder
# Usage: ./verify_scale_aware_encoder.sh

set -e

# Activate virtual environment
source ~/envs/torch-cuda/bin/activate

# Navigate to project root
cd /home/fyb/mydir/rf-detr

echo "Starting Scale-Aware Encoder verification..."
LOG_DIR="/home/fyb/mydir/rf-detr/experiements/results/debug_scale_aware_encoder"
mkdir -p "$LOG_DIR"

python experiements/scripts/train_scale_aware_encoder.py > "$LOG_DIR/verify.log" 2>&1

echo "Verification complete. Check $LOG_DIR/verify.log for details."
cat "$LOG_DIR/verify.log" | tail -n 20
