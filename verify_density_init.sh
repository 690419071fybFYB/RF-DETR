#!/bin/bash
# Verify Density-Guided Query Initialization
# Usage: ./verify_density_init.sh

set -e

# Activate virtual environment
source ~/envs/torch-cuda/bin/activate

# Create output directory
OUTPUT_DIR="/home/fyb/mydir/rf-detr/experiements/results/debug_density_init"
mkdir -p "$OUTPUT_DIR"

# Navigate to project root
cd /home/fyb/mydir/rf-detr

# Run a quick training job (1 epoch, few batches) to check for errors and loss presence
echo "Starting Density Init verification..."
echo "Output directory: $OUTPUT_DIR"

python experiements/scripts/train_baseline.py \
    --output_dir "$OUTPUT_DIR" \
    --epochs 1 \
    --batch_size 2 \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Verification complete. Check log for 'loss_density'."
