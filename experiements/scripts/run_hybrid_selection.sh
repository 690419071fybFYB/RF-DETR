#!/bin/bash
# Run Hybrid Score-Density Query Selection (Full Experiment)
# Usage: ./run_hybrid_selection.sh

set -e

# Activate virtual environment
source ~/envs/torch-cuda/bin/activate

# Navigate to project root
cd /home/fyb/mydir/rf-detr

# Define output and log directory
OUTPUT_DIR="/home/fyb/mydir/rf-detr/experiements/results/hybrid_selection"
mkdir -p "$OUTPUT_DIR"

echo "Starting Hybrid Selection Experiment..."
echo "Output Directory: $OUTPUT_DIR"

# Run training in background
# We can reuse train_hybrid_selection.py but override output_dir and epochs
# Or just call the module directly.
# Let's modify train_hybrid_selection.py slightly via command line overrides if possible?
# The script overrides args internally. Ideally we should create a proper script or edit the existing one to be flexible.
# But for now, let's just create a new python file for full run to be safe and explicit.

cat <<EOF > experiements/scripts/run_hybrid_selection_full.py
from rfdetr import RFDETRBase
import os

output_dir = "/home/fyb/mydir/rf-detr/experiements/results/hybrid_selection"
os.makedirs(output_dir, exist_ok=True)

model = RFDETRBase(
    enable_density_init=True,
    density_loss_coef=2.0,
    enable_scale_aware_encoder=True,
    hybrid_selection_alpha=0.5,
)

model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=12,  # Full schedule
    batch_size=4, # Use standard batch size or 4 to match baseline? Baseline was 4.
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=output_dir,
)
EOF

nohup python experiements/scripts/run_hybrid_selection_full.py > "$OUTPUT_DIR/train.log" 2>&1 &

echo "Experiment started. PID: $!"
echo "Logs: $OUTPUT_DIR/train.log"
