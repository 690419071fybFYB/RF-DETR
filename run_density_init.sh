#!/bin/bash
# Run Density-Guided Query Initialization Training
# Usage: ./run_density_init.sh

set -e

# Activate virtual environment
source ~/envs/torch-cuda/bin/activate

# Navigate to project root
cd /home/fyb/mydir/rf-detr

echo "Starting Density-Guided Query Initialization Training..."
python experiements/scripts/train_density_init.py
