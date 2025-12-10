#!/bin/bash
source /home/fyb/envs/torch-cuda/bin/activate
cd /home/fyb/mydir/rf-detr

# Create output directory for logs
RESULTS_DIR="experiements/ablation_results"
mkdir -p "$RESULTS_DIR"

echo "========================================================"
echo "Starting Ablation Study"
echo "Results will be saved to $RESULTS_DIR"
echo "========================================================"

# 1. Baseline
echo "[1/4] Running Baseline Experiment..."
python experiements/scripts/ablation_baseline.py \
    2>&1 | tee "$RESULTS_DIR/log_baseline.txt"
echo "Baseline completed."

# 2. Density Init Only
echo "[2/4] Running Density Init Experiment..."
python experiements/scripts/ablation_density_init.py \
    2>&1 | tee "$RESULTS_DIR/log_density_init.txt"
echo "Density Init completed."

# 3. SOQB Only
echo "[3/4] Running SOQB Experiment..."
python experiements/scripts/ablation_soqb.py \
    2>&1 | tee "$RESULTS_DIR/log_soqb.txt"
echo "SOQB completed."

# 4. Combined
echo "[4/4] Running Combined Experiment..."
python experiements/scripts/ablation_combined.py \
    2>&1 | tee "$RESULTS_DIR/log_combined.txt"
echo "Combined completed."

echo "========================================================"
echo "All ablation experiments completed."
echo "========================================================"
