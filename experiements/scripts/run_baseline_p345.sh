#!/bin/bash
# ------------------------------------------------------------------------
# Shell script to run baseline experiment in background with logging
# ------------------------------------------------------------------------

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/train_baseline_p345.py"
LOG_DIR="/home/fyb/mydir/rf-detr/experiements/logs"
LOG_FILE="${LOG_DIR}/baseline_p345_$(date +%Y%m%d_%H%M%S).log"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Print info
echo "=========================================="
echo "Starting Baseline P3+P4+P5 Training"
echo "=========================================="
echo "Script: ${PYTHON_SCRIPT}"
echo "Log file: ${LOG_FILE}"
echo "=========================================="

# Run training in background and redirect output to log file
nohup /home/fyb/envs/torch-cuda/bin/python "${PYTHON_SCRIPT}" > "${LOG_FILE}" 2>&1 &

# Get the process ID
PID=$!
echo "Training started with PID: ${PID}"
echo "To monitor progress, run: tail -f ${LOG_FILE}"
echo "To stop training, run: kill ${PID}"
