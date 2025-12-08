#!/bin/bash
# ------------------------------------------------------------------------
# Shell script to run query repulsion experiment in background
# ------------------------------------------------------------------------

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/train_query_repulsion.py"
LOG_DIR="/home/fyb/mydir/rf-detr/experiements/logs"
LOG_FILE="${LOG_DIR}/query_repulsion_$(date +%Y%m%d_%H%M%S).log"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Print info
echo "=========================================="
echo "Starting Query Repulsion Training"
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
