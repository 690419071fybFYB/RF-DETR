#!/bin/bash
# Verify Spectral Density Loss
# Usage: ./verify_spectral_loss.sh

set -e

# Activate virtual environment
source ~/envs/torch-cuda/bin/activate

# Navigate to project root
cd /home/fyb/mydir/rf-detr

# Define output and log directory
LOG_DIR="experiements/results/debug_spectral_loss"
mkdir -p "$LOG_DIR"

echo "Starting Spectral Density Loss verification..."
echo "Logs will be saved to $LOG_DIR/verify.log"

# Run in background with nohup
nohup python experiements/scripts/train_spectral_loss.py > "$LOG_DIR/verify.log" 2>&1 &

echo "Verification started. PID: $!"

# Wait for a few seconds to ensure process starts
sleep 5

# Check if process is still running
if ps -p $! > /dev/null; then
   echo "Process is running."
   echo "Tailing log file (Ctrl+C to stop tailing, process will continue):"
   tail -f "$LOG_DIR/verify.log"
else
   echo "Process failed to start. Checking log:"
   cat "$LOG_DIR/verify.log"
   exit 1
fi
