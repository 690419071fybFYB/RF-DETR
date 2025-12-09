#!/bin/bash
# Verify Band-Pass Spectral Density Loss
# Usage: ./verify_bandpass_spectral.sh

set -e

# Activate virtual environment
source ~/envs/torch-cuda/bin/activate

# Navigate to project root
cd /home/fyb/mydir/rf-detr

# Define output and log directory
LOG_DIR="experiements/results/debug_bandpass_spectral"
mkdir -p "$LOG_DIR"

echo "Starting Band-Pass Spectral Loss verification..."
echo "Logs will be saved to $LOG_DIR/verify.log"

# Run in background with nohup
nohup python experiements/scripts/train_bandpass_spectral.py > "$LOG_DIR/verify.log" 2>&1 &

echo "Verification started. PID: $!"

# Wait
sleep 5

# Check status
if ps -p $! > /dev/null; then
   echo "Process is running."
   echo "Tailing log file (Ctrl+C to stop tailing, process will continue):"
   tail -f "$LOG_DIR/verify.log"
else
   echo "Process failed to start. Checking log:"
   cat "$LOG_DIR/verify.log"
   exit 1
fi
