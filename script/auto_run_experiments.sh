#!/bin/bash

# PID to wait for
TARGET_PID=3245833

echo "=========================================================="
echo "Automated Innovation Experiment Script"
echo "Waiting for current S-FPN training (PID: $TARGET_PID) to finish..."
echo "=========================================================="

# Wait loop
while kill -0 $TARGET_PID 2> /dev/null; do
    sleep 30
done

echo "S-FPN Training finished."
echo "Saving S-FPN logs..."
git add script/RSOD_SFPN
git commit -m "chore: Save S-FPN training logs"

# ---------------------------------------------------------
# Innovation 1: F-DQS
# ---------------------------------------------------------
echo "=========================================================="
echo "Switching to feat/F-DQS..."
echo "=========================================================="
git checkout feat/F-DQS

echo "Starting F-DQS Training..."
/home/fyb/envs/torch-cuda/bin/python script/train_FDQS.py

echo "F-DQS Training finishes."
echo "Saving F-DQS logs..."
git add script/RSOD_FDQS
git commit -m "chore: Save F-DQS training logs"

# ---------------------------------------------------------
# Innovation 2: CFD-Head
# ---------------------------------------------------------
echo "=========================================================="
echo "Switching to feat/CFD-Head..."
echo "=========================================================="
git checkout feat/CFD-Head

echo "Starting CFD-Head Training..."
/home/fyb/envs/torch-cuda/bin/python script/train_CFDHead.py

echo "CFD-Head Training finishes."
echo "Saving CFD-Head logs..."
git add script/RSOD_CFDHead
git commit -m "chore: Save CFD-Head training logs"

echo "=========================================================="
echo "ALL EXPERIMENTS COMPLETED."
echo "=========================================================="
