#!/bin/bash

echo "========================================"
echo "开始消融实验对比: Baseline vs Density Init"
echo "========================================"

# 任务1: Density Init
echo ""
echo "[任务 1/2] 开始训练 Density Init..."
echo "日志: /home/fyb/mydir/rf-detr/experiements/ablation_results/log_density_init_6bs.txt"
python /home/fyb/mydir/rf-detr/experiements/scripts/ablation_density_init.py > /home/fyb/mydir/rf-detr/experiements/ablation_results/log_density_init_6bs.txt 2>&1
echo "[任务 1/2] Density Init 训练完成!"

# 任务2: Baseline
echo ""
echo "[任务 2/2] 开始训练 Baseline..."
echo "日志: /home/fyb/mydir/rf-detr/experiements/ablation_results/log_baseline_6bs.txt"
python /home/fyb/mydir/rf-detr/experiements/scripts/ablation_baseline.py > /home/fyb/mydir/rf-detr/experiements/ablation_results/log_baseline_6bs.txt 2>&1
echo "[任务 2/2] Baseline 训练完成!"

echo ""
echo "========================================"
echo "所有消融实验完成!"
echo "========================================"