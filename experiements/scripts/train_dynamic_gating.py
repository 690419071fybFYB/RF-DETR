#!/usr/bin/env python3
"""
Train RF-DETR with Scale-Aware Query Grouping AND Dynamic Multi-Scale Gating on RSOD dataset.

This script enables both complementary features:
1. Scale-Aware Query Grouping: Predicts scale bins and applies hard masking to feature levels
2. Dynamic Multi-Scale Gating: Predicts soft weights for adaptive feature level fusion

The combination provides both coarse-grained (grouping) and fine-grained (gating) control
over multi-scale feature attention.

Usage:
    python experiements/scripts/train_dynamic_gating.py
"""

from rfdetr import RFDETRBase

# Initialize RF-DETR Base model with both Scale-Aware Grouping AND Dynamic Gating
model = RFDETRBase(
    enable_scale_aware_query_grouping=True,  # Enable scale-aware hard masking
    scale_aware_num_bins=3,  # 3 bins: small, medium, large
    enable_dynamic_multiscale_gating=True,  # Enable dynamic soft gating
    gating_temperature=1.0,  # Softmax temperature for gating weights
)

# Train on RSOD dataset (COCO format)
model.train(
    dataset_file='coco',  # Specify COCO format
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=12,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/e2_dynamic_multiscale_gating',
    tensorboard=True,
    freeze_encoder=True,  # Freeze backbone for fair comparison
)
