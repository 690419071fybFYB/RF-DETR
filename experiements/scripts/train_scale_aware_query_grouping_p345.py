#!/usr/bin/env python3
"""
Train RF-DETR with Scale-Aware Query Grouping on RSOD dataset.

This script enables the scale-aware query grouping feature which:
1. Predicts the target scale bin (small/medium/large) for each query
2. Masks cross-attention so each query only attends to relevant feature pyramid levels
3. Small objects attend to high-resolution features, large objects to low-resolution

Usage:
    python experiements/scripts/train_scale_aware_query_grouping.py
"""

from rfdetr import RFDETRBase

# Initialize RF-DETR Base model with Scale-Aware Query Grouping enabled
model = RFDETRBase(
    projector_scale=["P3", "P4", "P5"],  # Multi-scale feature pyramid
    enable_scale_aware_query_grouping=True,  # Enable the new feature
    scale_aware_num_bins=3,  # 3 bins: small, medium, large
    # Exclude mismatched weights due to architecture change
    pretrain_exclude_keys=[
        "backbone.0.projector*",         # Projector shape changed (1 level -> 3 levels)
        "transformer.decoder.layers*",   # Decoder cross-attn shape changed
    ]
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
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/e1_scale_aware_query_grouping_p345',
    tensorboard=True,
    freeze_encoder=True,  # Freeze backbone for fair comparison
)
