#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR Experiment: Scale-Aware Query Grouping + Dynamic Multi-Scale Gating
# ------------------------------------------------------------------------
# This experiment combines two innovations:
# 1. Scale-Aware Query Grouping: Hard-masks queries to attend to specific feature levels
#    based on predicted object scale (small/medium/large)
# 2. Dynamic Multi-Scale Gating: Soft-weights feature levels dynamically per query
# ------------------------------------------------------------------------

from rfdetr import RFDETRBase

print("=" * 80)
print("Experiment: Scale-Aware Query Grouping + Dynamic Multi-Scale Gating")
print("=" * 80)

# Initialize RF-DETR Base model with both innovations
model = RFDETRBase(
    projector_scale=["P3", "P4", "P5"],  # Multi-scale feature pyramid (required!)
    enable_scale_aware_query_grouping=True,  # Innovation 1: Scale-aware hard masking
    scale_aware_num_bins=3,  # 3 bins: small, medium, large
    enable_dynamic_multiscale_gating=True,  # Innovation 2: Dynamic soft gating
    gating_temperature=1.0,  # Softmax temperature for gating weights
    # Exclude mismatched weights due to architecture change
    pretrain_exclude_keys=[
        "backbone.0.projector*",         # Projector shape changed (1 level -> 3 levels)
        "transformer.decoder.layers*",   # Decoder cross-attn shape changed
    ]
)

# Train on RSOD dataset (COCO format)
model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=20,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/e2-scale_aware_dynamic_gating_p345',
    tensorboard=True,
    freeze_encoder=True,  # Freeze backbone for fair comparison
)
