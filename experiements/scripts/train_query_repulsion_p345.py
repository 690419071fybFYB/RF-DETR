#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR Experiment: Query Repulsion Loss
# ------------------------------------------------------------------------
# This experiment tests the Query Repulsion Loss which:
# 1. Identifies pairs of queries whose predicted boxes have high IoU (overlapping)
# 2. Applies margin-based hinge loss to push their embeddings apart
# 3. Encourages diversity among queries and reduces duplicate detections
# ------------------------------------------------------------------------

from rfdetr import RFDETRBase

print("=" * 80)
print("Experiment: Query Repulsion Loss")
print("=" * 80)

# Initialize RF-DETR Base model with Query Repulsion Loss
model = RFDETRBase(
    projector_scale=["P3", "P4", "P5"],  # Multi-scale feature pyramid
    enable_query_repulsion_loss=True,  # Innovation: Query repulsion loss
    query_repulsion_loss_weight=1.0,  # Weight for the repulsion loss term
    query_repulsion_margin=0.5,  # L2 distance margin for hinge loss
    query_repulsion_iou_threshold=0.3,  # IoU threshold to identify close boxes
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
    epochs=12,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/e3-query_repulsion_p345',
    tensorboard=True,
    freeze_encoder=True,  # Freeze backbone for fair comparison
)
