#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR Baseline Experiment: Multi-Scale (P3+P4+P5)
# ------------------------------------------------------------------------
# This script trains RF-DETR Base with 3 feature levels (P3, P4, P5)
# WITHOUT any innovations (no scale-aware grouping, no dynamic gating, no query repulsion)
# This serves as the baseline for comparison.
# ------------------------------------------------------------------------

from rfdetr import RFDETRBase

print("=" * 80)
print("Baseline Experiment: RF-DETR Base with P3+P4+P5 (No Innovations)")
print("=" * 80)

# Initialize RF-DETR Base model with 3 feature levels
model = RFDETRBase(
    projector_scale=["P3", "P4", "P5"],  # Multi-scale feature pyramid
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
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/e1-baseline_p345',
    tensorboard=True,
    freeze_encoder=True,  # Freeze backbone for fair comparison
)
