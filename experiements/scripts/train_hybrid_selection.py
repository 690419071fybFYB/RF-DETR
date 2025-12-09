from rfdetr import RFDETRBase

# Initialize RF-DETR Base model with Hybrid Query Selection
# Only effective if both Scale-Aware Encoder (optional but encouraged) and Density Init are verified,
# but can run with just Density Init.
# Here we test the FULL combination: Scale-Aware + Density + Hybrid Selection
model = RFDETRBase(
    enable_density_init=True,
    density_loss_coef=2.0,
    enable_scale_aware_encoder=True,
    hybrid_selection_alpha=0.5, # 50% Score, 50% Density
)

# Train on RSOD dataset (Quick Verification)
model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=12,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/debug_hybrid_selection',
)
