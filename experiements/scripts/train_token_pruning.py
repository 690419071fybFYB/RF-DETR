from rfdetr import RFDETRBase

# Initialize RF-DETR Base model with Density-Guided Token Pruning enabled
model = RFDETRBase(
    enable_density_init=True,
    density_loss_coef=2.0,
    enable_token_pruning=True,
    pruning_ratio=0.5, # Prune 50%
    pruning_keep_background_ratio=0.1 # Keep 10% of background
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
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/density_token_pruning',
)
