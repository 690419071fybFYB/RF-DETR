from rfdetr import RFDETRBase

# Initialize RF-DETR Base model with BOTH strategies enabled
model = RFDETRBase(
    enable_density_init=True,
    density_loss_coef=2.0,
    enable_scale_aware_encoder=True
)

# Train on RSOD dataset
model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=12,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/combined_strategies',
)
