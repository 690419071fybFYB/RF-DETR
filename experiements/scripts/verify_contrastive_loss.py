from rfdetr import RFDETRBase

# Initialize RF-DETR Base model with Contrastive Density Regularization
# Requires enable_density_init=True
model = RFDETRBase(
    enable_density_init=True,
    density_loss_coef=1.0,
    enable_small_object_query_boost=False, # Example: Disable boost
    soqb_boost_factor=2.0 
)

# Train with 1 epoch for verification
model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=1,
    batch_size=2, # Small batch for quick debug
    grad_accum_steps=1,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/verify_contrastive',
)
