from rfdetr import RFDETRBase

# Initialize RF-DETR Base model with Density Initialization AND Sampling Modulation
model = RFDETRBase(
    enable_density_init=True,
    density_loss_coef=2.0,
    enable_density_guided_sampling_modulation=True,
    density_modulation_scale=1.0
)

# Verify training with 1 epoch
model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=1,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/debug_density_sampling',
)
