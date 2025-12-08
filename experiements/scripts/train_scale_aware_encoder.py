from rfdetr import RFDETRBase

# Initialize RF-DETR Base model with Scale-Aware Encoder enabled
model = RFDETRBase(
    enable_scale_aware_encoder=True
)

# Train with a few epochs/steps to verify flow
model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=12,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/scale_aware_encoder',
)
