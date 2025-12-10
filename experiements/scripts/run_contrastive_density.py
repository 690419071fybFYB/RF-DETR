from rfdetr import RFDETRBase

# Initialize RF-DETR Base model with Contrastive Density Regularization
model = RFDETRBase(
    enable_density_init=True,
    density_loss_coef=1.0,
    enable_contrastive_density_loss=True,
    contrastive_density_loss_coef=1.0,
    contrastive_density_temperature=0.1
)

# Train with 12 epochs
model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=12,
    batch_size=4, 
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/contrastive_density',
)
