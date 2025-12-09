from rfdetr import RFDETRBase
import os

output_dir = "/home/fyb/mydir/rf-detr/experiements/results/hybrid_selection"
os.makedirs(output_dir, exist_ok=True)

model = RFDETRBase(
    enable_density_init=True,
    density_loss_coef=2.0,
    enable_scale_aware_encoder=True,
    hybrid_selection_alpha=0.5,
)

model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=12,  # Full schedule
    batch_size=4, # Use standard batch size or 4 to match baseline? Baseline was 4.
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=output_dir,
)
