from rfdetr import RFDETRBase
import os

output_dir = "/home/fyb/mydir/rf-detr/experiements/results/spectral_density_loss"
os.makedirs(output_dir, exist_ok=True)

# Full Experiment: Density Init + Spectral Density Loss
model = RFDETRBase(
    enable_density_init=True,
    density_loss_coef=2.0,
    enable_spectral_density_loss=True,
    spectral_density_loss_coef=0.1,
    spectral_density_band_start=0.05,
    spectral_density_band_end=0.3,
)

model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=12,  # Full schedule
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=output_dir,
)
