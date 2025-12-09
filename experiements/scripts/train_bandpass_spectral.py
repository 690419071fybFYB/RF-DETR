from rfdetr import RFDETRBase
import os

# Test Band-Pass Spectral Density Loss
# Base: Density Init + Band-Pass Spectral Loss
model = RFDETRBase(
    enable_density_init=True,
    density_loss_coef=2.0,
    enable_spectral_density_loss=True,
    spectral_density_loss_coef=0.1,
    spectral_density_band_start=0.05,
    spectral_density_band_end=0.3,
)

# Train on RSOD dataset (Quick Verification - 1 epoch)
model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=1,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/debug_bandpass_spectral',
)
