
from rfdetr import RFDETRBase

# Initialize RF-DETR Base model
model = RFDETRBase()

# Train baseline with 12 epochs on RSOD dataset (COCO format)
model.train(
    dataset_file='coco',  # Specify COCO format (not Roboflow)
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',  # Required by TrainConfig
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',  # Path to COCO dataset
    epochs=12,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/baseline_debug',
)