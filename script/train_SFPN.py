from rfdetr import RFDETRBase

# Initialize model
# Note: The code on this branch (feat/S-FPN) automatically uses the S-FPN logic in MultiScaleProjector
# because we modified the code in place.
model = RFDETRBase(use_dynamic_query=True)

dataset = "/home/fyb/datasets/RSOD_cocoFormat"
output_dir = "/home/fyb/mydir/rf-detr/script/RSOD_SFPN"

print(f"Starting S-FPN training with dataset: {dataset}")
print(f"Output directory: {output_dir}")

model.train(
    dataset_dir=dataset,
    dataset_file="coco",
    coco_path=dataset,
    epochs=200,
    batch_size=6,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=output_dir,
    early_stopping=True,
    early_stopping_patience=5,
)
