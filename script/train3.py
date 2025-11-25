from rfdetr import RFDETRBase

model = RFDETRBase()

dataset="/home/fyb/datasets/DIOR_cocoFormat"
output_dir="/home/fyb/mydir/rf-detr/script/DIOR3"
model.train(
    dataset_dir=dataset,
    dataset_file="coco",
    coco_path=dataset,
    epochs=50,
    batch_size=6,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=output_dir,
)
 