from rfdetr import RFDETR800

model = RFDETR800()

dataset = "/home/fyb/datasets/DIOR_cocoFormat"
output_dir = "/home/fyb/mydir/rf-detr/script/DIOR_800"

model.train(
    dataset_dir=dataset,
    dataset_file="coco",
    coco_path=dataset,
    epochs=30,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=output_dir,
    # 这里可以根据显存调整 resolution/batch size 等，但不要改结构参数，默认就会加载 large 预训权重
    # multi_scale=False,
    # do_random_resize_via_padding=False,
)
