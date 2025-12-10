"""
验证 Density Sampling Offset Modulation 功能的训练脚本

在 Density Init 的基础上启用密度采样偏移调制
"""
from rfdetr import RFDETRBase
from torchinfo import summary

# 实验：Density Init + Density Sampling Offset
model = RFDETRBase(
    enable_density_init=True,
    enable_small_object_query_boost=False,
    enable_density_sampling_offset=True,
    density_sampling_offset_scale=0.1,
    density_loss_coef=1.0,
)

summary(model.model.model, input_size=(1, 3, 560, 560), depth=7)

model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=12,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/density_sampling_offset',
)
