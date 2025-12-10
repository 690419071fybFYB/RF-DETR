"""
验证 Density Positional Bias Modulation 功能的训练脚本

在 Density Init 的基础上启用密度位置偏置调制
"""
from rfdetr import RFDETRBase

# 实验：Density Init + Density Positional Bias
model = RFDETRBase(
    enable_density_init=True,
    enable_small_object_query_boost=False,
    enable_density_positional_bias=True,
    density_pos_bias_scale=0.1,
    density_loss_coef=1.0,
)

model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=12,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/density_pos_bias',
)
