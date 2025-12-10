"""
验证 Density-Augmented Cross-Attention 功能的训练脚本

在 Density Init 的基础上启用密度增强交叉注意力
"""
from rfdetr import RFDETRBase

# 实验：Density Init + Density Augmented Cross-Attention
model = RFDETRBase(
    enable_density_init=True,
    enable_small_object_query_boost=False,  # 先单独测试密度增强
    enable_density_augmented_cross_attn=True,
    density_augment_scale_factor=0.1,
)

model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=12,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/density_augmented_cross_attn',
)
