"""
验证 Density-Aware Loss Weighting 功能

在 Density Init 基础上启用密度感知损失加权
"""
from rfdetr import RFDETRBase
from torchinfo import summary

# 实验：Density Init + Density-Aware Loss
model = RFDETRBase(
    enable_density_init=True,
    enable_improved_density=False,  # 使用原始简单 CNN
    enable_density_aware_loss=True,  # 新功能：密度感知损失
    density_aware_alpha=0.5,  # 加权强度
    enable_small_object_query_boost=False,  # 排除 SOQB 干扰
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
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/density_aware_loss',
)
