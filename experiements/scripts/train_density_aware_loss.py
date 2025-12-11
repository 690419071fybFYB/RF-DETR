"""
验证 Density-Aware Loss Weighting 功能

在 Density Init 基础上启用密度感知损失加权
使用更平稳的训练配置
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
    # 更平稳的训练配置
    lr=1e-4,
    lr_scheduler='cosine',       # 余弦退火调度
    lr_min_factor=0.01,          # 最低衰减到 1%
    warmup_epochs=2,             # 预热 2 个 epoch
    clip_max_norm=0.05,          # 更小的梯度裁剪
    use_ema=True,                # 启用 EMA
    ema_decay=0.9998,            # EMA 衰减系数
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/density_aware_loss',
)
