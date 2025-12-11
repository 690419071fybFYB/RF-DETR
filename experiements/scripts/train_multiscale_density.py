"""
验证 Multi-Scale Density Supervision 功能

在 P3/P4/P5 每个层级分别预测密度图
P3 → 小目标, P4 → 中目标, P5 → 大目标
使用更平稳的训练配置
"""
from rfdetr import RFDETRBase
from torchinfo import summary

# 实验：Multi-Scale Density Supervision
model = RFDETRBase(
    enable_density_init=True,
    enable_improved_density=False,  # 使用原始简单 CNN
    enable_multiscale_density=True,  # 新功能：多尺度密度监督
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
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/multiscale_density',
)
