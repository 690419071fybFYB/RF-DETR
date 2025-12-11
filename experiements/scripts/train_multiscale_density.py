"""
验证 Multi-Scale Density Supervision 功能

在 P3/P4/P5 每个层级分别预测密度图
P3 → 小目标, P4 → 中目标, P5 → 大目标
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
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/multiscale_density',
)
