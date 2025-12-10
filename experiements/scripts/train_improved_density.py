"""
验证改进的密度预测模块 (UNet-based)

使用 ImprovedDensityPredictor 替代原始简单 CNN
"""
from rfdetr import RFDETRBase
from torchinfo import summary

# 实验：Improved Density Init (UNet)
model = RFDETRBase(
    enable_density_init=True,
    enable_improved_density=True,  # 使用 UNet 密度预测器
    density_adaptive_threshold=True,  # 使用自适应阈值
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
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/improved_density_init',
)
