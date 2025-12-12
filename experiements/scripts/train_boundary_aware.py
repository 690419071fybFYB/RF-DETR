"""
Boundary-Aware Query Refinement 训练脚本

在 Density Init 基础上添加边界感知查询精炼
目标：提升 bbox 回归精度（尤其是 AP@IoU=0.75）
"""
import sys
sys.path.insert(0, '/home/fyb/mydir/rf-detr')

from rfdetr import RFDETRBase
from torchinfo import summary
# 创建模型
model = RFDETRBase(
    # 启用 Density Init (已验证有效)
    enable_density_init=True,
    density_loss_coef=1.0,
    # 启用边界感知查询精炼 (新功能)
    enable_boundary_refinement=True,
    boundary_loss_coef=1.0,
    boundary_refinement_scale=0.05,  # 精炼幅度
    # 其他
    enable_small_object_query_boost=False,  # 关闭 (消融实验显示无效)
)
summary(model.model.model, input_size=(1, 3, 560, 560),depth=7)
# 训练配置
model.train(
    dataset_file="coco",  # 使用 COCO 格式
    dataset_dir="/home/fyb/datasets/RSOD_cocoFormat",  # 必需参数
    coco_path="/home/fyb/datasets/RSOD_cocoFormat",  # COCO 格式数据集路径
    epochs=12,
    batch_size=4,
    grad_accum_steps=4,
    output_dir="/home/fyb/mydir/rf-detr/experiements/results/boundary_aware_refinement",
    lr=1e-4,
    # 平稳训练配置
    lr_scheduler='cosine',
    lr_min_factor=0.01,
    warmup_epochs=2,
    clip_max_norm=0.05,
    use_ema=True,
    ema_decay=0.9998,
)
