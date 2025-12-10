#!/usr/bin/env python3
"""
测试密度图可视化功能的脚本
"""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')

from rfdetr.models import build_model
from rfdetr.config import RFDETRBaseConfig
from rfdetr.util.density_visualizer import DensityVisualizer
from rfdetr.util.misc import NestedTensor


def test_density_visualization():
    """测试密度图可视化功能"""

    print("正在测试密度图可视化功能...")

    # 配置模型，启用密度相关功能
    args = RFDETRBaseConfig(
        num_classes=80,
        enable_density_init=True,
        enable_density_augmented_cross_attn=True,
        visualize_density=True,
        density_vis_max_samples=2,
        density_vis_interval=10
    )

    # 构建模型
    print("正在构建模型...")
    model, criterion, postprocessors = build_model(args)
    model.eval()

    # 创建可视化器
    output_dir = Path("test_density_vis")
    output_dir.mkdir(exist_ok=True)

    visualizer = DensityVisualizer(
        output_dir=str(output_dir),
        max_samples=2
    )

    # 创建模拟数据
    batch_size = 2
    img_h, img_w = 560, 560

    # 创建随机图像
    images = torch.randn(batch_size, 3, img_h, img_w)
    images = images.clamp(0, 1)  # 确保在[0,1]范围内

    # 创建NestedTensor
    masks = torch.zeros(batch_size, img_h, img_w, dtype=torch.bool)
    samples = NestedTensor(images, masks)

    # 创建模拟的目标（边界框和标签）
    targets = []
    for i in range(batch_size):
        # 每张图片2-4个随机边界框
        num_boxes = np.random.randint(2, 5)

        # 生成随机边界框 (cx, cy, w, h) 格式，归一化到[0,1]
        boxes = torch.rand(num_boxes, 4)
        # 限��宽高在合理范围内
        boxes[:, 2] = boxes[:, 2] * 0.3 + 0.1  # width: 0.1-0.4
        boxes[:, 3] = boxes[:, 3] * 0.3 + 0.1  # height: 0.1-0.4

        # 确保边界框在图像内
        boxes[:, 0] = torch.clamp(boxes[:, 0], boxes[:, 2]/2, 1 - boxes[:, 2]/2)
        boxes[:, 1] = torch.clamp(boxes[:, 1], boxes[:, 3]/2, 1 - boxes[:, 3]/2)

        # 随机类别标签
        labels = torch.randint(0, 80, (num_boxes,))

        targets.append({
            'boxes': boxes,
            'labels': labels
        })

    print("正在运行模型前向传播...")

    # 运行模型
    with torch.no_grad():
        outputs = model(samples, targets)

    print("模型输出键:", list(outputs.keys()))

    # 检查是否有密度图输出
    if 'pred_density' in outputs:
        print(f"找到密度图输出，形状: {outputs['pred_density'].shape}")
        print(f"密度图数值范围: [{outputs['pred_density'].min().item():.4f}, {outputs['pred_density'].max().item():.4f}]")
    else:
        print("警告: 模型输出中没有找到密度图")
        print("可用的输出键:", list(outputs.keys()))

    # 使用可视化器
    print("正在创建可视化...")

    class_names = {i: f"class_{i}" for i in range(80)}

    visualizer.visualize_batch(
        images=images,
        targets=targets,
        model_outputs=outputs,
        epoch=0,
        step=0,
        class_names=class_names
    )

    # 保存统计信息
    density_maps = visualizer.extract_density_maps(outputs)
    if density_maps:
        visualizer.save_density_statistics(density_maps, epoch=0, step=0)
        print("密度图统计信息已保存")

    print(f"\n测试完成！可视化结果保存在: {output_dir.absolute()}")
    print("请检查生成的PNG文件以查看密度图可视化效果。")

    # 打印一些统计信息
    if 'pred_density' in outputs:
        density = outputs['pred_density']
        print(f"\n密度图统计:")
        print(f"  批次大小: {density.shape[0]}")
        print(f"  密度图尺寸: {density.shape[2]} x {density.shape[3]}")
        print(f"  平均密度值: {density.mean().item():.4f}")
        print(f"  最大密度值: {density.max().item():.4f}")
        print(f"  最小密度值: {density.min().item():.4f}")


if __name__ == "__main__":
    test_density_visualization()