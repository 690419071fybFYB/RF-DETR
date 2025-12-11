"""
RF-DETR 推理和可视化脚本 (含密度图可视化)

使用训练好的 checkpoint 进行目标检测推理并可视化结果
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import torchvision.transforms.functional as F

# 添加项目路径
sys.path.insert(0, '/home/fyb/mydir/rf-detr')

from rfdetr import RFDETRBase
import supervision as sv

# 配置
CHECKPOINT_PATH = '/home/fyb/mydir/rf-detr/experiements/results/ablation_density_init/checkpoint_best_total.pth'
IMAGE_DIR = '/home/fyb/datasets/RSOD_cocoFormat/val2017'
OUTPUT_DIR = '/home/fyb/mydir/rf-detr/experiements/visualizations'
CONFIDENCE_THRESHOLD = 0.3
NUM_IMAGES = 10

# RSOD 类别名称
CLASS_NAMES = {
    1: 'aircraft',
    2: 'oiltank', 
    3: 'overpass',
    4: 'playground'
}

COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']


def load_model(checkpoint_path):
    """加载模型和 checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    model = RFDETRBase(
        enable_density_init=True,
        enable_improved_density=False,
        enable_small_object_query_boost=False,
        pretrain_weights=None,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.model.model.load_state_dict(state_dict, strict=False)
    model.model.model.eval()
    model.model.class_names = list(CLASS_NAMES.values())
    
    print("Model loaded successfully!")
    return model


def preprocess_image(image_path, resolution=560):
    """预处理图片并返回原始尺寸"""
    image = Image.open(image_path).convert('RGB')
    orig_size = image.size  # (W, H)
    
    # 转换为 tensor
    img_tensor = F.to_tensor(image)
    
    # 归一化
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    img_tensor = F.normalize(img_tensor, means, stds)
    
    # resize
    img_tensor = F.resize(img_tensor, (resolution, resolution))
    
    return img_tensor.unsqueeze(0), image, orig_size


def run_inference_with_density(model, image_tensor):
    """
    运行推理并获取密度图
    
    Returns:
        outputs: 模型输出字典
        density_map: 密度图 [1, 1, H, W]
    """
    device = model.model.device
    image_tensor = image_tensor.to(device)
    
    with torch.inference_mode():
        # 直接调用底层模型获取完整输出（包括密度图）
        outputs = model.model.model(image_tensor)
    
    # 提取密度图
    density_map = None
    if isinstance(outputs, dict) and 'pred_density' in outputs:
        density_map = outputs['pred_density']
    
    return outputs, density_map


def visualize_simple(orig_image, detections, output_path):
    """
    简单可视化：只显示检测结果，不显示密度图
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    image_np = np.array(orig_image)
    ax.imshow(image_np)
    
    if detections is not None and len(detections) > 0:
        for i in range(len(detections)):
            box = detections.xyxy[i]
            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            
            x1, y1, x2, y2 = box
            color = COLORS[class_id % len(COLORS)]
            
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=3, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
            ax.text(
                x1, y1 - 5, f'{class_name}: {confidence:.2f}',
                fontsize=10, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.85)
            )
    
    ax.set_title(f'Detections: {len(detections) if detections else 0}', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_with_density(image_path, detections, density_map, orig_image, output_path):
    """
    同时可视化检测结果和密度图（三栏布局）
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 原始图片
    axes[0].imshow(orig_image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # 2. 密度图
    if density_map is not None:
        density_np = density_map.squeeze().cpu().numpy()
        im = axes[1].imshow(density_np, cmap='jet', interpolation='bilinear')
        axes[1].set_title(f'Density Map\n(max: {density_np.max():.2f})', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].text(0.5, 0.5, 'No density map', ha='center', va='center', fontsize=14)
        axes[1].set_title('Density Map', fontsize=12)
        axes[1].axis('off')
    
    # 3. 检测结果叠加
    image_np = np.array(orig_image)
    axes[2].imshow(image_np)
    
    if detections is not None and len(detections) > 0:
        for i in range(len(detections)):
            box = detections.xyxy[i]
            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            
            x1, y1, x2, y2 = box
            color = COLORS[class_id % len(COLORS)]
            
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            axes[2].add_patch(rect)
            
            class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
            axes[2].text(
                x1, y1 - 3, f'{class_name}: {confidence:.2f}',
                fontsize=8, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8)
            )
    
    axes[2].set_title(f'Detections: {len(detections) if detections else 0}', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='RF-DETR 推理可视化')
    parser.add_argument('--images', '-i', nargs='+', type=str, default=None,
                        help='指定图片路径 (可多个)')
    parser.add_argument('--dir', '-d', type=str, default=IMAGE_DIR,
                        help='图片目录')
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_DIR,
                        help='输出目录')
    parser.add_argument('--num', '-n', type=int, default=NUM_IMAGES,
                        help='随机选择的图片数量')
    parser.add_argument('--threshold', '-t', type=float, default=CONFIDENCE_THRESHOLD,
                        help='置信度阈值')
    parser.add_argument('--checkpoint', '-c', type=str, default=CHECKPOINT_PATH,
                        help='模型 checkpoint 路径')
    parser.add_argument('--simple', '-s', action='store_true',
                        help='简单模式: 只输出检测结果，不输出密度图')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    model = load_model(args.checkpoint)
    
    # 获取图片列表
    if args.images:
        # 使用指定的图片
        image_files = [Path(p) for p in args.images]
        print(f"\n使用指定的 {len(image_files)} 张图片")
    else:
        # 从目录随机选择
        image_files = list(Path(args.dir).glob('*.jpg')) + list(Path(args.dir).glob('*.png'))
        if len(image_files) == 0:
            print(f"No images found in {args.dir}")
            return
        if len(image_files) > args.num:
            image_files = random.sample(image_files, args.num)
        print(f"\n从目录随机选择 {len(image_files)} 张图片")
    
    mode_str = "简单模式 (只显示检测结果)" if args.simple else "完整模式 (含密度图)"
    print(f"可视化模式: {mode_str}")
    print(f"置信度阈值: {args.threshold}")
    print(f"输出目录: {args.output}\n")
    
    for img_path in image_files:
        if not img_path.exists():
            print(f"文件不存在: {img_path}")
            continue
            
        try:
            img_tensor, orig_image, orig_size = preprocess_image(str(img_path))
            detections = model.predict(orig_image, threshold=args.threshold)
            
            if args.simple:
                # 简单模式：只输出检测结果
                output_path = os.path.join(args.output, f'det_{img_path.stem}.png')
                visualize_simple(orig_image, detections, output_path)
            else:
                # 完整模式：输出密度图 + 检测结果
                _, density_map = run_inference_with_density(model, img_tensor)
                output_path = os.path.join(args.output, f'density_{img_path.stem}.png')
                visualize_with_density(str(img_path), detections, density_map, orig_image, output_path)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*50}")
    print(f"可视化完成! 结果保存至: {args.output}")


if __name__ == '__main__':
    main()
