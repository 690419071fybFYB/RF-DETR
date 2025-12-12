"""
RF-DETR 推理和可视化脚本 (含密度图可视化)

使用训练好的 checkpoint 进行目标检测推理并可视化结果
支持多种可视化模式:
- 简单模式: 只显示检测结果
- 完整模式: 三栏布局 (原图 + 密度图 + 检测结果)
- 四栏模式: 2x2 布局 (P3热力图 + 密度图 + 融合热力图 + 检测结果)
"""
import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import torchvision.transforms.functional as F

# 全局存储钩子捕获的数据
hook_data = {
    'feat_shape': None,      # 最高分辨率特征图的形状
    'feat_logits': None,     # 分类 logits
    'pred_density': None,    # 密度图
    'feature_level': 'P4'    # 当前特征层级名称 (由模型配置决定)
}

def clean_hook_data():
    """清理钩子数据"""
    for k in hook_data:
        hook_data[k] = None

def transformer_hook(module, inputs, outputs):
    """
    Transformer 钩子函数
    从输入获取最高分辨率特征图形状，从输出获取密度图
    注意: srcs[0] 对应 projector_scale 中的第一个层级 (由模型配置决定)
    """
    # 从输入获取最高分辨率特征图 shape
    srcs = inputs[0]
    highest_res_feat = srcs[0] 
    hook_data['feat_shape'] = highest_res_feat.shape[-2:]  # (H, W)
    
    # 从输出获取密度图
    # Transformer 返回: (hs, references, memory_ts, boxes_ts, scale_logits, density_outputs)
    if isinstance(outputs, tuple) and len(outputs) >= 6:
        density_outputs = outputs[5]
        if density_outputs is not None and 'pred_density' in density_outputs:
            hook_data['pred_density'] = density_outputs['pred_density'].detach().cpu()

def enc_class_output_hook(module, inputs, outputs):
    """
    编码器分类输出钩子函数
    捕获最高分辨率特征层级的分类 logits
    """
    # 过滤重复调用
    if hook_data['feat_logits'] is not None:
        return
    if outputs.shape[1] <= 900: 
        return
    hook_data['feat_logits'] = outputs.detach().cpu()

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


def load_model(checkpoint_path, register_hooks=False):
    """
    加载模型和 checkpoint
    
    Args:
        checkpoint_path: checkpoint 文件路径
        register_hooks: 是否注册钩子函数 (用于四栏可视化模式)
    
    Returns:
        model: RFDETRBase 模型包装器
        inner_model: 底层 LWDETR 模型 (仅在 register_hooks=True 时返回)
        feature_level: 最高分辨率特征层级名称 (仅在 register_hooks=True 时返回)
    """
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
    
    inner_model = model.model.model
    
    # 获取 projector_scale 配置，确定最高分辨率特征层级
    # model.model 是 Model 实例，有 args 属性
    projector_scale = getattr(model.model.args, 'projector_scale', ['P4'])
    # projector_scale 可能是列表或字符串
    if isinstance(projector_scale, str):
        projector_scale = [projector_scale]
    feature_level = projector_scale[0] if projector_scale else 'P4'  # 默认 P4
    print(f"Model projector_scale: {projector_scale} -> Highest resolution level: {feature_level}")
    
    if register_hooks:
        # 注册钩子以捕获特征热力图数据
        inner_model.transformer.register_forward_hook(transformer_hook)
        inner_model.transformer.enc_out_class_embed[0].register_forward_hook(enc_class_output_hook)
        # 在全局变量中记录特征层级
        hook_data['feature_level'] = feature_level
        print(f"Hooks registered for {feature_level} heatmap visualization")
    
    print("Model loaded successfully!")
    
    if register_hooks:
        return model, inner_model, feature_level
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


def visualize_quad(orig_image, heatmap_feat, density_map, fused_heatmap, detections, output_path, feature_level='P4'):
    """
    2x2 四栏布局可视化（全部叠加在原图上）：
    - 左上(0,0): 特征热力图 + 原图叠加 (根据 feature_level 动态命名)
    - 右上(0,1): 密度图 + 原图叠加
    - 左下(1,0): 特征热力图 + 密度图融合后的热力图 + 原图叠加
    - 右下(1,1): 最终目标检测结果
    
    Args:
        orig_image: 原始 PIL 图像
        heatmap_feat: 特征分类热力图 [H, W] numpy array
        density_map: 密度图 [H, W] numpy array
        fused_heatmap: 融合热力图 [H, W] numpy array
        detections: supervision 检测结果
        output_path: 输出路径
        feature_level: 特征层级名称 (如 'P3', 'P4' 等)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    orig_np = np.array(orig_image)
    h, w = orig_np.shape[:2]
    
    def apply_overlay(ax, img, heatmap, title, alpha=0.5):
        """将热力图叠加在原图上显示"""
        # 1. 显示背景图
        ax.imshow(img)
        
        if heatmap is None:
            ax.set_title(title + '\n(No data)', fontsize=14)
            ax.axis('off')
            return
        
        # 2. 处理热力图
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 归一化到 [0, 1]
        min_val, max_val = heatmap_resized.min(), heatmap_resized.max()
        if max_val - min_val > 1e-6:
            norm_map = (heatmap_resized - min_val) / (max_val - min_val)
        else:
            norm_map = np.zeros_like(heatmap_resized)
            
        # 3. 应用颜色映射 (Jet)
        heatmap_rgba = plt.cm.jet(norm_map)
        
        # 4. 设置透明度
        heatmap_rgba[..., 3] = alpha
        
        # 5. 叠加显示
        ax.imshow(heatmap_rgba, interpolation='bilinear')
        
        ax.set_title(f"{title}\n(Min:{heatmap.min():.3f}, Max:{heatmap.max():.3f})", fontsize=14)
        ax.axis('off')
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=min_val, vmax=max_val))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    # ========== 左上(0,0): 特征热力图 + 原图叠加 ==========
    apply_overlay(axes[0, 0], orig_np, heatmap_feat, f"{feature_level} Classification Heatmap")
    
    # ========== 右上(0,1): 密度图 + 原图叠加 ==========
    apply_overlay(axes[0, 1], orig_np, density_map, "Density Map")
    
    # ========== 左下(1,0): 融合热力图 + 原图叠加 ==========
    apply_overlay(axes[1, 0], orig_np, fused_heatmap, f"Fused Heatmap ({feature_level} + Density)")
    
    # ========== 右下(1,1): 最终检测结果 ==========
    ax_det = axes[1, 1]
    ax_det.imshow(orig_np)
    ax_det.set_title(f"Detection Results (N={len(detections) if detections else 0})", fontsize=14)
    ax_det.axis('off')
    
    if detections is not None and len(detections) > 0:
        for i in range(len(detections)):
            box = detections.xyxy[i]
            class_id = detections.class_id[i]
            confidence = detections.confidence[i]
            
            x1, y1, x2, y2 = box
            class_name = CLASS_NAMES.get(class_id, str(class_id))
            color = COLORS[class_id % len(COLORS)]
            
            # 绘制检测框
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax_det.add_patch(rect)
            
            # 绘制标签
            label_text = f'{class_name} {confidence:.2f}'
            ax_det.text(
                x1, y1 - 2, label_text,
                fontsize=9, color='white', fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1.5)
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved (2x2 quad): {output_path}")


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
    parser.add_argument('--quad', '-q', action='store_true',
                        help='四栏模式: 2x2 布局输出特征热力图、密度图、融合热力图和检测结果')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 根据模式决定是否需要注册钩子
    feature_level = 'P4'  # 默认值
    if args.quad:
        model, inner_model, feature_level = load_model(args.checkpoint, register_hooks=True)
        device = next(inner_model.parameters()).device
    else:
        model = load_model(args.checkpoint)
        inner_model = None
    
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
    
    # 确定可视化模式
    if args.quad:
        mode_str = f"四栏模式 (2x2: {feature_level}热力图 + 密度图 + 融合热力图 + 检测结果)"
    elif args.simple:
        mode_str = "简单模式 (只显示检测结果)"
    else:
        mode_str = "完整模式 (含密度图)"
    print(f"可视化模式: {mode_str}")
    print(f"置信度阈值: {args.threshold}")
    print(f"输出目录: {args.output}\n")
    
    for img_path in image_files:
        if not img_path.exists():
            print(f"文件不存在: {img_path}")
            continue
            
        try:
            img_tensor, orig_image, orig_size = preprocess_image(str(img_path))
            
            if args.quad:
                # 四栏模式：需要通过钩子获取特征热力图
                clean_hook_data()  # 清理之前的钩子数据
                # 保留 feature_level 设置
                hook_data['feature_level'] = feature_level
                
                # 1. 运行底层模型前向传播以触发钩子
                img_tensor_device = img_tensor.to(device)
                with torch.no_grad():
                    inner_model(img_tensor_device)
                
                # 2. 获取检测结果
                detections = model.predict(orig_image, threshold=args.threshold)
                
                # 3. 处理钩子捕获的数据
                heatmap_feat = None
                density_map = None
                fused_heatmap = None
                
                # 处理特征分类热力图 (使用更新后的变量名)
                if hook_data['feat_logits'] is not None and hook_data['feat_shape'] is not None:
                    H, W = hook_data['feat_shape']
                    len_feat = H * W
                    logits_feat = hook_data['feat_logits'][0, :len_feat, :]
                    scores_feat = torch.sigmoid(logits_feat)
                    objectness_feat, _ = scores_feat.max(dim=-1)
                    heatmap_feat = objectness_feat.view(H, W).numpy()
                
                # 处理密度图
                if hook_data['pred_density'] is not None:
                    density_map = hook_data['pred_density'][0, 0].numpy()
                    
                    # 生成融合热力图
                    if heatmap_feat is not None:
                        # 使用 sigmoid 归一化密度图后与特征热力图融合
                        density_weight = torch.sigmoid(hook_data['pred_density'][0, 0]).numpy()
                        fused_heatmap = heatmap_feat + 0.5 * density_weight
                
                # 输出2x2可视化 (传递 feature_level)
                output_path = os.path.join(args.output, f'quad_{img_path.stem}.png')
                visualize_quad(orig_image, heatmap_feat, density_map, fused_heatmap, detections, output_path, feature_level)
                
            elif args.simple:
                # 简单模式：只输出检测结果
                detections = model.predict(orig_image, threshold=args.threshold)
                output_path = os.path.join(args.output, f'det_{img_path.stem}.png')
                visualize_simple(orig_image, detections, output_path)
            else:
                # 完整模式：输出密度图 + 检测结果
                detections = model.predict(orig_image, threshold=args.threshold)
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
