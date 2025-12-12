
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as TF
from typing import Tuple, Dict, Optional

# Add project root to path if needed (implicit in your env but good for portability)
import sys
project_root = '/home/fyb/mydir/rf-detr'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rfdetr.detr import RFDETRBase

# --- 1. Configuration ---
CHECKPOINT_PATH = '/home/fyb/mydir/rf-detr/experiements/results/ablation_density_init/checkpoint_best_total.pth'
IMAGE_PATH = '/home/fyb/datasets/RSOD_cocoFormat/val2017/aircraft_aircraft_905.jpg'
OUTPUT_PATH = 'comparison_heatmap_triplet.jpg'
RESOLUTION = 560
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global storage for hooks
hook_data = {
    'p3_shape': None,
    'p3_logits': None,
    'pred_density': None
}

def clean_hook_data():
    for k in hook_data:
        hook_data[k] = None

# --- 2. Hook Definitions ---
def transformer_hook(module, inputs, outputs):
    # Retrieve P3 shape from inputs
    srcs = inputs[0]
    p3_feat = srcs[0] 
    hook_data['p3_shape'] = p3_feat.shape[-2:] # (H, W)
    
    # Retrieve density from outputs
    # Transformer returns:
    # (hs, references, memory_ts, boxes_ts, scale_logits, density_outputs)
    # density_outputs is the last element (index 5)
    # Check if outputs is a tuple and has enough elements
    if isinstance(outputs, tuple) and len(outputs) >= 6:
        density_outputs = outputs[5]
        if density_outputs is not None and 'pred_density' in density_outputs:
            hook_data['pred_density'] = density_outputs['pred_density'].detach().cpu()
    else:
        print(f"Warning: Transformer output format unexpected. Type: {type(outputs)}")

def enc_class_output_hook(module, inputs, outputs):
    # Filter repeated calls
    if hook_data['p3_logits'] is not None:
        return
    if outputs.shape[1] <= 900: 
        return
    hook_data['p3_logits'] = outputs.detach().cpu()

# --- 3. Functions ---
def load_rfdetr_model(checkpoint_path: str, device: torch.device):
    print("Loading model architecture...")
    model_wrapper = RFDETRBase(
        enable_density_init=True,
        density_loss_coef=1.0,
        enable_small_object_query_boost=False,
        soqb_boost_factor=2.0,
        pretrain_weights=None, 
    )
    model = model_wrapper.model.model
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights!")
    
    model.eval()
    model.to(device)
    
    # Register hooks
    # Hook transformer to get BOTH input (shape) and output (density)
    model.transformer.register_forward_hook(transformer_hook)
    model.transformer.enc_out_class_embed[0].register_forward_hook(enc_class_output_hook)
        
    return model

def preprocess_image(image_path: str, resolution: int) -> Tuple[torch.Tensor, np.ndarray, int, int]:
    print(f"Loading image: {image_path}")
    original_image = Image.open(image_path).convert("RGB")
    w_orig, h_orig = original_image.size
    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    img_tensor = TF.to_tensor(original_image)
    img_tensor = TF.normalize(img_tensor, means, stds)
    img_tensor = TF.resize(img_tensor, [resolution, resolution])
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, np.array(original_image), w_orig, h_orig

# --- Class Names (Update based on your dataset) ---
# Assuming RSOD classes based on inference script context
CLASS_NAMES = {
    1: 'aircraft',
    2: 'oiltank', 
    3: 'overpass',
    4: 'playground'
}
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

def visualize_comparison_matplotlib(original_img: np.ndarray, 
                                  heatmap_cls: np.ndarray, 
                                  heatmap_density: np.ndarray, 
                                  heatmap_modulated: np.ndarray,
                                  detections,
                                  output_path: str):
    """
    使用 matplotlib 生成 2x2 可视化网格：
    - 左上(0): P3热力图 (直接显示，不叠加原图)
    - 右上(1): 密度图 (直接显示)
    - 左下(2): 融合热力图 (热力图+密度图融合，叠加在原图上)
    - 右下(3): 最终目标检测结果
    """
    import matplotlib.patches as patches
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()  # [ax0, ax1, ax2, ax3]
    
    # 获取原图尺寸
    h, w = original_img.shape[:2]
    
    def display_raw_heatmap(ax, heatmap, title, cmap='jet'):
        """直接显示热力图（不叠加原图）"""
        # 调整大小以便更好显示
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        
        im = ax.imshow(heatmap_resized, cmap=cmap, interpolation='bilinear')
        ax.set_title(f"{title}\n(Min:{heatmap.min():.3f}, Max:{heatmap.max():.3f})", fontsize=14)
        ax.axis('off')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return im
    
    def apply_overlay(ax, img, heatmap, title):
        """将热力图叠加在原图上显示"""
        # 1. 显示背景图
        ax.imshow(img)
        
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
        heatmap_rgba[..., 3] = 0.5
        
        # 5. 叠加显示
        ax.imshow(heatmap_rgba, interpolation='bilinear')
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=min_val, vmax=max_val))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    # ========== 左上(0): P3热力图 (直接显示) ==========
    display_raw_heatmap(axes[0], heatmap_cls, "P3 Classification Heatmap")
    
    # ========== 右上(1): 密度图 (直接显示) ==========
    display_raw_heatmap(axes[1], heatmap_density, "Density Map")
    
    # ========== 左下(2): 融合热力图 (叠加在原图上) ==========
    apply_overlay(axes[2], original_img, heatmap_modulated, "Fused Heatmap (P3 + Density) Overlay")
    
    # ========== 右下(3): 最终检测结果 ==========
    ax_det = axes[3]
    ax_det.imshow(original_img)
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
    print(f"2x2 可视化图已保存: {output_path}")

# Redefining load_rfdetr_model to return wrapper
def load_rfdetr_system(checkpoint_path: str, device: torch.device):
    print("Loading model architecture...")
    model_wrapper = RFDETRBase(
        enable_density_init=True,
        density_loss_coef=1.0,
        enable_small_object_query_boost=False,
        soqb_boost_factor=2.0,
        pretrain_weights=None, 
    )
    inner_model = model_wrapper.model.model
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        inner_model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights!")
    
    inner_model.eval()
    inner_model.to(device)
    
    # Register hooks on inner model
    inner_model.transformer.register_forward_hook(transformer_hook)
    inner_model.transformer.enc_out_class_embed[0].register_forward_hook(enc_class_output_hook)
        
    return model_wrapper, inner_model

def main():
    clean_hook_data()
    # Updated loader usage
    model_wrapper, inner_model = load_rfdetr_system(CHECKPOINT_PATH, DEVICE)
    
    img_tensor, orig_np, w_orig, h_orig = preprocess_image(IMAGE_PATH, RESOLUTION)
    img_tensor = img_tensor.to(DEVICE)
    
    print("Running inference (Hooking)...")
    # 1. Run raw forward for hooks
    with torch.no_grad():
        inner_model(img_tensor)
        
    # Check captures
    if hook_data['p3_logits'] is None or hook_data['p3_shape'] is None:
        print("Error: Hooks did not capture critical data.")
        return

    # 2. Run prediction for final results (returns detections object)
    # RFDETRBase.predict expects PIL image or numpy array, and handles transforms internally usually?
    # Let's check inference_visualize.py. It passes `orig_image` (PIL).
    print("Running prediction (Final Detections)...")
    orig_pil = Image.fromarray(orig_np)
    # Set classes for the wrapper to handle ID mapping if it does internal decoding with names
    model_wrapper.model.class_names = list(CLASS_NAMES.values())
    detections = model_wrapper.predict(orig_pil, threshold=0.3)

    H3, W3 = hook_data['p3_shape']
    len_p3 = H3 * W3
    
    # Process Data
    # 1. Classification
    logits_p3 = hook_data['p3_logits'][0, :len_p3, :]
    scores_p3 = torch.sigmoid(logits_p3)
    objectness_p3, _ = scores_p3.max(dim=-1)
    heatmap_obj = objectness_p3.view(H3, W3).numpy()
    
    # 2. Density & Modulation
    pred_density = hook_data['pred_density']
    if pred_density is None:
        print("Error: Density map not captured.")
        return

    density_map = pred_density[0, 0].numpy()
    density_weight = torch.sigmoid(pred_density[0, 0]).numpy()
    modulated_map = heatmap_obj + 0.5 * density_weight

    # Visualization
    output_abs_path = Path(OUTPUT_PATH).absolute()
    visualize_comparison_matplotlib(orig_np, heatmap_obj, density_map, modulated_map, detections, str(output_abs_path))
    
    print(f"Visualization saved to: {output_abs_path}")

if __name__ == '__main__':
    main()
