"""
Boundary-Aware Query Position Refinement

边界感知查询位置精炼模块：
1. 预测目标边界热图
2. 使用边界信息精炼 query 的参考点位置
3. 将参考点推离边界，靠近目标中心
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryPredictionHead(nn.Module):
    """
    边界预测头：从 FPN 特征预测目标边界热图
    
    输入: P3 特征 [B, C, H, W]
    输出: 边界热图 [B, 1, H, W]，值在 [0, 1] 范围内
    """
    
    def __init__(self, in_channels=256, hidden_channels=128):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        self.conv3 = nn.Conv2d(hidden_channels, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.out_conv = nn.Conv2d(64, 1, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: P3 特征 [B, C, H, W]
        Returns:
            boundary_map: 边界热图 [B, 1, H, W]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        boundary_map = torch.sigmoid(self.out_conv(x))
        return boundary_map


class QueryPositionRefinement(nn.Module):
    """
    Query 位置精炼模块：使用边界信息精炼参考点
    
    核心思想：
    - 在参考点周围采样边界值
    - 计算梯度方向（指向边界较低的方向，即目标中心）
    - 沿梯度方向微调参考点
    """
    
    def __init__(self, d_model=256, refinement_scale=0.05):
        super().__init__()
        
        self.refinement_scale = refinement_scale
        
        # 学习精炼偏移的 MLP
        self.refine_mlp = nn.Sequential(
            nn.Linear(d_model + 4, 128),  # query_feat + boundary_samples
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # dx, dy
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 最后一层初始化为接近 0，保证初始精炼量很小
        nn.init.constant_(self.refine_mlp[-1].weight, 0)
        nn.init.constant_(self.refine_mlp[-1].bias, 0)
    
    def sample_boundary_values(self, boundary_map, ref_points):
        """
        在参考点的 4 个方向（上下左右）采样边界值
        
        Args:
            boundary_map: [B, 1, H, W]
            ref_points: [B, N, 2] 归一化坐标
        Returns:
            samples: [B, N, 4] 4 个方向的边界值
        """
        B, N, _ = ref_points.shape
        H, W = boundary_map.shape[2:]
        
        # 采样偏移量（归一化空间）
        delta = 0.02  # 约 2% 的图像尺寸
        
        # 4 个方向的采样点
        offsets = torch.tensor([
            [0, -delta],   # 上
            [0, delta],    # 下
            [-delta, 0],   # 左
            [delta, 0],    # 右
        ], device=ref_points.device, dtype=ref_points.dtype)  # [4, 2]
        
        samples_list = []
        for i in range(4):
            sample_pts = ref_points + offsets[i:i+1]  # [B, N, 2]
            # 限制在 [0, 1] 范围内
            sample_pts = sample_pts.clamp(0, 1)
            # 转换为 grid_sample 格式 [-1, 1]
            grid = sample_pts * 2 - 1  # [B, N, 2]
            grid = grid.unsqueeze(1)  # [B, 1, N, 2]
            
            # 采样
            sampled = F.grid_sample(
                boundary_map, grid, 
                mode='bilinear', 
                padding_mode='border',
                align_corners=True
            )  # [B, 1, 1, N]
            samples_list.append(sampled.squeeze(1).squeeze(1))  # [B, N]
        
        samples = torch.stack(samples_list, dim=-1)  # [B, N, 4]
        return samples
    
    def forward(self, query_feat, ref_points, boundary_map):
        """
        精炼 query 的参考点位置
        
        Args:
            query_feat: [B, N, D] query 特征
            ref_points: [B, N, 2] 归一化参考点坐标
            boundary_map: [B, 1, H, W] 边界热图
        Returns:
            refined_ref_points: [B, N, 2] 精炼后的参考点
        """
        B, N, D = query_feat.shape
        
        # 采样边界值
        boundary_samples = self.sample_boundary_values(boundary_map, ref_points)  # [B, N, 4]
        
        # 拼接特征
        mlp_input = torch.cat([query_feat, boundary_samples], dim=-1)  # [B, N, D+4]
        
        # 预测偏移
        delta = self.refine_mlp(mlp_input)  # [B, N, 2]
        delta = torch.tanh(delta) * self.refinement_scale  # 限制精炼幅度
        
        # 应用偏移
        refined_ref_points = ref_points + delta
        refined_ref_points = refined_ref_points.clamp(0, 1)  # 确保在有效范围内
        
        return refined_ref_points


def generate_boundary_gt(targets, img_h, img_w, boundary_width=3):
    """
    从 ground truth boxes 生成边界热图
    
    Args:
        targets: 包含 'boxes' 的字典，boxes 为 [N, 4] 格式 (cx, cy, w, h) 归一化坐标
        img_h, img_w: 特征图尺寸
        boundary_width: 边界宽度（像素）
    Returns:
        boundary_gt: [1, img_h, img_w] 边界热图
    """
    device = targets['boxes'].device
    boundary_gt = torch.zeros(1, img_h, img_w, device=device)
    
    boxes = targets['boxes']  # [N, 4] (cx, cy, w, h)
    if len(boxes) == 0:
        return boundary_gt
    
    # 转换为像素坐标 (x1, y1, x2, y2)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = ((cx - w/2) * img_w).long().clamp(0, img_w - 1)
    y1 = ((cy - h/2) * img_h).long().clamp(0, img_h - 1)
    x2 = ((cx + w/2) * img_w).long().clamp(0, img_w - 1)
    y2 = ((cy + h/2) * img_h).long().clamp(0, img_h - 1)
    
    for i in range(len(boxes)):
        # 上边界
        y_start = y1[i].item()
        y_end = min(y1[i].item() + boundary_width, img_h)
        boundary_gt[0, y_start:y_end, x1[i]:x2[i]+1] = 1
        # 下边界
        y_start = max(0, y2[i].item() - boundary_width + 1)
        boundary_gt[0, y_start:y2[i]+1, x1[i]:x2[i]+1] = 1
        # 左边界
        x_start = x1[i].item()
        x_end = min(x1[i].item() + boundary_width, img_w)
        boundary_gt[0, y1[i]:y2[i]+1, x_start:x_end] = 1
        # 右边界
        x_start = max(0, x2[i].item() - boundary_width + 1)
        boundary_gt[0, y1[i]:y2[i]+1, x_start:x2[i]+1] = 1
    
    return boundary_gt
