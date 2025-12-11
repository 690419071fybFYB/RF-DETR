# ------------------------------------------------------------------------
# RF-DETR
# Multi-Scale Density Supervision Module
# ------------------------------------------------------------------------
"""
Multi-scale density prediction for scale-aware small object detection.
P3 focuses on small objects, P4 on medium, P5 on large.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ScaleDensityPredictor(nn.Module):
    """Simple density predictor for a single scale."""
    
    def __init__(self, hidden_dim: int, target_scale: str = 'small'):
        """
        Args:
            hidden_dim: Input feature dimension
            target_scale: 'small', 'medium', or 'large'
        """
        super().__init__()
        self.target_scale = target_scale
        
        self.predictor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=3, padding=1),
            nn.GroupNorm(32, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=1),
            nn.ReLU()  # Density must be non-negative
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W]
        Returns:
            density: Density map [B, 1, H, W]
        """
        return self.predictor(x)


class MultiScaleDensityHead(nn.Module):
    """
    Multi-scale density prediction head.
    Predicts density maps at P3, P4, P5 levels for different object sizes.
    """
    
    def __init__(self, hidden_dim: int, num_levels: int = 3):
        """
        Args:
            hidden_dim: Feature dimension
            num_levels: Number of FPN levels (default 3 for P3, P4, P5)
        """
        super().__init__()
        self.num_levels = num_levels
        
        # Scale-specific density predictors
        self.predictors = nn.ModuleList([
            ScaleDensityPredictor(hidden_dim, 'small'),   # P3 - small objects
            ScaleDensityPredictor(hidden_dim, 'medium'),  # P4 - medium objects
            ScaleDensityPredictor(hidden_dim, 'large'),   # P5 - large objects
        ])
        
        # Learnable scale weights for loss weighting
        self.scale_weights = nn.Parameter(torch.ones(num_levels))
    
    def forward(self, srcs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            srcs: List of feature maps [P3, P4, P5], each [B, C, H, W]
            
        Returns:
            densities: List of density maps, each [B, 1, H, W]
        """
        densities = []
        for i, (src, predictor) in enumerate(zip(srcs, self.predictors)):
            if i < len(srcs):
                density = predictor(src)
                densities.append(density)
        
        return densities
    
    def get_normalized_weights(self) -> torch.Tensor:
        """Get softmax-normalized scale weights for loss weighting."""
        return F.softmax(self.scale_weights, dim=0)


def compute_multiscale_density_loss(
    pred_densities: List[torch.Tensor],
    targets: List[dict],
    scale_weights: torch.Tensor,
    sigma_small: float = 1.0,
    sigma_medium: float = 1.5,
    sigma_large: float = 2.0,
) -> torch.Tensor:
    """
    Compute multi-scale density loss.
    
    Args:
        pred_densities: List of predicted density maps [P3, P4, P5]
        targets: List of target dicts containing 'boxes' (cx, cy, w, h normalized)
        scale_weights: Learnable weights for each scale
        sigma_small/medium/large: Gaussian sigma for each scale
        
    Returns:
        total_loss: Weighted sum of density losses across scales
    """
    sigmas = [sigma_small, sigma_medium, sigma_large]
    # Area thresholds (normalized by image area)
    # small: area < 0.01, medium: 0.01 <= area < 0.1, large: area >= 0.1
    area_thresholds = [(0, 0.01), (0.01, 0.1), (0.1, 1.0)]
    
    total_loss = 0.0
    normalized_weights = F.softmax(scale_weights, dim=0)
    
    for level_idx, (pred_density, sigma, (area_min, area_max)) in enumerate(
        zip(pred_densities, sigmas, area_thresholds)
    ):
        # Generate GT density for this scale (filter by object size)
        gt_density = generate_scale_specific_gt_density(
            targets, pred_density.shape, sigma, area_min, area_max
        )
        
        # MSE loss for this scale
        loss = F.mse_loss(pred_density, gt_density)
        total_loss = total_loss + normalized_weights[level_idx] * loss
    
    return total_loss


def generate_scale_specific_gt_density(
    targets: List[dict],
    density_shape: Tuple[int, int, int, int],
    sigma: float,
    area_min: float,
    area_max: float,
) -> torch.Tensor:
    """
    Generate GT density map for specific object size range.
    
    Args:
        targets: List of target dicts containing 'boxes'
        density_shape: (B, 1, H, W) shape of output density
        sigma: Gaussian kernel sigma
        area_min, area_max: Normalized area range for this scale
        
    Returns:
        gt_density: [B, 1, H, W] ground truth density map
    """
    B, _, H, W = density_shape
    device = targets[0]['labels'].device if len(targets) > 0 and 'labels' in targets[0] else 'cuda'
    
    gt_density_list = []
    
    # Create coordinate grid
    y_range = torch.arange(H, device=device, dtype=torch.float32)
    x_range = torch.arange(W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    grid_y = grid_y.unsqueeze(0)  # [1, H, W]
    grid_x = grid_x.unsqueeze(0)  # [1, H, W]
    
    for i in range(B):
        if i >= len(targets):
            gt_density_list.append(torch.zeros((1, H, W), device=device))
            continue
            
        boxes = targets[i].get('boxes', torch.zeros((0, 4), device=device))
        
        if boxes.shape[0] == 0:
            gt_density_list.append(torch.zeros((1, H, W), device=device))
            continue
        
        # Filter boxes by area
        areas = boxes[:, 2] * boxes[:, 3]  # w * h (normalized)
        mask = (areas >= area_min) & (areas < area_max)
        filtered_boxes = boxes[mask]
        
        if filtered_boxes.shape[0] == 0:
            gt_density_list.append(torch.zeros((1, H, W), device=device))
            continue
        
        # Convert to feature map coordinates
        cx = filtered_boxes[:, 0] * W  # [N]
        cy = filtered_boxes[:, 1] * H  # [N]
        
        # Reshape for broadcasting
        cx = cx.view(-1, 1, 1)  # [N, 1, 1]
        cy = cy.view(-1, 1, 1)  # [N, 1, 1]
        
        # Compute Gaussian
        squared_dist = (grid_x - cx)**2 + (grid_y - cy)**2
        gaussian = torch.exp(-squared_dist / (2 * sigma**2))
        
        # Sum across all objects
        img_density = gaussian.sum(dim=0, keepdim=True)  # [1, H, W]
        gt_density_list.append(img_density)
    
    return torch.stack(gt_density_list, dim=0)  # [B, 1, H, W]
