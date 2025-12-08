
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
    inverse_sigmoid,
)

class DensityPredictor(nn.Module):
    """
    Predicts a density map from encoder features to guide query initialization.
    """
    def __init__(self, hidden_dim, kernel_size=3):
        super().__init__()
        # Lightweight FCN head to predict density map
        # Input: [Batch, HiddenDim, H, W]
        # Output: [Batch, 1, H, W]
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.ReLU() # Density must be non-negative
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)

class DensityGuidedQueryInit(nn.Module):
    def __init__(self, hidden_dim, sigma=1.0):
        super().__init__()
        self.predictor = DensityPredictor(hidden_dim)
        self.sigma = sigma
        
    def forward(self, features):
        """
        Args:
            features: Encoder output features [Batch, HiddenDim, H, W] or sequence [Batch, Len, HiddenDim]
                      If sequence, needs reshaping.
        Returns:
            density_map: Predicted density map [Batch, 1, H, W]
        """
        # Assuming features are already in spatial format [B, C, H, W]
        # If they are flattened [B, L, C], the caller needs to reshape them first
        if features.dim() == 3:
            # Simple heuristic reshaping if needed, but better handling in caller
            pass 
            
        return self.predictor(features)
        
    def sample_queries(self, density_map, num_queries, class_scores=None, alpha=0.5):
        """
        Sample query coordinates based on density map.
        
        Args:
            density_map: [Batch, 1, H, W]
            num_queries: Number of queries to sample (K)
            class_scores: Optional [Batch, H*W, NumClasses] or [Batch, H, W] objectness scores
            alpha: Weighting factor for mixing Score * alpha + Density * (1-alpha)
            
        Returns:
            topk_indices: Indices of sampled points in flattened map [Batch, K]
            topk_coords: Normalized coordinates [Batch, K, 2] (x, y)
        """
        B, C, H, W = density_map.shape
        device = density_map.device
        
        # Flatten density map [B, H*W]
        flat_density = density_map.flatten(2).squeeze(1)
        
        # Add a small epsilon to avoid zero issues
        flat_density = flat_density + 1e-6
        
        if class_scores is not None:
            # Assume class_scores is already flattened [B, H*W] or [B, H*W, 1]
            if class_scores.dim() == 3:
                # Take max score across classes as "objectness"
                class_scores = class_scores.max(dim=-1)[0]
            
            # Normalize both to 0-1 range for fair combination
            norm_density = (flat_density - flat_density.min(1, keepdim=True)[0]) / \
                           (flat_density.max(1, keepdim=True)[0] - flat_density.min(1, keepdim=True)[0] + 1e-6)
            
            norm_scores = (class_scores - class_scores.min(1, keepdim=True)[0]) / \
                          (class_scores.max(1, keepdim=True)[0] - class_scores.min(1, keepdim=True)[0] + 1e-6)
                          
            # Combine strategies
            ranking_metric = alpha * norm_scores + (1 - alpha) * norm_density
        else:
            ranking_metric = flat_density
            
        # Select Top-K indices
        # We use Top-K instead of multinomial sampling for stability and determinism in eval
        topk_values, topk_indices = torch.topk(ranking_metric, num_queries, dim=1)
        
        # Convert indices to coordinates
        # indices are in range [0, H*W-1]
        # y = indices // W, x = indices % W
        y = topk_indices // W
        x = topk_indices % W
        
        # Normalize to [0, 1]
        # Add 0.5 to be at pixel center
        # Important: use dynamic H, W from shape
        norm_x = (x.float() + 0.5) / W
        norm_y = (y.float() + 0.5) / H
        
        topk_coords = torch.stack([norm_x, norm_y], dim=-1)
        
        return topk_indices, topk_coords

    @staticmethod
    def generate_gt_density_map(batched_inputs, features_shape, sigma=1.0):
        """
        Generate Ground Truth density map from boxes.
        
        Args:
            batched_inputs: List of dicts (from COCO dataset), containing 'boxes' (cx, cy, w, h normalized)
            features_shape: (B, C, H, W) of the feature map to align with
            sigma: Kernel size for Gaussian
            
        Returns:
            gt_density: [B, 1, H, W]
        """
        B, C, H, W = features_shape
        device = batched_inputs[0]['labels'].device # Assuming tensor
        
        gt_density_list = []
        
        # Create coordinate grid
        # x_grid: [1, 1, H, W]
        y_range = torch.arange(H, device=device).float()
        x_range = torch.arange(W, device=device).float()
        grid_y, grid_x = torch.meshgrid(y_range, x_range)
        
        # Add batch dim
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)
        
        for i in range(B):
            # Get boxes for this image
            # 'boxes' are typically [N, 4] in (cx, cy, w, h) normalized format
            # We need to handle potential 'targets' dict structure
            if isinstance(batched_inputs, list) and 'boxes' in batched_inputs[i]:
                 boxes = batched_inputs[i]['boxes'] # [N, 4]
            elif isinstance(batched_inputs[i], dict) and 'boxes' in batched_inputs[i]:
                 boxes = batched_inputs[i]['boxes']
            else:
                 # Fallback for NestedTensor structure where targets might be separate
                 # Assuming caller handles this mapping, but for now placeholder
                 boxes = torch.zeros((0, 4), device=device)
            
            # If no boxes, empty map
            if boxes.shape[0] == 0:
                gt_density_list.append(torch.zeros((1, H, W), device=device))
                continue
                
            # Convert normalized boxes to feature map coordinates
            # cx, cy are in [0, 1]
            cx = boxes[:, 0] * W
            cy = boxes[:, 1] * H
            
            # Reshape for broadcasting
            # cx: [N, 1, 1]
            cx = cx.view(-1, 1, 1)
            cy = cy.view(-1, 1, 1)
            
            # Calculate Gaussian for each box
            # exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))
            # We sum them up
            
            # [N, H, W]
            squared_dist = (grid_x - cx)**2 + (grid_y - cy)**2
            gaussian = torch.exp(-squared_dist / (2 * sigma**2))
            
            # Sum across all objects to get density map
            # [1, H, W] after sum over N (dim=1)
            # gaussian shape: [1, N, H, W]
            img_density = gaussian.sum(dim=1, keepdim=False)
            gt_density_list.append(img_density)
            
        return torch.stack(gt_density_list, dim=0)

