# ------------------------------------------------------------------------
# RF-DETR
# Density-Augmented Cross-Attention Module
# ------------------------------------------------------------------------
"""
Density Feature Augmentation for Cross-Attention.
Fuses predicted density map into memory features for enhanced small object detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DensityFeatureAugmentation(nn.Module):
    """
    Augments encoder memory features with density information.
    
    Projects the single-channel density map to hidden_dim and adds it 
    to the specified feature level (typically P4) via residual connection.
    """
    
    def __init__(self, hidden_dim: int, scale_factor: float = 0.1):
        """
        Args:
            hidden_dim: Dimension of the encoder features (d_model)
            scale_factor: Scaling factor for the density embedding (controls contribution strength)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale_factor = scale_factor
        
        # Project 1-channel density map to hidden_dim
        self.density_proj = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=1),
        )
        
        # LayerNorm for stability
        self.norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        memory: Tensor, 
        density_map: Tensor, 
        target_h: int, 
        target_w: int,
        start_idx: int,
        end_idx: int
    ) -> Tensor:
        """
        Augment a portion of memory features with density information.
        
        Args:
            memory: [B, sum(H*W), C] - flattened multi-scale encoder features
            density_map: [B, 1, H_d, W_d] - predicted density map (usually from P3)
            target_h, target_w: Spatial dimensions of the target level (P4)
            start_idx: Start index of target level in flattened memory
            end_idx: End index of target level in flattened memory
            
        Returns:
            augmented_memory: [B, sum(H*W), C] - memory with augmented target level
        """
        B, _, C = memory.shape
        device = memory.device
        dtype = memory.dtype
        
        # 1. Resize density map to match target spatial size
        # density_map: [B, 1, H_d, W_d] -> [B, 1, target_h, target_w]
        if density_map.shape[2] != target_h or density_map.shape[3] != target_w:
            density_resized = F.interpolate(
                density_map, 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            density_resized = density_map
        
        # 2. Project density to hidden_dim
        # [B, 1, H, W] -> [B, C, H, W]
        density_embed = self.density_proj(density_resized)
        
        # 3. Flatten to match memory format
        # [B, C, H, W] -> [B, H*W, C]
        density_flat = density_embed.flatten(2).transpose(1, 2)
        
        # 4. Apply LayerNorm and scale
        density_flat = self.norm(density_flat) * self.scale_factor
        
        # 5. Clone memory and add density embedding to target level only
        augmented_memory = memory.clone()
        augmented_memory[:, start_idx:end_idx, :] = (
            memory[:, start_idx:end_idx, :] + density_flat.to(dtype)
        )
        
        return augmented_memory


class DensityAugmentedMemory(nn.Module):
    """
    Wrapper module that handles the full density augmentation pipeline.
    Automatically identifies P4 level and applies augmentation.
    """
    
    def __init__(self, hidden_dim: int, target_level: int = 1, scale_factor: float = 0.1):
        """
        Args:
            hidden_dim: Dimension of encoder features
            target_level: Feature level to augment (0=P3, 1=P4, 2=P5)
            scale_factor: Strength of density contribution
        """
        super().__init__()
        self.target_level = target_level
        self.augmentation = DensityFeatureAugmentation(hidden_dim, scale_factor)
    
    def forward(
        self,
        memory: Tensor,
        density_map: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor
    ) -> Tensor:
        """
        Apply density augmentation to the target feature level.
        
        Args:
            memory: [B, sum(H*W), C] - flattened encoder features
            density_map: [B, 1, H, W] - predicted density map
            spatial_shapes: [num_levels, 2] - (H, W) for each level
            level_start_index: [num_levels] - start index for each level
            
        Returns:
            augmented_memory: [B, sum(H*W), C]
        """
        if density_map is None:
            return memory
        
        num_levels = spatial_shapes.shape[0]
        
        # Validate target level
        if self.target_level >= num_levels:
            return memory
        
        # Get target level spatial info
        target_h, target_w = spatial_shapes[self.target_level].tolist()
        start_idx = level_start_index[self.target_level].item()
        
        # Calculate end index
        if self.target_level + 1 < num_levels:
            end_idx = level_start_index[self.target_level + 1].item()
        else:
            end_idx = memory.shape[1]
        
        return self.augmentation(
            memory, density_map, target_h, target_w, start_idx, end_idx
        )
