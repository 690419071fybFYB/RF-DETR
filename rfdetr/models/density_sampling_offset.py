# ------------------------------------------------------------------------
# RF-DETR
# Density-Guided Sampling Offset Modulation Module
# ------------------------------------------------------------------------
"""
Modulates multi-scale deformable attention sampling offsets based on density map.
Applies offset modulation only to P4 level to avoid noise from P3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DensitySamplingOffsetModulation(nn.Module):
    """
    Modulates sampling offsets in MSDeformAttn based on density map.
    
    At each query's reference point, samples the density value and generates
    an offset modulation vector that adjusts the sampling pattern.
    In dense regions, sampling points become more concentrated.
    """
    
    def __init__(
        self, 
        n_heads: int = 8, 
        n_points: int = 4, 
        scale_factor: float = 0.1,
        target_level: int = 1  # P4 level
    ):
        """
        Args:
            n_heads: Number of attention heads
            n_points: Number of sampling points per head per level
            scale_factor: Scaling factor for modulation (controls strength)
            target_level: Which level to apply modulation (0=P3, 1=P4, 2=P5)
        """
        super().__init__()
        self.n_heads = n_heads
        self.n_points = n_points
        self.scale_factor = scale_factor
        self.target_level = target_level
        
        # MLP: density value -> offset modulation vector
        # Output shape: [n_heads * n_points * 2] for (dx, dy) per point per head
        self.mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_heads * n_points * 2),
            nn.Tanh(),  # Bound output to [-1, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def sample_density_at_points(
        self, 
        density_map: Tensor, 
        points: Tensor
    ) -> Tensor:
        """
        Sample density values at specified points using bilinear interpolation.
        
        Args:
            density_map: [B, 1, H, W] - predicted density map
            points: [B, N, 2] - normalized coordinates (x, y) in [0, 1]
            
        Returns:
            density_values: [B, N] - sampled density values
        """
        B, N, _ = points.shape
        
        # Convert from [0, 1] to [-1, 1] for grid_sample
        grid = points * 2 - 1  # [B, N, 2]
        grid = grid.unsqueeze(2)  # [B, N, 1, 2]
        
        # Sample: output shape [B, 1, N, 1]
        sampled = F.grid_sample(
            density_map, 
            grid, 
            mode='bilinear', 
            padding_mode='border',
            align_corners=True
        )
        
        # Reshape to [B, N]
        density_values = sampled.squeeze(1).squeeze(-1)
        
        return density_values
    
    def forward(
        self, 
        sampling_offsets: Tensor, 
        reference_points: Tensor, 
        density_map: Tensor
    ) -> Tensor:
        """
        Apply density-guided modulation to sampling offsets.
        
        Args:
            sampling_offsets: [B, N, n_heads, n_levels, n_points, 2]
            reference_points: [B, N, n_levels, 2] or [B, N, n_levels, 4]
            density_map: [B, 1, H, W]
            
        Returns:
            modulated_offsets: [B, N, n_heads, n_levels, n_points, 2]
        """
        B, N, n_heads, n_levels, n_points, _ = sampling_offsets.shape
        
        if self.target_level >= n_levels:
            # Target level doesn't exist, return unchanged
            return sampling_offsets
        
        # Extract reference points for target level (P4)
        if reference_points.dim() == 4:
            # Shape: [B, N, n_levels, 2] or [B, N, n_levels, 4]
            ref_xy = reference_points[:, :, self.target_level, :2]
        else:
            # Fallback
            return sampling_offsets
        
        # Sample density values at reference points
        # density_values: [B, N]
        density_values = self.sample_density_at_points(density_map, ref_xy)
        
        # Normalize density to [0, 1] with sigmoid
        density_values = density_values.sigmoid()
        
        # Generate offset modulation through MLP
        # Input: [B, N, 1], Output: [B, N, n_heads * n_points * 2]
        modulation = self.mlp(density_values.unsqueeze(-1))
        
        # Reshape to [B, N, n_heads, n_points, 2]
        modulation = modulation.view(B, N, self.n_heads, self.n_points, 2)
        
        # Apply modulation only to target level (P4)
        # Clone to avoid in-place modification issues
        modulated_offsets = sampling_offsets.clone()
        modulated_offsets[:, :, :, self.target_level, :, :] = (
            sampling_offsets[:, :, :, self.target_level, :, :] + 
            modulation * self.scale_factor
        )
        
        return modulated_offsets
