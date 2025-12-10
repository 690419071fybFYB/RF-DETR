# ------------------------------------------------------------------------
# RF-DETR
# Density Positional Bias Modulation Module
# ------------------------------------------------------------------------
"""
Density Positional Bias for Transformer Decoder.
Samples density values at query reference points and generates positional bias vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DensityPositionalBias(nn.Module):
    """
    Generates positional bias vectors from density map for each query.
    
    At each query's reference point, samples the density value using bilinear
    interpolation, then projects it through an MLP to generate a bias vector
    that modulates the query positional embedding.
    """
    
    def __init__(self, d_model: int, scale_factor: float = 0.1):
        """
        Args:
            d_model: Dimension of the model (hidden_dim)
            scale_factor: Scaling factor for the bias (controls contribution strength)
        """
        super().__init__()
        self.d_model = d_model
        self.scale_factor = scale_factor
        
        # MLP: scalar density value -> d_model bias vector
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 4, d_model),
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
        # grid_sample expects (x, y) in [-1, 1] where (-1, -1) is top-left
        grid = points * 2 - 1  # [B, N, 2]
        
        # Reshape for grid_sample: [B, N, 1, 2]
        grid = grid.unsqueeze(2)
        
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
        query_pos: Tensor, 
        reference_points: Tensor, 
        density_map: Tensor
    ) -> Tensor:
        """
        Generate density-modulated query positional embeddings.
        
        Args:
            query_pos: [B, num_queries, d_model] - original positional embeddings
            reference_points: [B, num_queries, 2] or [B, num_queries, 4] 
                              - normalized reference points (cx, cy, [w, h])
            density_map: [B, 1, H, W] - predicted density map
            
        Returns:
            modulated_query_pos: [B, num_queries, d_model]
        """
        B, N, D = query_pos.shape
        
        # Extract (cx, cy) from reference points
        if reference_points.dim() == 4:
            # [B, N, num_levels, 4] -> use first level
            ref_xy = reference_points[:, :, 0, :2]
        elif reference_points.shape[-1] >= 2:
            ref_xy = reference_points[..., :2]
        else:
            # Fallback: no modification
            return query_pos
        
        # Sample density values at reference points
        # density_values: [B, N]
        density_values = self.sample_density_at_points(density_map, ref_xy)
        
        # Normalize to [0, 1] range using sigmoid for stability
        density_values = density_values.sigmoid()
        
        # Project through MLP: [B, N, 1] -> [B, N, d_model]
        bias = self.mlp(density_values.unsqueeze(-1))
        
        # Scale and add to query_pos
        modulated_query_pos = query_pos + bias * self.scale_factor
        
        return modulated_query_pos
