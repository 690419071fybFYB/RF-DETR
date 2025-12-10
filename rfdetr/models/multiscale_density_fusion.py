# ------------------------------------------------------------------------
# RF-DETR
# Multi-Scale Density Fusion Module
# ------------------------------------------------------------------------
"""
Multi-scale density fusion for better density prediction across different object sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class MultiScaleDensityFusion(nn.Module):
    """
    Fuses density maps from multiple feature scales (P3, P4, P5) for comprehensive density prediction.
    """

    def __init__(self, hidden_dim: int, scales: List[str] = ['P3', 'P4', 'P5']):
        super().__init__()
        self.scales = scales
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

        # Scale-specific density predictors
        self.density_predictors = nn.ModuleDict()
        for scale in scales:
            self.density_predictors[scale] = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1),
                nn.GroupNorm(32, hidden_dim // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 4, 1, 1),
                nn.ReLU()  # Ensure non-negative density
            )

        # Feature fusion module
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(len(scales), hidden_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 8, 1, 1),
            nn.Sigmoid()  # Attention weights
        )

    def forward(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_dict: Dictionary with feature maps for each scale
                         Keys: 'P3', 'P4', 'P5', values: [B, C, H, W]

        Returns:
            fused_density: Fused density map [B, 1, H_base, W_base]
        """
        device = next(self.parameters()).device
        batch_size = next(iter(features_dict.values())).shape[0]

        # Get base resolution (P3 - highest resolution)
        base_h, base_w = features_dict['P3'].shape[2:]

        density_maps = []

        # Predict density for each scale
        for i, scale in enumerate(self.scales):
            if scale in features_dict:
                feat = features_dict[scale]
                density = self.density_predictors[scale](feat)  # [B, 1, H, W]

                # Resize to base resolution
                if density.shape[2:] != (base_h, base_w):
                    density = F.interpolate(
                        density,
                        size=(base_h, base_w),
                        mode='bilinear',
                        align_corners=False
                    )

                density_maps.append(density * torch.softmax(self.scale_weights, dim=0)[i])

        # Stack density maps: [B, num_scales, H, W]
        stacked_densities = torch.stack(density_maps, dim=1)

        # Generate fusion weights
        fusion_weights = self.fusion_conv(stacked_densities)  # [B, 1, H, W]

        # Weighted fusion
        fused_density = torch.sum(stacked_densities * fusion_weights, dim=1, keepdim=True)

        return fused_density


class AdaptiveDensityThreshold(nn.Module):
    """
    Learns adaptive thresholds for density-based decision making.
    """

    def __init__(self, hidden_dim: int, num_thresholds: int = 3):
        super().__init__()
        self.num_thresholds = num_thresholds

        # Learn threshold predictors based on global features
        self.threshold_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, num_thresholds, 1),
            nn.Sigmoid()  # Output thresholds in [0, 1]
        )

    def forward(self, features: torch.Tensor, density_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Feature map [B, C, H, W]
            density_map: Density map [B, 1, H, W]

        Returns:
            thresholds: Adaptive thresholds [B, num_thresholds, 1, 1]
        """
        thresholds = self.threshold_predictor(features)  # [B, num_thresholds, 1, 1]
        return thresholds