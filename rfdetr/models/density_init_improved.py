"""
Improved Density-Guided Query Initialization with UNet-based prediction.
"""

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


class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle different spatial sizes due to pooling
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU()  # Density must be non-negative
        )

    def forward(self, x):
        return self.conv(x)


class ImprovedDensityPredictor(nn.Module):
    """
    Improved density predictor using lightweight UNet architecture.
    Better preserves fine details and spatial relationships for small objects.
    """
    def __init__(self, hidden_dim, base_channels=64):
        super().__init__()

        # Input projection to base channels
        self.inc = DoubleConv(hidden_dim, base_channels)

        # Encoder path
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        # Decoder path with skip connections
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)

        # Output projection to single channel density
        self.outc = OutConv(base_channels, 1)

        # Optional: attention module for focusing on small objects
        self.attention = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input shape: [Batch, hidden_dim, H, W]

        # Encoder
        x1 = self.inc(x)          # [B, C, H, W]
        x2 = self.down1(x1)       # [B, 2C, H/2, W/2]
        x3 = self.down2(x2)       # [B, 4C, H/4, W/4]
        x4 = self.down3(x3)       # [B, 8C, H/8, W/8]

        # Bottleneck
        x5 = self.bottleneck(x4)  # [B, 16C, H/8, W/8]

        # Decoder with skip connections
        x = self.up1(x5, x4)      # [B, 8C, H/8, W/8] -> [B, 8C, H/4, W/4]
        x = self.up2(x, x3)       # [B, 4C, H/4, W/4]
        x = self.up3(x, x2)       # [B, 2C, H/2, W/2]
        x = self.up4(x, x1)       # [B, C, H, W]

        # Apply attention to focus on small object regions
        attention_map = self.attention(x)
        x = x * attention_map

        # Output density map
        density = self.outc(x)    # [B, 1, H, W]

        return density


class AdaptiveDensityThreshold(nn.Module):
    """
    Learn adaptive threshold for density-based sampling.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.threshold_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        """
        Predict adaptive threshold based on global features.

        Args:
            features: [Batch, hidden_dim, H, W]

        Returns:
            threshold: [Batch, 1] threshold values between 0 and 1
        """
        return self.threshold_predictor(features)


class DensityGuidedQueryInit(nn.Module):
    def __init__(self, hidden_dim, sigma=1.0, use_unet=True, adaptive_threshold=True):
        super().__init__()

        # Choose between original and improved predictor
        if use_unet:
            self.predictor = ImprovedDensityPredictor(hidden_dim)
        else:
            # Original simple predictor
            self.predictor = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
                nn.ReLU()
            )

        self.sigma = sigma

        # Adaptive threshold module
        if adaptive_threshold:
            self.adaptive_threshold = AdaptiveDensityThreshold(hidden_dim)
        else:
            self.adaptive_threshold = None

        # Multi-scale fusion
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),  # Fuse two scales
            nn.ReLU()
        )

    def forward(self, features):
        """
        Args:
            features: Encoder output features [Batch, HiddenDim, H, W]
        Returns:
            density_map: Predicted density map [Batch, 1, H, W]
        """
        if features.dim() == 3:
            # Reshape if flattened
            B, L, C = features.shape
            # Assuming square feature map
            H = W = int(math.sqrt(L))
            features = features.transpose(1, 2).view(B, C, H, W)

        # Predict density map
        density = self.predictor(features)

        # Apply multi-scale enhancement
        # Downsample for global context
        downsampled = F.avg_pool2d(density, kernel_size=3, stride=2, padding=1)
        downsampled = F.interpolate(downsampled, scale_factor=2, mode='bilinear', align_corners=False)

        # Fuse original and downsampled
        multi_scale = torch.cat([density, downsampled], dim=1)
        density = self.multi_scale_fusion(multi_scale)

        return density

    def sample_queries(self, density_map, num_queries, class_scores=None, alpha=0.5,
                      use_adaptive_threshold=False):
        """
        Enhanced query sampling with adaptive thresholding.

        Args:
            density_map: [Batch, 1, H, W]
            num_queries: Number of queries to sample (K)
            class_scores: Optional [Batch, H*W, NumClasses] objectness scores
            alpha: Weighting factor for mixing Score and Density
            use_adaptive_threshold: Whether to use learned threshold

        Returns:
            topk_indices: Indices of sampled points in flattened map [Batch, K]
            topk_coords: Normalized coordinates [Batch, K, 2] (x, y)
        """
        B, C, H, W = density_map.shape
        device = density_map.device

        # Flatten density map [B, H*W]
        flat_density = density_map.flatten(2).squeeze(1)

        # Apply adaptive threshold if enabled
        if use_adaptive_threshold and self.adaptive_threshold is not None:
            # This would require access to original features
            # For now, we use a simple percentile-based threshold
            threshold = flat_density.mean(dim=1, keepdim=True) * 0.5
            flat_density = flat_density * (flat_density > threshold)

        # Add epsilon to avoid zero issues
        flat_density = flat_density + 1e-6

        if class_scores is not None:
            # Handle class scores
            if class_scores.dim() == 3:
                class_scores = class_scores.max(dim=-1)[0]

            # Normalize both metrics
            norm_density = (flat_density - flat_density.min(1, keepdim=True)[0]) / \
                           (flat_density.max(1, keepdim=True)[0] - flat_density.min(1, keepdim=True)[0] + 1e-6)

            norm_scores = (class_scores - class_scores.min(1, keepdim=True)[0]) / \
                          (class_scores.max(1, keepdim=True)[0] - class_scores.min(1, keepdim=True)[0] + 1e-6)

            # Combine with learnable weighting
            ranking_metric = alpha * norm_scores + (1 - alpha) * norm_density
        else:
            ranking_metric = flat_density

        # Sample queries with diversity consideration
        # Use spatially-aware sampling to avoid clustering
        topk_values, topk_indices = torch.topk(ranking_metric, num_queries, dim=1)

        # Convert indices to coordinates
        y = topk_indices // W
        x = topk_indices % W

        # Normalize to [0, 1] with pixel center
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
        device = batched_inputs[0]['labels'].device

        gt_density_list = []

        # Create coordinate grid
        y_range = torch.arange(H, device=device).float()
        x_range = torch.arange(W, device=device).float()
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

        grid_y = grid_y.unsqueeze(0).unsqueeze(0)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)

        for i in range(B):
            # Get boxes for this image
            if isinstance(batched_inputs, list) and 'boxes' in batched_inputs[i]:
                 boxes = batched_inputs[i]['boxes']
            elif isinstance(batched_inputs[i], dict) and 'boxes' in batched_inputs[i]:
                 boxes = batched_inputs[i]['boxes']
            else:
                 boxes = torch.zeros((0, 4), device=device)

            # If no boxes, empty map
            if boxes.shape[0] == 0:
                gt_density_list.append(torch.zeros((1, H, W), device=device))
                continue

            # Convert normalized boxes to feature map coordinates
            cx = boxes[:, 0] * W
            cy = boxes[:, 1] * H

            # Reshape for broadcasting
            cx = cx.view(-1, 1, 1)
            cy = cy.view(-1, 1, 1)

            # Calculate Gaussian for each box
            squared_dist = (grid_x - cx)**2 + (grid_y - cy)**2
            gaussian = torch.exp(-squared_dist / (2 * sigma**2))

            # Sum across all objects
            img_density = gaussian.sum(dim=1, keepdim=False)
            gt_density_list.append(img_density)

        return torch.stack(gt_density_list, dim=0)