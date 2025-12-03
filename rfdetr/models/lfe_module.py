# ------------------------------------------------------------------------
# RF-DETR - Lightweight Feature Enhancement (LFE) Module
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Lightweight Feature Enhancement (LFE) Module
Integrates edge-aware and Gaussian smoothing features for enhanced detection.
"""

import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath


def build_norm_layer(norm_type, num_features):
    """
    Build normalization layer (replacement for mmcv.cnn.build_norm_layer).
    Returns a tuple (name, layer) to match mmcv API.
    """
    if norm_type == 'BN':
        return 'bn', nn.BatchNorm2d(num_features)
    elif norm_type == 'LN':
        return 'ln', nn.LayerNorm(num_features)
    elif norm_type == 'GN':
        return 'gn', nn.GroupNorm(32, num_features)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


class Conv_Extra(nn.Module):
    """Extra convolution block for feature refinement."""
    def __init__(self, channel, norm_layer, act_layer):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel, 64, 1),
            build_norm_layer(norm_layer, 64)[1],
            act_layer(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            build_norm_layer(norm_layer, 64)[1],
            act_layer(),
            nn.Conv2d(64, channel, 1),
            build_norm_layer(norm_layer, channel)[1],
        )
    
    def forward(self, x):
        return self.block(x)


class Scharr(nn.Module):
    """Scharr edge detection operator."""
    def __init__(self, channel, norm_layer, act_layer):
        super().__init__()
        # Scharr kernels for edge detection
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], 
                                dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], 
                                dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        # Initialize with Scharr kernels
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        
        self.norm = build_norm_layer(norm_layer, channel)[1]
        self.act = act_layer()
        self.conv_extra = Conv_Extra(channel, norm_layer, act_layer)
    
    def forward(self, x):
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        scharr_edge = self.act(self.norm(scharr_edge))
        out = self.conv_extra(x + scharr_edge)
        return out


class Gaussian(nn.Module):
    """Gaussian smoothing module."""
    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        
        # Create Gaussian kernel
        kernel = self.gaussian_kernel(size, sigma)
        kernel = nn.Parameter(data=kernel, requires_grad=False).clone()
        
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, 
                                  padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = kernel.repeat(dim, 1, 1, 1)
        
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()
        
        if feature_extra:
            self.conv_extra = Conv_Extra(dim, norm_layer, act_layer)
    
    def forward(self, x):
        g = self.act(self.norm(self.gaussian(x)))
        return self.conv_extra(x + g) if self.feature_extra else g
    
    def gaussian_kernel(self, size, sigma):
        """Generate 2D Gaussian kernel."""
        return torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(u ** 2 + v ** 2) / (2 * sigma ** 2))
             for u in range(-size // 2 + 1, size // 2 + 1)]
            for v in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)


class LFEA(nn.Module):
    """Lightweight Feature Enhancement Attention."""
    def __init__(self, channel, norm_layer, act_layer):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, dilation=1, bias=False),
            build_norm_layer(norm_layer, channel)[1],
            act_layer()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = build_norm_layer(norm_layer, channel)[1]
    
    def forward(self, c, att):
        """
        Args:
            c: original features
            att: attention map from edge/gaussian
        """
        att = c * att + c
        att = self.conv2d(att)
        wei = self.avg_pool(att)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = self.norm(c + att * wei)
        return x


class LFE_Module(nn.Module):
    """
    Lightweight Feature Enhancement Module.
    Combines edge detection (Scharr) or Gaussian smoothing with LFEA attention.
    """
    def __init__(self, dim, stage, mlp_ratio, drop_path, act_layer, norm_layer):
        super().__init__()
        self.stage = stage
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        )
        
        self.LFEA = LFEA(dim, norm_layer, act_layer)
        
        # Stage 0: use Scharr edge detection (shallow layers)
        # Stage 1+: use Gaussian smoothing (deeper layers)
        if stage == 0:
            self.Scharr_edge = Scharr(dim, norm_layer, act_layer)
        else:
            self.gaussian = Gaussian(dim, 5, 1.0, norm_layer, act_layer)
        
        self.norm = build_norm_layer(norm_layer, dim)[1]
    
    def forward(self, x):
        # Apply edge or Gaussian based on stage
        att = self.Scharr_edge(x) if self.stage == 0 else self.gaussian(x)
        
        # LFEA attention
        x_att = self.LFEA(x, att)
        
        # MLP with residual and drop path
        x = x + self.norm(self.drop_path(self.mlp(x_att)))
        return x


class BasicStage(nn.Module):
    """
    Basic stage that stacks multiple LFE_Module blocks.
    """
    def __init__(self, dim, stage, depth, mlp_ratio, drop_path, norm_layer, act_layer):
        super().__init__()
        self.blocks = nn.Sequential(*[
            LFE_Module(dim=dim, stage=stage, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                       norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
    
    def forward(self, x):
        return self.blocks(x)


# ============================================================================
# Optional: LoG-Stem Layer (for input preprocessing)
# ============================================================================

class LoGFilter(nn.Module):
    """Laplacian of Gaussian filter for edge enhancement."""
    def __init__(self, in_c, out_c, kernel_size, sigma, norm_layer, act_layer):
        super().__init__()
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        
        # Create LoG kernel
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * \
                 torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.sum()
        log_kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, 
                            padding=int(kernel_size // 2), groups=out_c, bias=False)
        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        
        self.act = act_layer()
        self.norm1 = build_norm_layer(norm_layer, out_c)[1]
        self.norm2 = build_norm_layer(norm_layer, out_c)[1]
    
    def forward(self, x):
        x = self.conv_init(x)
        LoG = self.LoG(x)
        LoG_edge = self.act(self.norm1(LoG))
        x = self.norm2(x + LoG_edge)
        return x


class DRFD(nn.Module):
    """Dual-Router Feature Downsampling."""
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = act_layer()
        self.norm_c = build_norm_layer(norm_layer, dim * 2)[1]
        
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = build_norm_layer(norm_layer, dim * 2)[1]
        
        self.fusion = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, stride=1)
        self.gaussian = Gaussian(dim * 2, 5, 0.5, norm_layer, act_layer, feature_extra=False)
        self.norm_g = build_norm_layer(norm_layer, dim * 2)[1]
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm_g(x + self.gaussian(x))
        m = self.norm_m(self.max_m(x))
        c = self.norm_c(self.act_c(self.conv_c(x)))
        x = torch.cat([c, m], dim=1)
        x = self.fusion(x)
        return x


class LoGStem(nn.Module):
    """
    LoG-Stem layer for input preprocessing.
    Performs edge enhancement and noise-robust downsampling.
    """
    def __init__(self, in_chans, stem_dim, act_layer, norm_layer):
        super().__init__()
        out_c14 = int(stem_dim / 4)
        out_c12 = int(stem_dim / 2)
        
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12),
            build_norm_layer(norm_layer, out_c12)[1],
        )
        
        self.LoG = LoGFilter(in_chans, out_c14, 7, 1.0, norm_layer, act_layer)
        self.gaussian = Gaussian(out_c12, 9, 0.5, norm_layer, act_layer)
        self.norm = build_norm_layer(norm_layer, out_c12)[1]
        self.drfd = DRFD(out_c12, norm_layer, act_layer)
    
    def forward(self, x):
        x = self.LoG(x)
        x = self.Conv_D(x)
        x = self.norm(x + self.gaussian(x))
        x = self.drfd(x)
        return x
