# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""
Scale-Aware Multi-Scale Deformable Attention Module

This module extends MSDeformAttn to add scale-aware query grouping.
Each query predicts its target scale bin (small/medium/large) and
attends only to corresponding feature pyramid levels.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from .ops.modules import MSDeformAttn
from .ops.functions import ms_deform_attn_core_pytorch


class ScaleAwareMSDeformAttn(nn.Module):
    """
    Scale-Aware Multi-Scale Deformable Attention Module.
    
    Extends MSDeformAttn by adding a scale prediction head that predicts
    which scale bin (small/medium/large) each query should attend to.
    Based on the predicted scale, attention weights are masked to focus
    on corresponding feature pyramid levels.
    
    Scale bin to level mapping (for 4-level pyramid):
    - Bin 0 (small objects): attend to levels 0, 1 (high resolution)
    - Bin 1 (medium objects): attend to all levels
    - Bin 2 (large objects): attend to levels 2, 3 (low resolution)
    """
    
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, num_scale_bins=3):
        """
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        :param num_scale_bins number of scale bins (default 3: small/medium/large)
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.num_scale_bins = num_scale_bins
        
        # Main attention components (same as MSDeformAttn)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Scale prediction head
        self.scale_pred_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_scale_bins)
        )
        
        # Create level mask for each scale bin
        # Shape: (num_scale_bins, n_levels)
        self._create_level_masks()
        
        self._reset_parameters()
        self._export = False
        
    def _create_level_masks(self):
        """Create attention level masks for each scale bin."""
        # level_masks[bin_idx, level_idx] = 1 if query in bin should attend to level
        level_masks = torch.zeros(self.num_scale_bins, self.n_levels)
        
        if self.n_levels == 4:
            # Small objects (bin 0): attend to high-res levels (0, 1)
            level_masks[0, 0] = 1.0
            level_masks[0, 1] = 1.0
            # Medium objects (bin 1): attend to all levels
            level_masks[1, :] = 1.0
            # Large objects (bin 2): attend to low-res levels (2, 3)
            level_masks[2, 2] = 1.0
            level_masks[2, 3] = 1.0
        elif self.n_levels == 3:
            # Small: level 0
            level_masks[0, 0] = 1.0
            # Medium: all levels
            level_masks[1, :] = 1.0
            # Large: levels 1, 2
            level_masks[2, 1] = 1.0
            level_masks[2, 2] = 1.0
        else:
            # Fallback: all bins attend to all levels
            level_masks[:, :] = 1.0
            
        self.register_buffer('level_masks', level_masks)
    
    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
        
        # Initialize scale prediction head
        for layer in self.scale_pred_head:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight.data)
                constant_(layer.bias.data, 0.)

    def export(self):
        """Export mode."""
        self._export = True

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None, return_scale_logits=False):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2) or (N, Length_{query}, n_levels, 4)
        :param input_flatten               (N, sum_{l=0}^{L-1} H_l * W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, )
        :param input_padding_mask          (N, sum_{l=0}^{L-1} H_l * W_l)
        :param return_scale_logits         If True, also return scale prediction logits

        :return output                     (N, Length_{query}, C)
        :return scale_logits               (N, Length_{query}, num_scale_bins) if return_scale_logits
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        
        # Predict scale bin for each query
        scale_logits = self.scale_pred_head(query)  # (N, Len_q, num_scale_bins)
        
        # Get soft scale assignment (for differentiable masking during training)
        if self.training:
            # Use soft assignment with temperature
            scale_probs = F.softmax(scale_logits / 0.5, dim=-1)  # (N, Len_q, num_scale_bins)
        else:
            # Use hard assignment during inference
            scale_bins = scale_logits.argmax(dim=-1)  # (N, Len_q)
            scale_probs = F.one_hot(scale_bins, self.num_scale_bins).float()
        
        # Compute level weights from scale predictions
        # level_masks: (num_scale_bins, n_levels)
        # scale_probs: (N, Len_q, num_scale_bins)
        # level_weights: (N, Len_q, n_levels)
        level_weights = torch.einsum('bqs,sl->bql', scale_probs, self.level_masks)
        
        # Project values
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        
        # Compute sampling offsets and attention weights
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        
        # Apply scale-aware level masking to attention weights before softmax
        # Reshape level_weights to match attention_weights shape
        # level_weights: (N, Len_q, n_levels) -> (N, Len_q, 1, n_levels, 1) -> expand to n_points
        level_mask_expanded = level_weights.unsqueeze(2).unsqueeze(-1)  # (N, Len_q, 1, n_levels, 1)
        level_mask_expanded = level_mask_expanded.expand(-1, -1, self.n_heads, -1, self.n_points)
        level_mask_expanded = level_mask_expanded.reshape(N, Len_q, self.n_heads, -1)  # (N, Len_q, n_heads, n_levels*n_points)
        
        # Apply mask: set masked positions to large negative value before softmax
        masked_attention_weights = attention_weights + (1 - level_mask_expanded) * (-1e9)
        attention_weights = F.softmax(masked_attention_weights, dim=-1)
        
        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        # Compute output using deformable attention core
        value = value.transpose(1, 2).contiguous().view(N, self.n_heads, self.d_model // self.n_heads, Len_in)
        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        
        if return_scale_logits:
            return output, scale_logits
        return output
