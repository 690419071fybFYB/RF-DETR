import torch
import pytest

from rfdetr.models.transformer import (
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    SmallObjectQueryBoost,
    build_transformer,
)
from rfdetr.models.backbone.projector import MiniFPNv3
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.models.lwdetr import SetCriterion


class DummyArgs:
    """Minimal args container for build_transformer."""

    def __init__(self):
        self.hidden_dim = 32
        self.sa_nheads = 4
        self.ca_nheads = 4
        self.num_queries = 8
        self.dropout = 0.0
        self.dim_feedforward = 64
        self.dec_layers = 2
        self.dec_n_points = 1
        self.group_detr = 1
        self.two_stage = False
        self.num_feature_levels = 1
        self.lite_refpoint_refine = False
        self.decoder_norm = "LN"
        self.bbox_reparam = False
        self.use_position_supervised_loss = False
        self.use_varifocal_loss = False
        self.ia_bce_loss = False
        # for matcher/criterion defaults
        self.set_cost_class = 1.0
        self.set_cost_bbox = 1.0
        self.set_cost_giou = 1.0
        self.focal_alpha = 0.25
        self.mask_point_sample_ratio = 16
        self.segmentation_head = False
        self.num_classes = 3
        self.cls_loss_coef = 1.0
        self.bbox_loss_coef = 5.0
        self.giou_loss_coef = 2.0
        self.aux_loss = False
        self.dec_layers = 2
        self.num_select = 100
        self.device = "cpu"
        self.group_detr = 1
        self.enable_small_obj_loss = True
        self.w_small = 2.0


def _dummy_targets(num_targets=2, num_classes=3):
    boxes = torch.rand(num_targets, 4)
    labels = torch.randint(0, num_classes, (num_targets,))
    return [{"boxes": boxes, "labels": labels}]


def test_build_transformer():
    args = DummyArgs()
    transformer = build_transformer(args)
    assert isinstance(transformer, Transformer)


def test_transformer_decoder_layer_with_soqb_forward():
    layer = TransformerDecoderLayer(
        d_model=32,
        sa_nhead=4,
        ca_nhead=4,
        dim_feedforward=64,
        num_feature_levels=1,
        dec_n_points=1,
        enable_soqb=True,
        soqb_boost_factor=2.0,
    )
    bs, num_queries = 2, 4
    tgt = torch.randn(bs, num_queries, 32)
    memory = torch.randn(bs, 1, 32)
    reference_points = torch.rand(bs, num_queries, 1, 4)
    spatial_shapes = torch.tensor([[1, 1]], dtype=torch.long)
    level_start_index = torch.tensor([0], dtype=torch.long)
    query_pos = torch.zeros_like(tgt)

    out = layer(
        tgt,
        memory,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )
    assert out.shape == tgt.shape


def test_small_object_query_boost_module():
    module = SmallObjectQueryBoost(d_model=16, boost_factor=2.0)
    tgt = torch.randn(1, 3, 16, requires_grad=True)
    ref = torch.rand(1, 3, 4)
    out = module(tgt, ref)
    assert out.shape == tgt.shape
    out.sum().backward()
    assert tgt.grad is not None


def test_mini_fpnv3_shape():
    neck = MiniFPNv3(channels=256)
    x = torch.randn(1, 256, 40, 40)
    y = neck(x)
    assert y.shape == x.shape


def test_matcher_returns_indices():
    matcher = HungarianMatcher(enable_small_obj_loss=True, w_small=2.0)
    outputs = {
        "pred_logits": torch.randn(1, 6, 3),
        "pred_boxes": torch.rand(1, 6, 4),
    }
    targets = _dummy_targets(num_targets=3, num_classes=3)
    indices = matcher(outputs, targets, group_detr=1)
    assert len(indices) == 1
    i, j = indices[0]
    assert i.numel() == j.numel()


def test_setcriterion_loss_and_backward_small_obj_weight():
    matcher = HungarianMatcher(enable_small_obj_loss=True, w_small=2.0)
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion = SetCriterion(
        num_classes=3,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=0.25,
        losses=["labels", "boxes"],
        group_detr=1,
        enable_small_obj_loss=True,
        w_small=2.0,
    )
    outputs = {
        "pred_logits": torch.randn(1, 6, 3, requires_grad=True),
        "pred_boxes": torch.rand(1, 6, 4, requires_grad=True),
    }
    targets = _dummy_targets(num_targets=2, num_classes=3)
    losses = criterion(outputs, targets)
    loss = sum(losses.values())
    loss.backward()
    # gradients should exist
    assert outputs["pred_logits"].grad is not None
    assert outputs["pred_boxes"].grad is not None


def test_training_step_with_optimizer():
    class TinyDet(torch.nn.Module):
        def __init__(self, num_queries=4, num_classes=3):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc_logits = torch.nn.Linear(3, num_queries * num_classes)
            self.fc_boxes = torch.nn.Linear(3, num_queries * 4)
            self.num_queries = num_queries
            self.num_classes = num_classes

        def forward(self, x):
            b = x.shape[0]
            feat = self.pool(x).flatten(1)  # [B, 3]
            logits = self.fc_logits(feat).view(b, self.num_queries, self.num_classes)
            boxes = torch.sigmoid(self.fc_boxes(feat)).view(b, self.num_queries, 4)
            return {"pred_logits": logits, "pred_boxes": boxes}

    model = TinyDet()
    matcher = HungarianMatcher(enable_small_obj_loss=True, w_small=2.0)
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion = SetCriterion(
        num_classes=3,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=0.25,
        losses=["labels", "boxes"],
        group_detr=1,
        enable_small_obj_loss=True,
        w_small=2.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    images = torch.randn(1, 3, 640, 640)
    targets = _dummy_targets(num_targets=2, num_classes=3)
    outputs = model(images)
    losses = criterion(outputs, targets)
    loss = sum(losses.values())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def test_transformer_decoder_soqb_end_to_end_minimal():
    layer = TransformerDecoderLayer(
        d_model=32,
        sa_nhead=4,
        ca_nhead=4,
        dim_feedforward=64,
        num_feature_levels=1,
        dec_n_points=1,
        enable_soqb=True,
    )
    decoder = TransformerDecoder(
        decoder_layer=layer,
        num_layers=1,
        norm=torch.nn.LayerNorm(32),
        return_intermediate=False,
        d_model=32,
        lite_refpoint_refine=False,
        bbox_reparam=False,
        enable_soqb=True,
    )
    decoder.bbox_embed = torch.nn.Linear(32, 4)
    tgt = torch.randn(1, 3, 32)
    memory = torch.randn(1, 1, 32)
    ref = torch.rand(1, 3, 4)
    spatial_shapes = torch.tensor([[1, 1]], dtype=torch.long)
    level_start_index = torch.tensor([0], dtype=torch.long)
    valid_ratios = torch.ones(1, 1, 2)

    out = decoder(
        tgt,
        memory,
        refpoints_unsigmoid=ref,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
    )
    assert out.shape == (1, 1, 3, 32) or out.shape == (1, 3, 32)
