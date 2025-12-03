#!/usr/bin/env python3
"""
Test script to verify LFE module integration.
Run this to ensure all components are working correctly.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rfdetr.models.lfe_module import (
    Scharr, Gaussian, LFEA, LFE_Module, BasicStage, 
    LoGFilter, DRFD, LoGStem
)


def test_scharr():
    """Test Scharr edge detection."""
    print("Testing Scharr edge detection...")
    x = torch.randn(2, 256, 40, 40)
    scharr = Scharr(channel=256, norm_layer='BN', act_layer=nn.GELU)
    out = scharr(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"✅ Scharr: Input {x.shape} → Output {out.shape}")


def test_gaussian():
    """Test Gaussian smoothing."""
    print("\nTesting Gaussian smoothing...")
    x = torch.randn(2, 256, 40, 40)
    gaussian = Gaussian(dim=256, size=5, sigma=1.0, norm_layer='BN', 
                       act_layer=nn.GELU, feature_extra=True)
    out = gaussian(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"✅ Gaussian: Input {x.shape} → Output {out.shape}")


def test_lfea():
    """Test LFEA attention."""
    print("\nTesting LFEA attention...")
    x = torch.randn(2, 256, 40, 40)
    att = torch.randn(2, 256, 40, 40)
    lfea = LFEA(channel=256, norm_layer='BN', act_layer=nn.GELU)
    out = lfea(x, att)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"✅ LFEA: Input {x.shape} + Attention {att.shape} → Output {out.shape}")


def test_lfe_module():
    """Test LFE_Module (stage 0 and 1)."""
    print("\nTesting LFE_Module...")
    x = torch.randn(2, 256, 40, 40)
    
    # Stage 0 (Scharr)
    lfe0 = LFE_Module(dim=256, stage=0, mlp_ratio=4.0, drop_path=0.1,
                      norm_layer='BN', act_layer=nn.GELU)
    out0 = lfe0(x)
    assert out0.shape == x.shape, f"Expected {x.shape}, got {out0.shape}"
    print(f"✅ LFE_Module (stage=0, Scharr): {x.shape} → {out0.shape}")
    
    # Stage 1 (Gaussian)
    lfe1 = LFE_Module(dim=256, stage=1, mlp_ratio=4.0, drop_path=0.1,
                      norm_layer='BN', act_layer=nn.GELU)
    out1 = lfe1(x)
    assert out1.shape == x.shape, f"Expected {x.shape}, got {out1.shape}"
    print(f"✅ LFE_Module (stage=1, Gaussian): {x.shape} → {out1.shape}")


def test_basic_stage():
    """Test BasicStage (stacked LFE modules)."""
    print("\nTesting BasicStage...")
    x = torch.randn(2, 256, 40, 40)
    
    stage = BasicStage(
        dim=256, stage=0, depth=3, mlp_ratio=4.0,
        drop_path=[0.1, 0.2, 0.3], norm_layer='BN', act_layer=nn.GELU
    )
    out = stage(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"✅ BasicStage (depth=3): {x.shape} → {out.shape}")


def test_log_filter():
    """Test LoG filter."""
    print("\nTesting LoG filter...")
    x = torch.randn(2, 3, 640, 640)
    log_filter = LoGFilter(in_c=3, out_c=64, kernel_size=7, sigma=1.0,
                           norm_layer='BN', act_layer=nn.GELU)
    out = log_filter(x)
    expected_shape = (2, 64, 640, 640)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print(f"✅ LoGFilter: {x.shape} → {out.shape}")


def test_drfd():
    """Test DRFD downsampling."""
    print("\nTesting DRFD...")
    x = torch.randn(2, 64, 160, 160)
    drfd = DRFD(dim=64, norm_layer='BN', act_layer=nn.GELU)
    out = drfd(x)
    expected_shape = (2, 128, 80, 80)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print(f"✅ DRFD: {x.shape} → {out.shape}")


def test_log_stem():
    """Test complete LoG-Stem."""
    print("\nTesting LoG-Stem...")
    x = torch.randn(2, 3, 640, 640)
    stem = LoGStem(in_chans=3, stem_dim=256, act_layer=nn.GELU, norm_layer='BN')
    out = stem(x)
    # After 2x downsampling: 640 -> 320 -> 160
    expected_shape = (2, 256, 160, 160)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print(f"✅ LoGStem: {x.shape} → {out.shape}")


def test_parameter_count():
    """Test parameter count for different configurations."""
    print("\n" + "="*60)
    print("Parameter Count Analysis")
    print("="*60)
    
    configs = [
        ("LFE_Module (stage=0, depth=1)", 
         lambda: LFE_Module(256, 0, 4.0, 0.1, nn.GELU, 'BN')),
        ("LFE_Module (stage=1, depth=1)", 
         lambda: LFE_Module(256, 1, 4.0, 0.1, nn.GELU, 'BN')),
        ("BasicStage (stage=0, depth=2)", 
         lambda: BasicStage(256, 0, 2, 4.0, [0.1, 0.2], 'BN', nn.GELU)),
        ("BasicStage (stage=1, depth=3)", 
         lambda: BasicStage(256, 1, 3, 4.0, [0.1, 0.2, 0.3], 'BN', nn.GELU)),
    ]
    
    for name, module_fn in configs:
        module = module_fn()
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name:40s}: {params/1e6:6.2f}M params ({trainable/1e6:6.2f}M trainable)")


def main():
    print("="*60)
    print("LFE Module Test Suite")
    print("="*60)
    
    try:
        test_scharr()
        test_gaussian()
        test_lfea()
        test_lfe_module()
        test_basic_stage()
        test_log_filter()
        test_drfd()
        test_log_stem()
        test_parameter_count()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nLFE module is ready for integration into RF-DETR.")
        print("See lfe_integration_guide.md for next steps.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
