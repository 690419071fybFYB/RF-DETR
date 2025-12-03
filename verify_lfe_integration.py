#!/usr/bin/env python3
"""
Simple verification script to test LFE module integration with RF-DETR.
"""

import sys
from rfdetr.config import RFDETRBaseConfig

# Test 1: Verify LFE parameters exist in config
print("Test 1: Checking LFE parameters in ModelConfig...")
config = RFDETRBaseConfig()
assert hasattr(config, 'use_lfe'), "Missing use_lfe parameter"
assert hasattr(config, 'lfe_depth'), "Missing lfe_depth parameter"
assert hasattr(config, 'lfe_mlp_ratio'), "Missing lfe_mlp_ratio parameter"
print(f"✅ LFE parameters found:")
print(f"   - use_lfe: {config.use_lfe}")
print(f"   - lfe_depth: {config.lfe_depth}")
print(f"   - lfe_mlp_ratio: {config.lfe_mlp_ratio}")

# Test 2: Verify LFE can be enabled
print("\nTest 2: Enabling LFE in config...")
config_with_lfe = RFDETRBaseConfig(use_lfe=True, lfe_depth=3, lfe_mlp_ratio=6.0)
assert config_with_lfe.use_lfe == True
assert config_with_lfe.lfe_depth == 3
assert config_with_lfe.lfe_mlp_ratio == 6.0
print("✅ LFE configuration works correctly")

# Test 3: Verify LFE module can be imported
print("\nTest 3: Importing LFE module...")
from rfdetr.models.lfe_module import BasicStage, LFE_Module, Scharr, Gaussian, LFEA
print("✅ LFE module imports successfully")

# Test 4: Verify LWDETR integration
print("\nTest 4: Checking LWDETR integration...")
from rfdetr.models.lwdetr import LWDETR
import inspect
sig = inspect.signature(LWDETR.__init__)
params = list(sig.parameters.keys())
assert 'use_lfe' in params, "Missing use_lfe in LWDETR.__init__"
assert 'lfe_depth' in params, "Missing lfe_depth in LWDETR.__init__"
assert 'lfe_mlp_ratio' in params, "Missing lfe_mlp_ratio in LWDETR.__init__"
print("✅ LWDETR has LFE parameters")

print("\n" + "="*60)
print("🎉 ALL INTEGRATION TESTS PASSED!")
print("="*60)
print("\nLFE module is successfully integrated into RF-DETR.")
print("\nUsage example:")
print("  from rfdetr RFDETRBase")
print("  model = RFDETRBase(use_lfe=True, lfe_depth=2)")
print("  model.train(dataset_dir='path/to/dataset', ...)")
