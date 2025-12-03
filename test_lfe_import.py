#!/usr/bin/env python3
"""
Simple import test for LFE module.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("Testing LFE module imports...")

try:
    from rfdetr.models.lfe_module import (
        Scharr, Gaussian, LFEA, LFE_Module, BasicStage, 
        LoGFilter, DRFD, LoGStem, Conv_Extra, build_norm_layer
    )
    print("✅ All LFE module components imported successfully!")
    print("\nAvailable components:")
    print("  - Conv_Extra: Feature refinement block")
    print("  - Scharr: Edge detection operator")
    print("  - Gaussian: Gaussian smoothing")
    print("  - LFEA: Lightweight Feature Enhancement Attention")
    print("  - LFE_Module: Complete LFE module")
    print("  - BasicStage: Stacked LFE modules")
    print("  - LoGFilter: Laplacian of Gaussian filter")
    print("  - DRFD: Dual-Router Feature Downsampling")
    print("  - LoGStem: Complete preprocessing stem")
    
    print("\n✅ LFE module is ready for integration!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
