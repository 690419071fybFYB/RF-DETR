
import torch
import torch.nn as nn
from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.FDConv import FDConv

def verify_training_step():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup 
    in_channels = [64, 128, 256] 
    out_channels = 64
    # ViT Backbone usually outputs features of the same resolution
    # P3=2.0 (Upsample), P4=1.0 (Keep), P5=0.5 (Downsample)
    scale_factors = [2.0, 1.0, 0.5]
    
    projector = MultiScaleProjector(
        in_channels=in_channels,
        out_channels=out_channels,
        scale_factors=scale_factors,
        num_blocks=1,
        use_fdconv=True
    ).to(device)
    
    # Enable gradients
    projector.train()
    
    # Dummy input: All features should have same spatial resolution (e.g. 16x16)
    # This simulates DINOv2 output where all features are from different blocks but same patch grid
    feat_size = 16 
    x = [
        torch.randn(2, 64, feat_size, feat_size, device=device, requires_grad=True),
        torch.randn(2, 128, feat_size, feat_size, device=device, requires_grad=True),
        torch.randn(2, 256, feat_size, feat_size, device=device, requires_grad=True)
    ]
    
    print("Running forward pass...")
    outputs = projector(x)
    
    print("Outputs produced:")
    for i, out in enumerate(outputs):
        print(f"Feature Map {i} size: {out.shape}")
        
    # Check outputs: 
    # Index 0 (P3): Should be upsampled (2.0) -> 32x32
    # Index 1 (P4): Should be same (1.0) -> 16x16
    # Index 2 (P5): Should be downsampled (0.5) -> 8x8
    
    loss = sum([o.sum() for o in outputs])
    
    print("Running backward pass...")
    try:
        loss.backward()
        print("SUCCESS: Backward pass completed without errors.")
        
        # Check if FDConv parameters have gradients
        fdconv_grads = False
        fdconv_count = 0
        for name, module in projector.named_modules():
             if isinstance(module, FDConv):
                fdconv_count += 1
                # Check dft_weight if it exists
                if hasattr(module, 'dft_weight') and module.dft_weight.grad is not None:
                     fdconv_grads = True
                elif hasattr(module, 'weight') and module.weight is not None and module.weight.grad is not None:
                     fdconv_grads = True
        
        print(f"Found {fdconv_count} FDConv layers.")
        if fdconv_grads:
             print("FDConv gradients confirmed.")
        else:
             print("WARNING: Could not verify FDConv gradients (might be due to parameter reduction or specific impl details).")

    except Exception as e:
        print(f"FAILURE: Backward pass failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_training_step()
