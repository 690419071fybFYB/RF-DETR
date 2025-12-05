
import torch
from rfdetr.models.backbone.projector import MultiScaleProjector

def verify_architecture():
    # Setup dummy arguments similar to standard usage
    in_channels = [512, 1024, 2048]
    out_channels = 256
    scale_factors = [2.0, 1.0, 0.5]
    
    projector = MultiScaleProjector(
        in_channels=in_channels,
        out_channels=out_channels,
        scale_factors=scale_factors,
        num_blocks=1,
        use_fdconv=True
    )
    
    print(projector)
    
    # Check if FDConv is present
    has_fdconv = False
    for name, module in projector.named_modules():
        if "FDConv" in str(type(module)):
            has_fdconv = True
            break
            
    if has_fdconv:
        print("\nSUCCESS: FDConv found in the architecture!")
    else:
        print("\nFAILURE: FDConv NOT found in the architecture.")

if __name__ == "__main__":
    verify_architecture()
