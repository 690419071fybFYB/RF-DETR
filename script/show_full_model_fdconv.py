
from rfdetr import RFDETRBase
import torch

def check_model():
    print("Loading full RF-DETR model...")
    rf = RFDETRBase()
    model = rf.model.model # LWDETR
    
    # Locate keys
    print("\nInspecting MultiScaleProjector in Backbone...")
    try:
        # Backbone is wrapped in Joiner
        # Joiner[0] is Backbone
        backbone = model.backbone[0] 
        projector = backbone.projector
        
        # Access the first C2f block in the pyramid
        # projector.stages is a ModuleList of Sequential of C2f...
        # Let's drill down to where we expect FDConv
        
        # Stage 0 -> Sequential -> C2f
        c2f = projector.stages[0][0]
        print(f"\nFound C2f Block type: {type(c2f)}")
        
        # C2f has .m which is ModuleList of Bottlenecks
        bottleneck = c2f.m[0]
        print(f"Found Bottleneck type: {type(bottleneck)}")
        
        # Bottleneck has cv2 which should be ConvX with FDConv
        convx = bottleneck.cv2
        print(f"Found ConvX (cv2) type: {type(convx)}")
        
        # ConvX has .conv
        conv_layer = convx.conv
        print(f"Found Internal Conv Layer type: {type(conv_layer)}")
        
        print("\nLayer String Representation:")
        print(conv_layer)
        
        if "FDConv" in str(type(conv_layer)):
            print("\nVERIFICATION SUCCESS: FDConv is correctly integrated in the full model.")
        else:
            print("\nVERIFICATION FAILED: FDConv not found.")
            
    except Exception as e:
        print(f"Error inspecting model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model()
