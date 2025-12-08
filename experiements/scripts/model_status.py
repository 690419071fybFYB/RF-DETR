import sys
from rfdetr import RFDETRBase
from rfdetr import RFDETRMedium
from torchinfo import summary

# Open file to save output
output_file = "/home/fyb/mydir/rf-detr/experiements/results/model_summary.txt"
with open(output_file, "w") as f:
    # Redirect stdout to file
    original_stdout = sys.stdout
    sys.stdout = f
    
    print("=" * 80)
    print("RF-DETR Medium Model Summary")
    print("=" * 80)
    model = RFDETRMedium()
    summary(model.model.model, input_size=(1, 3, 576, 576), device="cuda", col_names=[
        "input_size",
        "output_size",
        "num_params",
        "params_percent",
        "kernel_size",
        "mult_adds",
        "trainable",
    ], depth=7)
    
    print("\n" + "=" * 80)
    print("RF-DETR Base Model Summary")
    print("=" * 80)
    model = RFDETRBase(projector_scale=["P3", "P4", "P5"],    pretrain_exclude_keys=[
        "backbone.0.projector*",
        "transformer.decoder.layers*",
    ])
    summary(model.model.model, input_size=(1, 3, 560, 560), device="cuda", col_names=[
        "input_size",
        "output_size",
        "num_params",
        "params_percent",
        "kernel_size",
        "mult_adds",
        "trainable",
    ], depth=7)
    
    # Restore stdout
    sys.stdout = original_stdout

print(f"Model summary saved to: {output_file}")
