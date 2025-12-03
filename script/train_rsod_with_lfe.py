"""
RF-DETR Training with LFE Module
=================================

This script demonstrates training RF-DETR with the LFE (Lightweight Feature Enhancement) module enabled.
Run this after baseline training to compare performance improvements.
"""

from rfdetr import RFDETRBase

# Initialize model with LFE enabled
model = RFDETRBase(
    use_dynamic_query=True,
    use_lfe=True,           # Enable LFE module
    lfe_depth=2,           # Number of LFE blocks per scale
    lfe_mlp_ratio=4.0      # MLP expansion ratio
)

# Dataset configuration
dataset = "/home/fyb/datasets/RSOD_cocoFormat"
output_dir = "/home/fyb/mydir/rf-detr/script/RSOD_results_LFE"

print("="*70)
print("RF-DETR Training with LFE Module")
print("="*70)
print(f"Dataset: {dataset}")
print(f"Output directory: {output_dir}")
print(f"LFE enabled: True")
print(f"LFE depth: 2")
print(f"LFE MLP ratio: 4.0")
print("="*70)

# Train with LFE
model.train(
    dataset_dir=dataset,
    dataset_file="coco",
    coco_path=dataset,
    epochs=200,
    batch_size=6,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=output_dir,
    early_stopping=True,
    early_stopping_patience=5,
)

print("\n" + "="*70)
print("Training completed!")
print("="*70)
print(f"Results saved to: {output_dir}")
print("\nNext steps:")
print("1. Compare results with baseline (RSOD_results3)")
print("2. Check metrics in results.json")
print("3. Visualize improvements in metrics_plot.png")
