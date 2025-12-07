#!/usr/bin/env python3
"""
Script to print model structure using torchinfo.summary
Works with the develop branch (rfdetr package structure)
"""
import torch
from torchinfo import summary
from rfdetr import RFDETRBase

def main():
    print("Initializing RF-DETR Base model...")
    
    # Initialize model without pretrained weights for structure inspection
    model = RFDETRBase(
        pretrain_weights=None,  # Don't load weights, just inspect structure
        num_classes=4,  # RSOD has 4 classes
    )
    
    # Move to CPU to avoid GPU memory issues during inspection
    model.model.model.cpu()
    
    print("\n" + "="*80)
    print("RF-DETR MODEL STRUCTURE SUMMARY")
    print("="*80 + "\n")
    
    # Print detailed model summary
    summary(
        model.model.model,
        input_size=(1, 3, 560, 560),  # Must be divisible by 56 (block_size)
        depth=4,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
        verbose=1
    )
    
    print("\n" + "="*80)
    print("PARAMETER STATISTICS")
    print("="*80)
    
    total_params = sum(p.numel() for p in model.model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.model.parameters() if p.requires_grad)
    
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Frozen Parameters:     {total_params - trainable_params:,}")
    print(f"Model Size (MB):       {total_params * 4 / 1024 / 1024:.2f}")
    print("="*80)
    
    # Print key model components
    print("\nKEY MODEL COMPONENTS:")
    print("-" * 80)
    for name, module in model.model.model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name:30s} : {num_params:>15,} parameters")
    print("="*80)

if __name__ == "__main__":
    main()