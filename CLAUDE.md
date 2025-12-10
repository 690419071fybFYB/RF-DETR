# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Training and Fine-tuning
- `python rfdetr/main.py --dataset_dir path/to/dataset --num_classes N -batch_size B --epochs E` - Train a model with custom dataset
- `python rfdetr/main.py --eval --resume path/to/checkpoint.pth --dataset_dir path/to/dataset` - Evaluate a checkpoint
- `python rfdetr/main.py --resume path/to/checkpoint.pth --dataset_dir path/to/dataset` - Resume training from checkpoint

### CLI Tool
- `rfdetr train` - Access the training functionality via CLI
- Training uses configuration classes from `rfdetr.config.py` (RFDETRBaseConfig, RFDETRNanoConfig, etc.)

### Testing and Benchmarking
- `python -m pytest` (if pytest is configured) - Run tests
- Add `--do_benchmark` flag to training command for model benchmarking

### ONNX Export
- Install with `pip install rfdetr[onnxexport]` for export functionality
- `model.export(output_dir="onnx_output", simplify=True)` - Export trained model to ONNX

## Architecture Overview

RF-DETR is a real-time detection transformer built on top of LW-DETR and DINOv2 architectures with several key innovations:

### Core Components

1. **Backbone Architecture** (`rfdetr/models/backbone/`)
   - DINOv2 with windowed attention (`dinov2_with_windowed_attn.py`)
   - Supports small/base encoder variants
   - Configurable feature extraction at multiple scales

2. **Model Architecture** (`rfdetr/models/`)
   - `lwdetr.py` - Main model implementation
   - `transformer.py` - Transformer encoder-decoder
   - `density_init.py` - Density-Guided Query Initialization
   - `density_cross_attn.py` - Density-Augmented Cross-Attention
   - `density_pos_bias.py` - Density Positional Bias Modulation
   - `density_sampling_offset.py` - Density Sampling Offset Modulation
   - `scale_aware_attn.py` - Scale-Aware Query Grouping
   - `segmentation_head.py` - Instance segmentation head

3. **Training Infrastructure** (`rfdetr/main.py`)
   - `Model` class encapsulates training, evaluation, and export
   - Supports EMA (Exponential Moving Average)
   - Early stopping with configurable patience
   - Automatic mixed precision (AMP) support
   - Gradient accumulation for large effective batch sizes

4. **Configuration System** (`rfdetr/config.py`)
   - Pydantic-based configuration classes
   - Model variants: Nano, Small, Medium, Base, Large, Seg-Preview
   - Extensible feature flags for new architectural components

### Key Architectural Features

- **Multi-Scale Processing**: Configurable projector scales (P3, P4, P5) for feature pyramid
- **Group DETR**: Speed optimization through grouped query processing
- **Density-Based Enhancements**: Novel modules for handling varying object densities
- **Scale-Aware Mechanisms**: Adaptive processing for different object sizes
- **Two-Stage Processing**: Optional two-stage detection pipeline

### Dataset Support

- **COCO Format**: Standard COCO dataset integration
- **Roboflow Format**: Native support for Roboflow datasets
- **Objects365**: Pre-trained weights available
- **Custom Datasets**: Flexible dataset loading pipeline in `rfdetr/datasets/`

### Model Training Pipeline

1. **Model Building**: `rfdetr/models/__init__.py` - `build_model()` function
2. **Loss Functions**: Hungarian matching with configurable cost coefficients
3. **Optimization**: AdamW with cosine/step scheduling, layer-wise decay
4. **Evaluation**: COCO metrics integration with comprehensive logging

### Key Files to Understand

- `rfdetr/main.py:87-160` - Model initialization and weight loading
- `rfdetr/main.py:165-567` - Complete training loop with all features
- `rfdetr/config.py:13-62` - Base model configuration with feature flags
- `rfdetr/models/lwdetr.py` - Core model architecture
- `rfdetr/engine.py` - Training and evaluation functions

### Development Notes

- Python 3.9+ required with PyTorch 1.13+
- CUDA/MPS automatic device detection
- Distributed training support built-in
- Extensive logging with TensorBoard/W&B integration
- Model checkpointing with automatic best model selection