# LFE Integration Quick Reference

## Branch Information
```bash
Branch: feature/lfe-integration
Commit: 77e19c1
```

## Quick Start

### 1. Switch to LFE branch
```bash
cd /home/fyb/mydir/rf-detr
git checkout feature/lfe-integration
```

### 2. Activate environment
```bash
conda activate ai_scientist
```

### 3. Run baseline training
```bash
python script/train_rsod.py
```

### 4. Run LFE training
```bash
python script/train_rsod_with_lfe.py
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_lfe` | `False` | Enable/disable LFE module |
| `lfe_depth` | `2` | Number of LFE blocks per scale |
| `lfe_mlp_ratio` | `4.0` | MLP expansion ratio |

## Python API

```python
from rfdetr import RFDETRBase

# Without LFE (baseline)
model = RFDETRBase()

# With LFE (recommended)
model = RFDETRBase(use_lfe=True, lfe_depth=2, lfe_mlp_ratio=4.0)

# Lightweight LFE
model = RFDETRBase(use_lfe=True, lfe_depth=1)

# Strong LFE
model = RFDETRBase(use_lfe=True, lfe_depth=3, lfe_mlp_ratio=6.0)
```

## Files Modified

- ✅ `rfdetr/models/lfe_module.py` (new)
- ✅ `rfdetr/models/lwdetr.py` (modified)
- ✅ `rfdetr/config.py` (modified)

## Validation Status

✅ All integration tests passed with `ai_scientist` environment
✅ Syntax checks passed
✅ Git commit completed

## Expected Benefits

- Small object detection: +1-2 AP
- Low-contrast targets: +1-3 AP
- Computational overhead: ~5-10%
