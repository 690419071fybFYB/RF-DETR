# Boundary-Aware Query Position Refinement 实现报告

## 📋 Idea 概述

**名称**: `boundary_aware_query_refinement`

**目标**: 通过预测目标边界热图，精炼 Query 的参考点位置，使其远离边界、靠近目标中心，从而提升 BBox 回归精度（尤其是 AP@IoU=0.75）。

**动机**: 消融实验显示 Density Init 版本的 AP@IoU=0.75 下降了 2.30%（从 81.90% 降到 79.60%），说明在高 IoU 阈值下回归精度有所下降。

---

## 🏗️ 实现架构

```
                          ┌─────────────────────────────┐
                          │   P3 Features (高分辨率)     │
                          │   [B, 256, 40, 40]          │
                          └─────────────┬───────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        ┌───────────────────────┐              ┌───────────────────────┐
        │  DensityGuidedQueryInit│              │ BoundaryPredictionHead│
        │  预测密度图             │              │  预测边界热图          │
        │  [B, 1, 40, 40]        │              │  [B, 1, 40, 40]       │
        └───────────┬───────────┘              └───────────┬───────────┘
                    │                                       │
                    │                                       ▼
                    │                          ┌───────────────────────┐
                    │                          │ QueryPositionRefinement│
                    │                          │  精炼参考点坐标         │
                    │                          │  输入: query_feat,     │
                    │                          │        ref_points,     │
                    │                          │        boundary_map    │
                    │                          │  输出: refined_ref_pts │
                    │                          └───────────┬───────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                          ┌─────────────────────────────┐
                          │   Transformer Decoder       │
                          │   使用精炼后的参考点         │
                          └─────────────────────────────┘
```

---

## 📁 修改的文件清单

### 1. 新增文件

| 文件路径 | 说明 |
|---------|------|
| `rfdetr/models/boundary_head.py` | 边界预测模块 |
| `experiements/scripts/train_boundary_aware.py` | 训练脚本 |

### 2. 修改的文件

| 文件路径 | 修改内容 |
|---------|---------|
| `rfdetr/config.py` | 添加配置参数 |
| `rfdetr/models/transformer.py` | 集成边界模块，在 decoder 前精炼参考点 |
| `rfdetr/models/lwdetr.py` | 添加边界损失函数 |

---

## 🔧 核心代码实现

### 1. 配置参数 (`rfdetr/config.py`)

```python
# Boundary-Aware Query Position Refinement (改善bbox回归精度)
enable_boundary_refinement: bool = False
boundary_loss_coef: float = 1.0
boundary_refinement_scale: float = 0.05  # 精炼幅度
```

### 2. BoundaryPredictionHead (`rfdetr/models/boundary_head.py`)

```python
class BoundaryPredictionHead(nn.Module):
    """
    边界预测头：从 P3 特征预测目标边界热图
    
    结构: Conv(256→128) → Conv(128→128) → Conv(128→64) → Conv(64→1) → Sigmoid
    输入: P3 特征 [B, C, H, W]
    输出: 边界热图 [B, 1, H, W]，值在 [0, 1] 范围内
    """
```

### 3. QueryPositionRefinement (`rfdetr/models/boundary_head.py`)

```python
class QueryPositionRefinement(nn.Module):
    """
    Query 位置精炼模块
    
    工作流程:
    1. 在参考点的 4 个方向（上下左右）采样边界值
    2. 将 query 特征 + 边界采样值 拼接后送入 MLP
    3. MLP 预测 (dx, dy) 偏移量
    4. 应用偏移精炼参考点
    
    MLP 结构: Linear(D+4 → 128) → ReLU → Linear(128 → 64) → ReLU → Linear(64 → 2)
    
    关键设计:
    - 最后一层初始化为 0，确保训练初期精炼量很小
    - 使用 tanh * refinement_scale 限制精炼幅度
    """
```

### 4. Transformer 集成 (`rfdetr/models/transformer.py`)

**关键位置**: 在调用 Decoder 之前

```python
# Boundary-Aware Query Position Refinement
# 使用边界图精炼参考点的 xy 坐标
if self.enable_boundary_refinement and pred_boundary is not None:
    # refpoint_embed: [B, N, 4] (cx, cy, w, h) 或 unsigmoid 格式
    # 只精炼 cx, cy 部分
    ref_xy = refpoint_embed[..., :2].sigmoid()  # [B, N, 2] 归一化坐标
    refined_xy = self.query_refiner(tgt, ref_xy, pred_boundary)  # [B, N, 2]
    
    # 转回 unsigmoid 空间
    refined_xy = refined_xy.clamp(0.001, 0.999)
    refined_xy_unsigmoid = torch.log(refined_xy / (1 - refined_xy))  # inverse sigmoid
    
    # 更新 refpoint_embed
    refpoint_embed = torch.cat([refined_xy_unsigmoid, refpoint_embed[..., 2:]], dim=-1)
```

### 5. 边界损失 (`rfdetr/models/lwdetr.py`)

```python
def loss_boundary(self, outputs, targets, indices, num_boxes):
    """
    边界损失: BCE Loss
    
    GT 生成:
    - 对每个 GT box，在其四边 (上下左右) 绘制固定宽度的边界线
    - boundary_width = max(1, min(H, W) // 40)  # 约 2.5% 特征图尺寸
    """
    pred_boundary = outputs['pred_boundary']  # [B, 1, H, W]
    
    # 生成 GT 边界图
    gt_boundary = ...  # 根据 targets['boxes'] 生成
    
    # BCE 损失
    loss_boundary = F.binary_cross_entropy(pred_boundary, gt_boundary)
    return {'loss_boundary': loss_boundary * self.boundary_loss_coef}
```

---

## ✅ 验证结果

```bash
# 模型创建测试
Creating model with boundary refinement...
Model created successfully!
Transformer has boundary_head: True
Transformer has query_refiner: True
enable_boundary_refinement: True

# Forward pass 测试
Testing forward pass on GPU...
Forward pass successful!
Output keys: ['pred_logits', 'pred_boxes', 'aux_outputs', 'enc_outputs', 'pred_density', 'pred_boundary']
pred_boundary shape: torch.Size([1, 1, 40, 40])
```

---

## ⚠️ 审查要点

请检查以下关键点:

### 1. 边界采样逻辑 (`boundary_head.py` L99-142)
- 采样偏移量 `delta = 0.02` 是否合理？
- `grid_sample` 的坐标转换 `grid = sample_pts * 2 - 1` 是否正确？

### 2. 参考点精炼 (`transformer.py` L387-400)
- `inverse sigmoid` 计算 `torch.log(x / (1 - x))` 是否正确？
- `clamp(0.001, 0.999)` 是否足够防止数值溢出？
- 只精炼 `[..., :2]` (cx, cy)，保留 `[..., 2:]` (w, h) 是否正确？

### 3. GT 边界图生成 (`lwdetr.py` L565-620)
- 边界宽度计算 `max(1, min(H, W) // 40)` 是否合理？
- 坐标转换 `cx - bw // 2` 是否正确处理了边界情况？

### 4. 损失权重
- `boundary_loss_coef = 1.0` 是否需要调整？
- BCE Loss 是否适合稀疏边界预测？（或应改用 Focal Loss）

---

## 📌 使用方法

```python
from rfdetr import RFDETRBase

model = RFDETRBase(
    enable_density_init=True,           # 启用密度初始化
    enable_boundary_refinement=True,    # 启用边界感知精炼
    boundary_loss_coef=1.0,             # 边界损失权重
    boundary_refinement_scale=0.05,     # 精炼幅度 (最大偏移 5%)
)

model.train(
    dataset_dir="/path/to/dataset",
    epochs=12,
    ...
)
```

---

## 📊 预期效果

- **AP@IoU=0.75**: 预期提升（核心目标）
- **AP@IoU=0.50:0.95**: 预期略有提升
- **AP small**: 对小目标的边界定位应有改善

---

**报告生成时间**: 2025-12-11 18:31
**实现者**: AI Assistant
