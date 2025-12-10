# RF-DETR 密度图可视化功能

本功能允许在训练过程中可视化模型学习的密度图，帮助理解模型如何处理不同密度区域的物体检测。

## 功能特性

- **实时可视化**：训练过程中自动生成密度图可视化
- **边界框叠加**：将真值边界框叠加在密度图上，便于理解密度分布
- **统计信息记录**：保存密度图的统计信息用于分析
- **灵活配置**：可调整可视化间隔和样本数量

## 密度图输出说明

RF-DETR的密度图是模型预测的热图，表示图像中物体可能出现的位置密度：
- **高密度区域**（红色）：模型认为这里可能有多个或重要物体
- **低密度区域**（透明）：模型认为这里不太可能有物体

## 使用方法

### 1. 在训练脚本中启用

```python
from rfdetr.main import Model

model = Model()

# 训练时启用密度图可视化
model.train(
    dataset_dir="path/to/dataset",
    num_classes=80,
    epochs=100,

    # 启用密度相关功能
    enable_density_init=True,
    enable_density_augmented_cross_attn=True,

    # 启用可视化
    visualize_density=True,
    density_vis_max_samples=4,  # 每批次可视化的样本数
    density_vis_interval=50,    # 每50步可视化一次
)
```

### 2. 通过命令行使用

```bash
python rfdetr/main.py \
    --dataset_dir /path/to/coco \
    --num_classes 80 \
    --epochs 100 \
    --batch_size 8 \
    --enable_density_init \
    --enable_density_augmented_cross_attn \
    --visualize_density \
    --density_vis_interval 50 \
    --output_dir outputs/my_experiment
```

### 3. 测试可视化功能

运行测试脚本验证功能是否正常：

```bash
python test_density_visualization.py
```

## 输出文件结构

训练启用了密度图可视化后，会在输出目录生成以下结构：

```
outputs/
└── your_experiment/
    ├── density_visualizations/
    │   ├── epoch_0000/
    │   │   ├── step_000000_sample_00.png
    │   │   ├── step_000050_sample_00.png
    │   │   └── density_stats.json
    │   ├── epoch_0001/
    │   │   └── ...
    │   └── ...
    └── checkpoint.pth
```

### 可视化文件说明

每个PNG文件包含三个子图：
1. **原始图像**：显示输入图像和真值边界框（绿色）
2. **密度图**：模型预测的密度热图
3. **叠加图**：密度图与图像的叠加，边界框用亮绿色标记

### 统计文件

`density_stats.json`记录了每次可视化的统计信息：
```json
{
  "epoch": 0,
  "step": 0,
  "statistics": {
    "predicted": {
      "mean": 0.1234,
      "std": 0.0567,
      "min": 0.0,
      "max": 0.9876,
      "shape": [2, 1, 40, 40]
    }
  }
}
```

## 配置参数详解

- `visualize_density`: 是否启用密度图可视化
- `density_vis_max_samples`: 每批次最多可视化的样本数量（默认4）
- `density_vis_interval`: 可视化间隔，每隔多少步生成一次可视化（默认50）

## 注意事项

1. **性能影响**：可视化会增加训练时间，建议只在调试或分析时使用
2. **存储空间**：可视化文件会占用磁盘空间，定期清理旧的可视化结果
3. **依赖项**：确保安装了matplotlib和cv2（opencv-python）
4. **分布式训练**：可视化只在主进程进行，避免重复生成

## 常见问题

### Q: 为什么没有生成密度图？
A: 请确保：
- 启用了至少一个密度相关功能（如`enable_density_init`）
- 模型配置中包含了密度预测模块
- 输出中确实包含`pred_density`键

### Q: 如何调整可视化频率？
A: 修改`density_vis_interval`参数，值越大生成频率越低

### Q: 如何可视化自定义数据集的类别？
A: 在训练时提供`class_names`字典，将类别ID映射到类别名称

## 示例分析

通过观察密度图可视化，您可以：

1. **验证模型学习**：密度图是否集中在物体位置
2. **发现数据偏差**：某些区域是否持续高/低密度
3. **调试模型问题**：异常的密度分布可能指示训练问题
4. **优化超参数**：根据密度图调整密度相关模块的参数

## 进阶使用

### 自定义可视化

可以继承`DensityVisualizer`类来实现自定义的可视化逻辑：

```python
from rfdetr.util.density_visualizer import DensityVisualizer

class CustomDensityVisualizer(DensityVisualizer):
    def visualize_density_map(self, image, density_map, boxes=None, ...):
        # 实现自定义可视化逻辑
        super().visualize_density_map(image, density_map, boxes, ...)
```