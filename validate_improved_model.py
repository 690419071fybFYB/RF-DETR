#!/usr/bin/env python3
"""
验证改进后的RF-DETR模型
- 检查模型是否能正确初始化
- 验证前向传播
- 测试改进的密度预测头
"""

import torch
import numpy as np
from pathlib import Path
from rfdetr.models import build_model
from rfdetr.config import RFDETRBaseConfig


def create_mock_args():
    """创建模拟的训练参数"""
    class MockArgs:
        def __init__(self):
            # 基本配置
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.num_classes = 80

            # 密度配置
            self.enable_density_init = True
            self.enable_improved_density = True
            self.density_adaptive_threshold = True

            # 禁用的模块
            self.enable_density_augmented_cross_attn = False
            self.enable_density_positional_bias = False
            self.enable_density_sampling_offset = False

            # 其他配置
            self.enable_small_object_query_boost = True
            self.soqb_boost_factor = 2.0

            # Transformer配置
            self.hidden_dim = 256
            self.num_queries = 300
            self.group_detr = 13

    return MockArgs()


def test_model_initialization():
    """测试模型初始化"""
    print("="*60)
    print("1. 测试模型初始化")
    print("="*60)

    args = create_mock_args()

    try:
        # 使用默认配置
        config = RFDETRBaseConfig()

        # 更新配置以使用改进的密度预测
        config.enable_density_init = True
        config.enable_improved_density = True
        config.density_adaptive_threshold = True

        # 构建模型
        model, _, _, _ = build_model(args)

        print(f"✓ 模型初始化成功")
        print(f"  - 设备: {args.device}")
        print(f"  - 类别数: {config.num_classes}")
        print(f"  - 密度初始化: {'启用' if config.enable_density_init else '禁用'}")
        print(f"  - 改进的UNet预测器: {'启用' if config.enable_improved_density else '禁用'}")

        return model

    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        raise


def test_forward_pass(model):
    """测试前向传播"""
    print("\n" + "="*60)
    print("2. 测试前向传播")
    print("="*60)

    model.eval()
    device = next(model.parameters()).device

    try:
        # 创建模拟输入
        batch_size = 2
        height, width = 640, 640

        # 创建NestedTensor格式的输入
        from rfdetr.util.misc import nested_tensor_from_tensor_list

        # 模拟图片批次 [batch_size, 3, H, W]
        images = torch.randn(batch_size, 3, height, width, device=device)

        # 创建目标（训练时需要）
        targets = []
        for i in range(batch_size):
            target = {
                'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device),  # (cx, cy, w, h)
                'labels': torch.tensor([1], device=device),
            }
            targets.append(target)

        # 包装为NestedTensor
        samples = nested_tensor_from_tensor_list(images)

        print(f"输入形状: {images.shape}")

        # 前向传播
        with torch.no_grad():
            outputs = model(samples, targets)

        print("✓ 前向传播成功")
        print(f"  输出键: {list(outputs.keys())}")
        print(f"  预测logits形状: {outputs['pred_logits'].shape}")
        print(f"  预测boxes形状: {outputs['pred_boxes'].shape}")

        if 'aux_outputs' in outputs:
            print(f"  辅助输出层数: {len(outputs['aux_outputs'])}")

        return True

    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_density_module():
    """单独测试密度预测模块"""
    print("\n" + "="*60)
    print("3. 测试密度预测模块")
    print("="*60)

    try:
        from rfdetr.models.density_init_improved import ImprovedDensityPredictor

        # 创建模块
        hidden_dim = 256
        density_predictor = ImprovedDensityPredictor(hidden_dim)

        # 创建测试输入
        batch_size = 2
        height, width = 40, 40  # P3特征图大小
        features = torch.randn(batch_size, hidden_dim, height, width)

        print(f"输入特征形状: {features.shape}")

        # 前向传播
        with torch.no_grad():
            density_map = density_predictor(features)

        print("✓ 改进的密度预测器测试成功")
        print(f"  密度图形状: {density_map.shape}")
        print(f"  密度值范围: [{density_map.min():.4f}, {density_map.max():.4f}]")
        print(f"  非零像素比例: {(density_map > 0).float().mean():.2%}")

        return True

    except Exception as e:
        print(f"✗ 密度预测器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """测试内存使用情况"""
    print("\n" + "="*60)
    print("4. 内存使用情况")
    print("="*60)

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 初始内存
            initial_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"初始GPU内存: {initial_mem:.2f} GB")

            # 创建并运行模型
            args = create_mock_args()
            config = RFDETRBaseConfig()
            config.enable_density_init = True
            config.enable_improved_density = True

            model, _, _, _ = build_model(args)
            model.cuda()

            # 模型内存
            model_mem = torch.cuda.memory_allocated() / 1024**3
            print(f"模型参数内存: {model_mem - initial_mem:.2f} GB")

            # 前向传播
            batch_size = 4
            height, width = 640, 640
            from rfdetr.util.misc import nested_tensor_from_tensor_list

            images = torch.randn(batch_size, 3, height, width, device='cuda')
            samples = nested_tensor_from_tensor_list(images)

            with torch.no_grad():
                outputs = model(samples)

            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"峰值内存: {peak_mem:.2f} GB")
            print(f"前向传播增加: {peak_mem - model_mem:.2f} GB")

        else:
            print("CUDA不可用，跳过内存测试")

        return True

    except Exception as e:
        print(f"✗ 内存测试失败: {e}")
        return False


def main():
    print("\n🚀 验证改进的RF-DETR模型")
    print("改进内容:")
    print("1. 简化模型架构 - 只保留Density Init")
    print("2. 使用UNet改进密度预测头")
    print("3. 移除冗余的密度模块")

    # 测试列表
    tests = [
        ("模型初始化", test_model_initialization),
        ("前向传播", test_forward_pass),
        ("密度预测模块", test_density_module),
        ("内存使用", test_memory_usage),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            if test_name == "模型初始化":
                model = test_func()
            else:
                result = test_func()
                results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name}测试失败: {e}")
            results.append((test_name, False))

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 所有测试通过！模型改进成功。")
        print("\n下一步:")
        print("1. 使用 example_train_improved_density.py 开始训练")
        print("2. 监控小目标检测性能改进")
        print("3. 对比原始模型的训练时间和准确率")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)