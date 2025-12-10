#!/usr/bin/env python3
"""
使用改进的密度预测模块训练RF-DETR
- 简化模型：只保留Density Init，移除其他密度模块
- 使用UNet结构替代简单CNN
"""

import argparse
from pathlib import Path

from rfdetr.main import Model


def main():
    parser = argparse.ArgumentParser(description="RF-DETR训练 - 改进的密度预测")

    # 数据集参数
    parser.add_argument("--dataset_dir", required=True, help="数据集路径")
    parser.add_argument("--num_classes", type=int, default=80, help="类别数量")

    # 密度功能
    parser.add_argument("--enable_density_init", action="store_true", default=True,
                       help="启用密度引导查询初始化（推荐）")
    parser.add_argument("--enable_improved_density", action="store_true", default=True,
                       help="使用改进的UNet密度预测器")
    parser.add_argument("--density_adaptive_threshold", action="store_true", default=True,
                       help="启用自适应阈值采样")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--output_dir", default="outputs/improved_density_experiment",
                       help="输出目录")

    # 模型配置
    parser.add_argument("--model", choices=["nano", "small", "medium", "base", "large"],
                       default="base", help="模型大小")

    args = parser.parse_args()

    # 创建模型实例
    model = Model()

    # 根据模型选择配置
    model_configs = {
        "nano": "RFDETRNanoConfig",
        "small": "RFDETRSmallConfig",
        "medium": "RFDETRMediumConfig",
        "base": "RFDETRBaseConfig",
        "large": "RFDETRLargeConfig"
    }

    print("="*60)
    print("RF-DETR 改进的密度预测训练")
    print("="*60)
    print(f"模型: {args.model.upper()}")
    print(f"数据集: {args.dataset_dir}")
    print(f"类别数: {args.num_classes}")
    print(f"密度初始化: {'✓' if args.enable_density_init else '✗'}")
    print(f"改进的UNet预测器: {'✓' if args.enable_improved_density else '✗'}")
    print(f"自适应阈值: {'✓' if args.density_adaptive_threshold else '✗'}")
    print("-"*60)

    # 准备训练参数
    train_params = {
        "dataset_dir": args.dataset_dir,
        "num_classes": args.num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "output_dir": args.output_dir,

        # 密度功能配置
        "enable_density_init": args.enable_density_init,
        "enable_improved_density": args.enable_improved_density,
        "density_adaptive_threshold": args.density_adaptive_threshold,

        # 禁用其他密度模块（基于消融实验结果）
        "enable_density_augmented_cross_attn": False,
        "enable_density_positional_bias": False,
        "enable_density_sampling_offset": False,

        # 保持其他优化
        "enable_small_object_query_boost": True,
        "soqb_boost_factor": 2.0,

        # 训练优化
        "use_ema": True,
        "amp": True,
        "gradient_checkpointing": False,
    }

    # 开始训练
    print("\n开始训练...")
    print("预期改进:")
    print("- 小目标检测提升 +4-6% AP")
    print("- 训练更稳定")
    print("- 推理速度更快（移除了冗余模块）")
    print("="*60)

    model.train(
        model_name=model_configs[args.model],
        **train_params
    )

    print("\n训练完成！")
    print(f"检查结果: {args.output_dir}")
    print("\n性能评估建议:")
    print("1. 使用 --eval flag 评估模型")
    print("2. 特别关注小目标 (small) 的 AP")
    print("3. 对比原版模型的训练时间")


if __name__ == "__main__":
    main()