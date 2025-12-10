#!/usr/bin/env python3
"""
使用密度图可视化功能训练RF-DETR的示例脚本
"""

import argparse
from pathlib import Path

# 使用RF-DETR的训练API
from rfdetr.main import Model


def train_with_density_visualization():
    """使用密度图可视化功能进行训练的示例"""

    # 配置参数
    config = {
        # 基本训练参数
        "dataset_dir": "path/to/your/dataset",  # 替换为您的数据集路径
        "num_classes": 80,  # 根据您的数据集调整

        # 密度相关功能
        "enable_density_init": True,  # 启用密度引导的查询初始化
        "enable_density_augmented_cross_attn": True,  # 启用密度增强交叉注意力
        "enable_density_positional_bias": True,  # 启用密度位置偏置调制
        "enable_density_sampling_offset": True,  # 启用密度采样偏置调制

        # 密度图可视化参数
        "visualize_density": True,  # 启用密度图可视化
        "density_vis_max_samples": 4,  # 每批次可视化的最大样本数
        "density_vis_interval": 50,  # 可视化间隔（每N步一次）

        # 训练参数
        "batch_size": 8,
        "epochs": 100,
        "lr": 1e-4,

        # 输出目录
        "output_dir": "outputs/density_experiment",
    }

    # 创建模型实例
    model = Model()

    # 开始训练
    print("开始训练（启用密度图可视化）...")
    print(f"密度图可视化将保存到: {config['output_dir']}/density_visualizations")

    # 训练过程中会自动:
    # 1. 每50步生成密度图可视化
    # 2. 保存密度图统计信息
    # 3. 将密度图与真值边界框一起可视化

    model.train(
        dataset_dir=config["dataset_dir"],
        num_classes=config["num_classes"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],

        # 传递所有配置参数
        **config
    )

    print("训练完成！")
    print(f"检查 {config['output_dir']}/density_visualizations 目录查看生成的密度图")


def train_from_command_line():
    """通过命令行参数训练的示例"""

    parser = argparse.ArgumentParser(description="RF-DETR训练与密度图可视化")

    # 数据集参数
    parser.add_argument("--dataset_dir", required=True, help="数据集路径")
    parser.add_argument("--num_classes", type=int, default=80, help="类别数量")

    # 密度功能开关
    parser.add_argument("--enable_density_init", action="store_true",
                       help="启用密度引导查询初始化")
    parser.add_argument("--enable_density_augmented_cross_attn", action="store_true",
                       help="启用密度增强交叉注意力")
    parser.add_argument("--enable_density_positional_bias", action="store_true",
                       help="启用密度位置偏置调制")
    parser.add_argument("--enable_density_sampling_offset", action="store_true",
                       help="启用密度采样偏置调制")

    # 可视化参数
    parser.add_argument("--visualize_density", action="store_true",
                       help="启用密度图可视化")
    parser.add_argument("--density_vis_max_samples", type=int, default=4,
                       help="每批次可视化的最大样本数")
    parser.add_argument("--density_vis_interval", type=int, default=50,
                       help="可视化间隔（步数）")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--output_dir", default="outputs/density_experiment",
                       help="输出目录")

    args = parser.parse_args()

    # 创建模型
    model = Model()

    # 准备训练参数
    train_params = {
        "dataset_dir": args.dataset_dir,
        "num_classes": args.num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "output_dir": args.output_dir,

        # 密度功能
        "enable_density_init": args.enable_density_init,
        "enable_density_augmented_cross_attn": args.enable_density_augmented_cross_attn,
        "enable_density_positional_bias": args.enable_density_positional_bias,
        "enable_density_sampling_offset": args.enable_density_sampling_offset,

        # 可视化
        "visualize_density": args.visualize_density,
        "density_vis_max_samples": args.density_vis_max_samples,
        "density_vis_interval": args.density_vis_interval,
    }

    # 开始训练
    print("开始训练...")
    if args.visualize_density:
        print(f"密度图可视化已启用，间隔: {args.density_vis_interval}步")

    model.train(**train_params)

    print("训练完成！")


if __name__ == "__main__":
    # 选择运行方式
    # 方式1: 直接运行此脚本（需要修改数据集路径）
    # train_with_density_visualization()

    # 方式2: 通过命令行参数运行
    train_from_command_line()