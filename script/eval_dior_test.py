#!/usr/bin/env python3
"""
在 DIOR 测试集上评估 RF-DETR 模型。
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import List, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader, SequentialSampler

try:
    import pycocotools.mask  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "pycocotools 未安装，请先运行 `pip install pycocotools` 再执行评估脚本。"
    ) from exc

from rfdetr.datasets import get_coco_api_from_dataset
from rfdetr.datasets.coco import (
    CocoDetection,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)
from rfdetr.engine import evaluate
from rfdetr.main import populate_args
from rfdetr.models import build_model, build_criterion_and_postprocessors
import rfdetr.util.misc as utils


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR on the DIOR test split.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/home/fyb/datasets/DIOR_cocoFormat"),
        help="DIOR 数据集的 COCO 格式路径（包含 test2017 与 annotations 目录）。",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/fyb/mydir/rf-detr/script/DIOR_results/checkpoint_best_total.pth"),
        help="用于评估的模型权重。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="评估 batch size，按显存需求调整。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker 数量。",
    )
    return parser.parse_args()


def load_checkpoint_args(checkpoint_path: Path) -> Tuple[argparse.Namespace, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args")
    if ckpt_args is None:
        ckpt_args = populate_args()
    else:
        ckpt_args = copy.deepcopy(ckpt_args)
    return ckpt_args, checkpoint["model"]


def build_test_dataset(args: argparse.Namespace, dataset_dir: Path) -> CocoDetection:
    img_folder = dataset_dir / "test2017"
    ann_file = dataset_dir / "annotations" / "instances_test2017.json"
    if not img_folder.exists():
        raise FileNotFoundError(f"未找到测试图片目录: {img_folder}")
    if not ann_file.exists():
        raise FileNotFoundError(f"未找到测试标注: {ann_file}")

    transform_kwargs = dict(
        resolution=args.resolution,
        multi_scale=args.multi_scale,
        expanded_scales=args.expanded_scales,
        skip_random_resize=not args.do_random_resize_via_padding,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
    )
    if getattr(args, "square_resize_div_64", False):
        transforms = make_coco_transforms_square_div_64("val", **transform_kwargs)
    else:
        transforms = make_coco_transforms("val", **transform_kwargs)

    return CocoDetection(str(img_folder), str(ann_file), transforms=transforms)


def format_results(results: List[dict]) -> str:
    header = f"{'Class':20s} | {'mAP@50:95':>10s} | {'mAP@50':>8s} | {'Precision':>9s} | {'Recall':>7s}"
    lines = [header, "-" * len(header)]
    for item in results:
        lines.append(
            f"{item['class'][:20]:20s} | "
            f"{item['map@50:95']:10.4f} | "
            f"{item['map@50']:8.4f} | "
            f"{item['precision']:9.4f} | "
            f"{item['recall']:7.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    cli_args = parse_cli_args()
    model_args, state_dict = load_checkpoint_args(cli_args.checkpoint)

    dataset_dir = cli_args.dataset_dir.resolve()
    model_args.coco_path = str(dataset_dir)
    model_args.dataset_dir = str(dataset_dir)
    model_args.eval = True
    model_args.batch_size = cli_args.batch_size
    model_args.num_workers = cli_args.num_workers
    model_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model_args.distributed = False

    device = torch.device(model_args.device)
    model = build_model(model_args)
    model.to(device)
    model.load_state_dict(state_dict, strict=True)

    criterion, postprocess = build_criterion_and_postprocessors(model_args)

    dataset_test = build_test_dataset(model_args, dataset_dir)
    sampler_test = SequentialSampler(dataset_test)
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=model_args.batch_size,
        sampler=sampler_test,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=model_args.num_workers,
    )
    base_ds = get_coco_api_from_dataset(dataset_test)

    print("开始在 DIOR 测试集上评估 ...")
    test_stats, _ = evaluate(
        model,
        criterion,
        postprocess,
        data_loader_test,
        base_ds,
        device,
        args=model_args,
    )

    results_json = test_stats.get("results_json")
    if not results_json:
        raise RuntimeError("未从评估结果中解析到指标，请检查数据或标注。")

    all_entry = next((item for item in results_json["class_map"] if item["class"] == "all"), None)
    if all_entry:
        print(
            f"\n整体指标：mAP@50-95={all_entry['map@50:95']:.4f}, "
            f"mAP@50={all_entry['map@50']:.4f}, "
            f"Precision={all_entry['precision']:.4f}, "
            f"Recall={all_entry['recall']:.4f}"
        )

    print("\n逐类指标：")
    print(format_results(results_json["class_map"]))


if __name__ == "__main__":
    main()
