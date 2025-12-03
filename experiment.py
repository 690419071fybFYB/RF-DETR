"""
Minimal training/evaluation entrypoint for RF-DETR.

This script keeps everything in a single file while reusing the existing
model/loss/dataloader code under ``rfdetr.*``. It mirrors the lightweight
style of AI-Scientist templates: parse a few knobs, build the model, then
train or eval.

Usage (single-GPU example):
    python experiment.py --dataset-dir /path/to/coco \
        --num-classes 91 --epochs 12 --batch-size 2 --out-dir run_0

Eval-only from a checkpoint:
    python experiment.py --dataset-dir /path/to/coco --eval-only \
        --resume /path/to/checkpoint.pth
"""

import argparse
from collections import defaultdict

from rfdetr.main import Model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal RF-DETR train/eval runner (single file)."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="COCO-style dataset root (expects train/val splits inside).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=91,
        help="Number of object classes (COCO default is 91 including background).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=12, help="Total training epochs."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="run_0",
        help="Directory to save checkpoints/logs.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint to resume/eval.",
    )
    parser.add_argument(
        "--pretrain-weights",
        type=str,
        default=None,
        help="Optional pretrained weights to load before training.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="DataLoader workers."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        help="Input resolution (square, passed to existing args).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, run evaluation on the validation split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Map CLI to the kwargs expected by rfdetr.main.Model / populate_args.
    common_kwargs = dict(
        dataset_dir=args.dataset_dir,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        output_dir=args.out_dir,
        resume=args.resume,
        eval=args.eval_only,
        num_workers=args.num_workers,
        pretrain_weights=args.pretrain_weights,
        resolution=args.resolution,
    )

    model = Model(**common_kwargs)
    callbacks = defaultdict(list)  # No custom callbacks by default.
    model.train(callbacks, **common_kwargs)


if __name__ == "__main__":
    main()
