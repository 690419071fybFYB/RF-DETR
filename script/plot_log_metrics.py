import json
import os
from typing import List, Dict

import matplotlib.pyplot as plt


def load_logs(log_path: str) -> List[Dict]:
    records = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def plot_series(ax, epochs, values, label):
    ax.plot(epochs, values, label=label)
    ax.set_xlabel("Epoch")
    ax.grid(True, linestyle="--", alpha=0.4)


def main():
    log_path = "/home/fyb/mydir/rf-detr/script/DIOR_results/log.txt"
    out_dir = os.path.dirname(log_path)
    os.makedirs(out_dir, exist_ok=True)

    logs = load_logs(log_path)
    if not logs:
        print("No log data found.")
        return

    epochs = [r.get("epoch", idx) for idx, r in enumerate(logs)]

    # Scalar metrics to plot if present
    scalar_keys = [
        "train_loss",
        "test_loss",
        "train_loss_ce",
        "train_loss_bbox",
        "train_loss_giou",
        "test_loss_ce",
        "test_loss_bbox",
        "test_loss_giou",
    ]

    plt.figure(figsize=(12, 8))
    for key in scalar_keys:
        vals = [r.get(key) for r in logs if key in r]
        if len(vals) == len(epochs):
            plt.plot(epochs, vals, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training/Test Loss Metrics")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    loss_plot_path = os.path.join(out_dir, "loss_metrics.png")
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    print(f"Saved loss metrics plot to {loss_plot_path}")

    # AP metrics from coco_eval
    def extract_coco(key):
        vals = []
        for r in logs:
            arr = r.get(key)
            if isinstance(arr, list) and len(arr) > 0:
                vals.append(arr[0])  # AP50-95 at index 0
        return vals
    def extract_coco_index(key, idx):
        vals = []
        for r in logs:
            arr = r.get(key)
            if isinstance(arr, list) and len(arr) > idx:
                vals.append(arr[idx])
        return vals

    coco_vals = extract_coco("test_coco_eval_bbox")
    ema_coco_vals = extract_coco("ema_test_coco_eval_bbox")

    if coco_vals or ema_coco_vals:
        plt.figure(figsize=(10, 6))
        if coco_vals:
            plt.plot(epochs[: len(coco_vals)], coco_vals, label="test_coco_eval_bbox[0]")
        if ema_coco_vals:
            plt.plot(epochs[: len(ema_coco_vals)], ema_coco_vals, label="ema_test_coco_eval_bbox[0]")
        plt.xlabel("Epoch")
        plt.ylabel("mAP@50:95")
        plt.title("COCO Eval (bbox) mAP")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        coco_plot_path = os.path.join(out_dir, "coco_map.png")
        plt.tight_layout()
        plt.savefig(coco_plot_path, dpi=200)
        plt.close()
        print(f"Saved coco mAP plot to {coco_plot_path}")

    # mAP50, mAP small/medium/large
    coco_ap50 = extract_coco_index("test_coco_eval_bbox", 1)
    coco_ap_small = extract_coco_index("test_coco_eval_bbox", 3)
    coco_ap_medium = extract_coco_index("test_coco_eval_bbox", 4)
    coco_ap_large = extract_coco_index("test_coco_eval_bbox", 5)

    if coco_ap50 or coco_ap_small or coco_ap_medium or coco_ap_large:
        plt.figure(figsize=(10, 6))
        if coco_ap50:
            plt.plot(epochs[: len(coco_ap50)], coco_ap50, label="mAP50")
        if coco_ap_small:
            plt.plot(epochs[: len(coco_ap_small)], coco_ap_small, label="mAP_S")
        if coco_ap_medium:
            plt.plot(epochs[: len(coco_ap_medium)], coco_ap_medium, label="mAP_M")
        if coco_ap_large:
            plt.plot(epochs[: len(coco_ap_large)], coco_ap_large, label="mAP_L")
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.title("COCO mAP by scale")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        coco_scale_plot_path = os.path.join(out_dir, "coco_map_scales.png")
        plt.tight_layout()
        plt.savefig(coco_scale_plot_path, dpi=200)
        plt.close()
        print(f"Saved coco mAP scale plot to {coco_scale_plot_path}")


if __name__ == "__main__":
    main()
