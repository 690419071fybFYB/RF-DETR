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
    log_path = "/home/fyb/mydir/rf-detr/script/DIOR_RF_CSD_SOQB_FPN/log.txt"
    out_dir = os.path.dirname(log_path)
    os.makedirs(out_dir, exist_ok=True)

    logs = load_logs(log_path)
    if not logs:
        print("No log data found.")
        return

    epochs = [r.get("epoch", idx) for idx, r in enumerate(logs)]

    def extract_list(key, idx=None):
        vals = []
        for r in logs:
            arr = r.get(key)
            if isinstance(arr, list) and len(arr) > (idx if idx is not None else -1):
                vals.append(arr[idx] if idx is not None else arr)
            elif idx is None and key in r:
                vals.append(r[key])
        return vals

    # Prepare data
    train_loss = [r.get("train_loss") for r in logs if "train_loss" in r]
    val_loss = [r.get("test_loss") for r in logs if "test_loss" in r]
    ap50_base = extract_list("test_coco_eval_bbox", 1)
    ap50_95_base = extract_list("test_coco_eval_bbox", 0)
    ar50_95_base = extract_list("test_coco_eval_bbox", 8)
    ap50_ema = extract_list("ema_test_coco_eval_bbox", 1)
    ap50_95_ema = extract_list("ema_test_coco_eval_bbox", 0)
    ar50_95_ema = extract_list("ema_test_coco_eval_bbox", 8)

    # Plot 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle("RF-DETR Training Metrics", fontsize=16, fontweight="bold")

    # Loss
    ax = axs[0, 0]
    ax.plot(epochs[: len(train_loss)], train_loss, "-o", label="Training Loss", markersize=3)
    if val_loss:
        ax.plot(epochs[: len(val_loss)], val_loss, "-o", label="Validation Loss", markersize=3)
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("Loss Value")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # AP@0.50
    ax = axs[0, 1]
    if ap50_base:
        ax.plot(epochs[: len(ap50_base)], ap50_base, "-o", label="Base Model", markersize=3)
    if ap50_ema:
        ax.plot(epochs[: len(ap50_ema)], ap50_ema, "-o", label="EMA Model", markersize=3)
    ax.set_title("Average Precision @0.50")
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("AP@50")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # AP@0.50:0.95
    ax = axs[1, 0]
    if ap50_95_base:
        ax.plot(epochs[: len(ap50_95_base)], ap50_95_base, "-o", label="Base Model", markersize=3)
    if ap50_95_ema:
        ax.plot(epochs[: len(ap50_95_ema)], ap50_95_ema, "-o", label="EMA Model", markersize=3)
    ax.set_title("Average Precision @0.50:0.95")
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("AP")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    # AR@0.50:0.95
    ax = axs[1, 1]
    if ar50_95_base:
        ax.plot(epochs[: len(ar50_95_base)], ar50_95_base, "-o", label="Base Model", markersize=3)
    if ar50_95_ema:
        ax.plot(epochs[: len(ar50_95_ema)], ar50_95_ema, "-o", label="EMA Model", markersize=3)
    ax.set_title("Average Recall @0.50:0.95")
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("AR")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    grid_plot_path = os.path.join(out_dir, "training_metrics_grid.png")
    plt.savefig(grid_plot_path, dpi=200)
    plt.close()
    print(f"Saved training metrics grid to {grid_plot_path}")


if __name__ == "__main__":
    main()
