import argparse
import os
from typing import Dict, Any

import torch


TARGET_KEY = "logit_scaler.scale"


def strip_keys(state_dict: Dict[str, Any]) -> int:
    """
    Remove any entries whose key includes logit_scaler.scale.
    Returns the number of removed keys.
    """
    if state_dict is None:
        return 0
    to_drop = [k for k in list(state_dict.keys()) if TARGET_KEY in k]
    for k in to_drop:
        state_dict.pop(k, None)
    return len(to_drop)


def clean_checkpoint(in_path: str, out_path: str) -> None:
    checkpoint = torch.load(in_path, map_location="cpu", weights_only=False)
    removed = 0

    # main model weights
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        removed += strip_keys(checkpoint["model"])

    # EMA weights
    if "ema_model" in checkpoint and isinstance(checkpoint["ema_model"], dict):
        removed += strip_keys(checkpoint["ema_model"])

    # other possible nested state dicts
    for key in checkpoint:
        if key in {"model", "ema_model"}:
            continue
        if isinstance(checkpoint[key], dict):
            removed += strip_keys(checkpoint[key])

    torch.save(checkpoint, out_path)
    print(f"Saved cleaned checkpoint to: {out_path}")
    print(f"Dropped {removed} entries containing '{TARGET_KEY}'.")


def main():
    parser = argparse.ArgumentParser(description="Remove logit_scaler.scale keys from a checkpoint.")
    parser.add_argument(
        "--in", dest="in_path", required=True,
        help="Path to the checkpoint to clean."
    )
    parser.add_argument(
        "--out", dest="out_path", default=None,
        help="Path to write the cleaned checkpoint. Default: <input>_clean.pth"
    )
    parser.add_argument(
        "--inplace", action="store_true",
        help="Overwrite the input checkpoint."
    )
    args = parser.parse_args()

    out_path = args.out_path
    if args.inplace:
        out_path = args.in_path
    elif out_path is None:
        root, ext = os.path.splitext(args.in_path)
        out_path = f"{root}_clean{ext or '.pth'}"

    clean_checkpoint(args.in_path, out_path)


if __name__ == "__main__":
    main()
