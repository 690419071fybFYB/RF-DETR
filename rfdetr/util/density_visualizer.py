# ------------------------------------------------------------------------
# RF-DETR Density Map Visualizer
# ------------------------------------------------------------------------
"""
Utilities for visualizing density maps during training with bounding box overlays.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json


class DensityVisualizer:
    """
    Visualizes density maps with bounding box overlays for training monitoring.
    """

    def __init__(self,
                 output_dir: str = "density_visualizations",
                 max_samples: int = 4,
                 colormap: str = 'hot',
                 overlay_alpha: float = 0.6):
        """
        Args:
            output_dir: Directory to save visualizations
            max_samples: Maximum number of samples to visualize per batch
            colormap: Matplotlib colormap for density map
            overlay_alpha: Alpha blending for density overlay
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.colormap = colormap
        self.overlay_alpha = overlay_alpha

    def extract_density_maps(self, model_outputs: Dict) -> Dict[str, torch.Tensor]:
        """
        Extract density maps from model outputs.

        Args:
            model_outputs: Dictionary containing model outputs

        Returns:
            Dictionary of density maps
        """
        density_maps = {}

        # Extract predicted density map (if available)
        if 'pred_density' in model_outputs:
            density_maps['predicted'] = model_outputs['pred_density']

        # Extract auxiliary density maps
        if 'aux_outputs' in model_outputs:
            for i, aux in enumerate(model_outputs['aux_outputs']):
                if 'pred_density' in aux:
                    density_maps[f'aux_{i}'] = aux['pred_density']

        return density_maps

    def denormalize_boxes(self, boxes: torch.Tensor, img_h: int, img_w: int) -> np.ndarray:
        """
        Convert normalized boxes to pixel coordinates.

        Args:
            boxes: Normalized boxes [N, 4] in (cx, cy, w, h) format
            img_h, img_w: Image dimensions

        Returns:
            Boxes in pixel coordinates [N, 4] in (x1, y1, x2, y2) format
        """
        boxes = boxes.cpu().numpy()

        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
        x1 = (boxes[:, 0] - boxes[:, 2] / 2) * img_w
        y1 = (boxes[:, 1] - boxes[:, 3] / 2) * img_h
        x2 = (boxes[:, 0] + boxes[:, 2] / 2) * img_w
        y2 = (boxes[:, 1] + boxes[:, 3] / 2) * img_h

        return np.stack([x1, y1, x2, y2], axis=1).astype(int)

    def visualize_density_map(self,
                            image: torch.Tensor,
                            density_map: torch.Tensor,
                            boxes: Optional[torch.Tensor] = None,
                            labels: Optional[List[int]] = None,
                            class_names: Optional[Dict[int, str]] = None,
                            save_path: Optional[str] = None,
                            title: str = "Density Map Visualization") -> None:
        """
        Visualize a single density map with bounding box overlay.

        Args:
            image: Input image [C, H, W]
            density_map: Density map [1, H_d, W_d] or [H_d, W_d]
            boxes: Ground truth boxes [N, 4] in normalized (cx, cy, w, h) format
            labels: Class labels for each box [N]
            class_names: Mapping from class ID to name
            save_path: Path to save the visualization
            title: Title for the plot
        """
        # Convert tensors to numpy
        if image.is_floating_point():
            img_np = image.cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = image.cpu().numpy().transpose(1, 2, 0) / 255.0

        # Handle density map format
        if density_map.dim() == 4:
            density_np = density_map.squeeze(0).squeeze(0).cpu().numpy()
        else:
            density_np = density_map.squeeze(0).cpu().numpy()

        # Get image dimensions
        img_h, img_w = img_np.shape[:2]

        # Resize density map to match image size
        if density_np.shape != (img_h, img_w):
            density_tensor = torch.from_numpy(density_np).unsqueeze(0).unsqueeze(0).float()
            density_tensor = F.interpolate(density_tensor, size=(img_h, img_w), mode='bilinear', align_corners=False)
            density_np = density_tensor.squeeze(0).squeeze(0).numpy()

        # Normalize density map for visualization
        if density_np.max() > 0:
            density_np = density_np / density_np.max()

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)

        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Density map
        im = axes[1].imshow(density_np, cmap=self.colormap, vmin=0, vmax=1)
        axes[1].set_title('Density Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(img_np)
        axes[2].imshow(density_np, cmap=self.colormap, alpha=self.overlay_alpha, vmin=0, vmax=1)
        axes[2].set_title('Density + BBoxes')
        axes[2].axis('off')

        # Draw bounding boxes if provided
        if boxes is not None and len(boxes) > 0:
            boxes_pixel = self.denormalize_boxes(boxes, img_h, img_w)

            for i, (box, label) in enumerate(zip(boxes_pixel, labels or [])):
                x1, y1, x2, y2 = box

                # Draw on original image
                rect_orig = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                            linewidth=2, edgecolor='green', facecolor='none')
                axes[0].add_patch(rect_orig)

                # Draw on overlay
                rect_overlay = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                               linewidth=2, edgecolor='lime', facecolor='none')
                axes[2].add_patch(rect_overlay)

                # Add label if available
                if class_names and label in class_names:
                    axes[2].text(x1, y1-5, class_names[label],
                               color='white', bbox=dict(facecolor='green', alpha=0.5))

        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.close()

    def visualize_batch(self,
                       images: torch.Tensor,
                       targets: List[Dict],
                       model_outputs: Dict,
                       epoch: int,
                       step: int,
                       class_names: Optional[Dict[int, str]] = None) -> None:
        """
        Visualize a batch of images with their density maps.

        Args:
            images: Batch of images [B, C, H, W]
            targets: List of target dictionaries
            model_outputs: Model outputs containing density maps
            epoch: Current epoch
            step: Current step
            class_names: Mapping from class ID to name
        """
        # Extract density maps
        density_maps = self.extract_density_maps(model_outputs)

        if not density_maps:
            print("No density maps found in model outputs")
            return

        # Create epoch directory
        epoch_dir = self.output_dir / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(exist_ok=True)

        # Limit number of samples
        batch_size = min(images.shape[0], self.max_samples)

        # Visualize main predicted density map
        if 'predicted' in density_maps:
            for idx in range(batch_size):
                image = images[idx]
                density_map = density_maps['predicted'][idx:idx+1]

                # Get ground truth boxes
                boxes = targets[idx].get('boxes', None)
                labels = targets[idx].get('labels', None)

                # Create save path
                save_path = epoch_dir / f"step_{step:06d}_sample_{idx:02d}.png"

                # Create title
                num_boxes = len(boxes) if boxes is not None else 0
                title = f"Epoch {epoch}, Step {step}, Sample {idx} (GT boxes: {num_boxes})"

                # Visualize
                self.visualize_density_map(
                    image=image,
                    density_map=density_map,
                    boxes=boxes,
                    labels=labels,
                    class_names=class_names,
                    save_path=str(save_path),
                    title=title
                )

    def save_density_statistics(self,
                               density_maps: Dict[str, torch.Tensor],
                               epoch: int,
                               step: int) -> None:
        """
        Save density map statistics for analysis.

        Args:
            density_maps: Dictionary of density maps
            epoch: Current epoch
            step: Current step
        """
        stats = {
            'epoch': epoch,
            'step': step,
            'statistics': {}
        }

        for name, density in density_maps.items():
            density_np = density.cpu().numpy()

            stats['statistics'][name] = {
                'mean': float(density_np.mean()),
                'std': float(density_np.std()),
                'min': float(density_np.min()),
                'max': float(density_np.max()),
                'shape': list(density_np.shape)
            }

        # Save to JSON
        stats_path = self.output_dir / f"epoch_{epoch:04d}" / "density_stats.json"
        with open(stats_path, 'a') as f:
            f.write(json.dumps(stats) + '\n')


def visualize_density_maps_hook(model, inputs, outputs, epoch, step, visualizer, class_names=None):
    """
    Hook function to visualize density maps during training.

    Args:
        model: The model
        inputs: Model inputs (samples, targets)
        outputs: Model outputs
        epoch: Current epoch
        step: Current step
        visualizer: DensityVisualizer instance
        class_names: Optional class name mapping
    """
    samples, targets = inputs

    # Only visualize every N steps to avoid too many outputs
    if step % 50 == 0:  # Visualize every 50 steps
        visualizer.visualize_batch(
            images=samples.tensors,
            targets=targets,
            model_outputs=outputs,
            epoch=epoch,
            step=step,
            class_names=class_names
        )

        # Extract and save statistics
        density_maps = visualizer.extract_density_maps(outputs)
        if density_maps:
            visualizer.save_density_statistics(density_maps, epoch, step)