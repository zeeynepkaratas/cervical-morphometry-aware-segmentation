"""Inference restoration and metric helpers for pilot evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from src.measurements.morphometry import compute_circularity, compute_nc_ratio, relative_error
from src.segmentation.train_unet import INPUT_SIZE


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice score for boolean masks."""
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute IoU score for boolean masks."""
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def foreground_dice(pred_target: np.ndarray, gt_target: np.ndarray) -> float:
    """Mean Dice over cytoplasm-only and nucleus classes."""
    return float(np.mean([dice_score(pred_target == cls, gt_target == cls) for cls in (1, 2)]))


def restore_prediction_to_original_size(
    pred_target: np.ndarray,
    original_shape: tuple[int, int],
    input_size: tuple[int, int] = INPUT_SIZE,
) -> np.ndarray:
    """Undo aspect-ratio-preserving resize and center padding for class maps."""
    original_height, original_width = original_shape
    target_width, target_height = input_size
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    cropped = pred_target[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width]
    restored = Image.fromarray(cropped.astype(np.uint8), mode="L").resize(
        (original_width, original_height),
        Image.Resampling.NEAREST,
    )
    return np.asarray(restored, dtype=np.uint8)


def masks_to_target(nucleus_mask: np.ndarray, cytoplasm_mask: np.ndarray) -> np.ndarray:
    """Build an original-resolution 0/1/2 target map."""
    target = np.zeros(nucleus_mask.shape, dtype=np.uint8)
    target[np.logical_and(cytoplasm_mask, ~nucleus_mask)] = 1
    target[nucleus_mask] = 2
    return target


def safe_morphometry(pred_nucleus: np.ndarray, pred_cytoplasm: np.ndarray, gt_nucleus: np.ndarray, gt_cytoplasm: np.ndarray) -> dict:
    """Compute official morphometry metrics with invalid prediction flags."""
    gt_nc = compute_nc_ratio(gt_nucleus, gt_cytoplasm)
    gt_circularity = compute_circularity(gt_nucleus)
    row = {"gt_nc_ratio": gt_nc, "gt_circularity": gt_circularity}
    try:
        pred_nc = compute_nc_ratio(pred_nucleus, pred_cytoplasm)
        row.update(
            pred_nc_ratio=pred_nc,
            nc_absolute_error=abs(pred_nc - gt_nc),
            nc_relative_error=relative_error(pred_nc, gt_nc),
            invalid_nc_prediction=False,
        )
    except ValueError:
        row.update(
            pred_nc_ratio=float("nan"),
            nc_absolute_error=float("inf"),
            nc_relative_error=float("inf"),
            invalid_nc_prediction=True,
        )
    try:
        pred_circularity = compute_circularity(pred_nucleus)
        row.update(
            pred_circularity=pred_circularity,
            circularity_absolute_error=abs(pred_circularity - gt_circularity),
            invalid_circularity_prediction=False,
        )
    except ValueError:
        row.update(
            pred_circularity=float("nan"),
            circularity_absolute_error=float("inf"),
            invalid_circularity_prediction=True,
        )
    return row


def finite_mean(values: Iterable[float]) -> float:
    """Mean over finite values, or NaN if none exist."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def finite_median(values: Iterable[float]) -> float:
    """Median over finite values, or NaN if none exist."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if len(arr) else float("nan")


def read_checkpoint(path: Path, device: str):
    """Load a checkpoint with a clear missing-file error."""
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=torch.device(device))
