"""Phase 1 boundary-aware fine-tuning pilot.

This script intentionally uses only the clean pilot train/validation split.
It does not access test, corruptions, calibration, conformal, or extra seeds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_prep.load_herlev import list_herlev_images, load_image_and_masks
from src.measurements.morphometry import compute_circularity, compute_nc_ratio
from src.pilot.config_io import load_config, resolve_herlev_data_dir
from src.pilot.fair_repeat import seed_everything, write_csv
from src.pilot.metrics import dice_score, foreground_dice, masks_to_target, restore_prediction_to_original_size
from src.segmentation.train_unet import HerlevSegmentationDataset, preprocess_rgb_image, segmentation_loss
from src.segmentation.unet_model import UNet


SEED = 20260803
MAX_EPOCHS = 10
BASELINE_CHECKPOINT = ROOT_DIR / "results" / "pilot" / "fair_repeat" / "checkpoints" / "baseline_seed_20260803_best_val_dice.pt"
OUT_DIR = ROOT_DIR / "results" / "boundary_pilot"
FINE_TUNE_LR = 1e-4
FIRST_EPOCH_STOP_SECONDS = 180.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def boundary_band_from_targets(targets: torch.Tensor) -> torch.Tensor:
    """Return a deterministic narrow GT boundary band for nucleus and foreground."""
    if targets.ndim != 3:
        raise ValueError(f"Expected N x H x W targets, got {tuple(targets.shape)}")
    bands = []
    for class_mask in ((targets == 2).float(), (targets > 0).float()):
        expanded = class_mask.unsqueeze(1)
        dilated = F.max_pool2d(expanded, kernel_size=3, stride=1, padding=1)
        eroded = 1.0 - F.max_pool2d(1.0 - expanded, kernel_size=3, stride=1, padding=1)
        bands.append((dilated - eroded).squeeze(1) > 0)
    return torch.logical_or(bands[0], bands[1])


def boundary_weighted_term(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy restricted to the GT-derived boundary band."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    band = boundary_band_from_targets(targets)
    if not bool(band.any()):
        return ce.mean() * 0.0
    return ce[band].mean()


def safe_perimeter(mask: np.ndarray) -> float:
    mask_u8 = np.asarray(mask).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float("nan")
    contour = max(contours, key=cv2.contourArea)
    return float(cv2.arcLength(contour, True))


def load_split(config: dict) -> dict:
    return json.loads((ROOT_DIR / config["splits"]["pilot_split"]).read_text(encoding="utf-8"))


def load_model_from_checkpoint(config: dict, device: torch.device) -> tuple[UNet, dict]:
    checkpoint = torch.load(BASELINE_CHECKPOINT, map_location=device)
    model = UNet(base_channels=int(config["training"]["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def lock_lambda(config: dict, data_dir: Path, device: torch.device) -> tuple[float, dict]:
    split = load_split(config)
    image_size = tuple(config["image_size"])
    batch_size = int(config["training"]["batch_size"])
    train_ds = HerlevSegmentationDataset(raw_dir=data_dir, image_ids=split["train_ids"], image_size=image_size)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED + 1),
    )
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = targets.to(device)
    model, checkpoint = load_model_from_checkpoint(config, device)
    model.eval()
    with torch.no_grad():
        logits = model(images)
        seg = segmentation_loss(logits, targets)
        boundary = boundary_weighted_term(logits, targets)
    seg_value = float(seg.item())
    boundary_value = float(boundary.item())
    if boundary_value <= 0 or not math.isfinite(boundary_value):
        raise ValueError(f"Invalid initial boundary loss: {boundary_value}")
    lambda_boundary = 0.15 * seg_value / boundary_value
    payload = {
        "seed": SEED,
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "baseline_checkpoint_sha256": sha256_file(BASELINE_CHECKPOINT),
        "baseline_best_epoch": int(checkpoint["epoch"]),
        "baseline_validation_dice": float(checkpoint["val_dice"] if "val_dice" in checkpoint else checkpoint.get("best_val_dice", float("nan"))),
        "first_train_batch_generator_seed": SEED + 1,
        "batch_size": batch_size,
        "L_seg_initial": seg_value,
        "L_boundary_initial": boundary_value,
        "target_fraction_of_segmentation_loss": 0.15,
        "lambda_boundary": lambda_boundary,
        "lambda_boundary_times_L_boundary_initial": lambda_boundary * boundary_value,
        "fine_tune_learning_rate": FINE_TUNE_LR,
        "fine_tune_learning_rate_reason": "Pre-specified before validation because this is lightweight fine-tuning from an already trained checkpoint.",
        "boundary_definition": "3x3 morphological dilation minus erosion on GT nucleus mask and GT foreground mask; union of both bands.",
        "validation_used_for_lambda": False,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lambda_lock.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return lambda_boundary, payload


def train_boundary(config: dict, data_dir: Path, device: torch.device, lambda_boundary: float, lambda_payload: dict) -> tuple[Path, list[dict], float, bool]:
    split = load_split(config)
    image_size = tuple(config["image_size"])
    batch_size = int(config["training"]["batch_size"])
    train_ds = HerlevSegmentationDataset(raw_dir=data_dir, image_ids=split["train_ids"], image_size=image_size)
    model, baseline_checkpoint = load_model_from_checkpoint(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
    history: list[dict] = []
    run_start = time.time()
    stopped_early_for_runtime = False
    last_path = OUT_DIR / "boundary_aware_seed_20260803_last.pt"
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        seed_everything(SEED)
        model.train()
        loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(SEED + epoch),
        )
        totals = {"total": 0.0, "seg": 0.0, "boundary": 0.0}
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            seg = segmentation_loss(logits, targets)
            boundary = boundary_weighted_term(logits, targets)
            total = seg + float(lambda_boundary) * boundary
            total.backward()
            optimizer.step()
            totals["total"] += float(total.item())
            totals["seg"] += float(seg.item())
            totals["boundary"] += float(boundary.item())
        n_batches = max(1, len(loader))
        epoch_seconds = time.time() - epoch_start
        row = {
            "seed": SEED,
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "estimated_remaining_seconds": (MAX_EPOCHS - epoch) * epoch_seconds,
            "train_total_loss": totals["total"] / n_batches,
            "train_segmentation_loss": totals["seg"] / n_batches,
            "train_boundary_loss": totals["boundary"] / n_batches,
            "lambda_boundary": float(lambda_boundary),
            "weighted_boundary_loss": float(lambda_boundary) * totals["boundary"] / n_batches,
            "learning_rate": FINE_TUNE_LR,
        }
        history.append(row)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "seed": SEED,
                "lambda_boundary": float(lambda_boundary),
                "baseline_checkpoint": str(BASELINE_CHECKPOINT),
                "baseline_checkpoint_sha256": lambda_payload["baseline_checkpoint_sha256"],
                "config": {
                    "image_size": config["image_size"],
                    "base_channels": int(config["training"]["base_channels"]),
                    "batch_size": batch_size,
                    "learning_rate": FINE_TUNE_LR,
                    "max_epochs": MAX_EPOCHS,
                    "boundary_definition": lambda_payload["boundary_definition"],
                },
            },
            last_path,
        )
        (OUT_DIR / "training_history.json").write_text(json.dumps({"history": history}, indent=2), encoding="utf-8")
        print(
            f"boundary epoch {epoch:03d}/{MAX_EPOCHS:03d} "
            f"epoch_seconds={epoch_seconds:.1f} eta_seconds={row['estimated_remaining_seconds']:.1f} "
            f"train_total={row['train_total_loss']:.4f} seg={row['train_segmentation_loss']:.4f} "
            f"boundary={row['train_boundary_loss']:.4f}",
            flush=True,
        )
        if epoch == 1 and epoch_seconds > FIRST_EPOCH_STOP_SECONDS:
            stopped_early_for_runtime = True
            break
    runtime = time.time() - run_start
    (OUT_DIR / "runtime.json").write_text(
        json.dumps({"runtime_seconds": runtime, "completed_epochs": len(history), "stopped_early_for_runtime": stopped_early_for_runtime}, indent=2),
        encoding="utf-8",
    )
    return last_path, history, runtime, stopped_early_for_runtime


def evaluate_model(config: dict, data_dir: Path, device: torch.device, checkpoint_path: Path, model_name: str) -> list[dict]:
    split = load_split(config)
    image_size = tuple(config["image_size"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet(base_channels=int(config["training"]["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image_by_id = {path.stem: path for path in list_herlev_images(data_dir)}
    rows = []
    with torch.no_grad():
        for cell_id in split["validation_ids"]:
            image_path = image_by_id[cell_id]
            image, gt_nucleus, gt_cytoplasm = load_image_and_masks(image_path)
            logits = model(preprocess_rgb_image(image, size=image_size).unsqueeze(0).to(device))
            pred_resized = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_target = restore_prediction_to_original_size(pred_resized, image.shape[:2], image_size)
            gt_target = masks_to_target(gt_nucleus, gt_cytoplasm)
            pred_nucleus = pred_target == 2
            pred_cytoplasm = pred_target == 1
            gt_perimeter = safe_perimeter(gt_nucleus)
            pred_perimeter = safe_perimeter(pred_nucleus)
            gt_nc = compute_nc_ratio(gt_nucleus, gt_cytoplasm)
            row = {
                "cell_id": cell_id,
                "class": image_path.parent.name,
                "model": model_name,
                "nucleus_dice": dice_score(pred_nucleus, gt_nucleus),
                "foreground_dice": foreground_dice(pred_target, gt_target),
                "gt_circularity": compute_circularity(gt_nucleus),
                "pred_circularity": compute_circularity(pred_nucleus) if pred_nucleus.any() else float("nan"),
                "gt_perimeter": gt_perimeter,
                "pred_perimeter": pred_perimeter,
                "gt_nc": gt_nc,
                "pred_nc": compute_nc_ratio(pred_nucleus, pred_cytoplasm) if pred_nucleus.any() and pred_cytoplasm.any() else float("nan"),
                "gt_nucleus_area": int(np.count_nonzero(gt_nucleus)),
                "pred_nucleus_area": int(np.count_nonzero(pred_nucleus)),
            }
            row["circularity_AE"] = abs(row["pred_circularity"] - row["gt_circularity"]) if math.isfinite(row["pred_circularity"]) else float("nan")
            row["perimeter_AE"] = abs(row["pred_perimeter"] - row["gt_perimeter"]) if math.isfinite(row["pred_perimeter"]) else float("nan")
            row["nc_AE"] = abs(row["pred_nc"] - row["gt_nc"]) if math.isfinite(row["pred_nc"]) else float("nan")
            row["area_AE"] = abs(row["pred_nucleus_area"] - row["gt_nucleus_area"])
            rows.append(row)
    return rows


def finite(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def summarize(rows: list[dict]) -> dict:
    out = {"n_cells": len(rows)}
    for key in ["nucleus_dice", "foreground_dice", "circularity_AE", "perimeter_AE", "nc_AE", "area_AE"]:
        vals = finite([row[key] for row in rows])
        out[f"mean_{key}"] = float(np.mean(vals)) if vals.size else float("nan")
        out[f"median_{key}"] = float(np.median(vals)) if vals.size else float("nan")
    gt_c = finite([row["gt_circularity"] for row in rows])
    pred_c = finite([row["pred_circularity"] for row in rows])
    gt_p = finite([row["gt_perimeter"] for row in rows])
    pred_p = finite([row["pred_perimeter"] for row in rows])
    out["circularity_variance_ratio"] = float(np.var(pred_c) / np.var(gt_c)) if pred_c.size and gt_c.size and np.var(gt_c) > 0 else float("nan")
    out["perimeter_variance_ratio"] = float(np.var(pred_p) / np.var(gt_p)) if pred_p.size and gt_p.size and np.var(gt_p) > 0 else float("nan")
    out["median_predicted_circularity"] = float(np.median(pred_c)) if pred_c.size else float("nan")
    out["median_gt_circularity"] = float(np.median(gt_c)) if gt_c.size else float("nan")
    out["median_predicted_perimeter"] = float(np.median(pred_p)) if pred_p.size else float("nan")
    out["median_gt_perimeter"] = float(np.median(gt_p)) if gt_p.size else float("nan")
    return out


def paired_rows(rows: list[dict]) -> list[dict]:
    by_cell: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_cell.setdefault(str(row["cell_id"]), {})[str(row["model"])] = row
    out = []
    for cell_id, models in sorted(by_cell.items()):
        if set(models) != {"baseline", "boundary_aware"}:
            continue
        base = models["baseline"]
        bound = models["boundary_aware"]
        out.append(
            {
                "cell_id": cell_id,
                "class": base["class"],
                "delta_circularity_AE": bound["circularity_AE"] - base["circularity_AE"],
                "delta_perimeter_AE": bound["perimeter_AE"] - base["perimeter_AE"],
                "delta_nc_AE": bound["nc_AE"] - base["nc_AE"],
                "delta_nucleus_dice": bound["nucleus_dice"] - base["nucleus_dice"],
                "delta_foreground_dice": bound["foreground_dice"] - base["foreground_dice"],
                "delta_area_AE": bound["area_AE"] - base["area_AE"],
            }
        )
    return out


def paired_summary(rows: list[dict]) -> dict:
    out = {"n_cells": len(rows)}
    for key in ["delta_circularity_AE", "delta_perimeter_AE", "delta_nc_AE", "delta_nucleus_dice", "delta_foreground_dice", "delta_area_AE"]:
        vals = finite([row[key] for row in rows])
        out[f"mean_{key}"] = float(np.mean(vals)) if vals.size else float("nan")
        out[f"median_{key}"] = float(np.median(vals)) if vals.size else float("nan")
    circ = finite([row["delta_circularity_AE"] for row in rows])
    perim = finite([row["delta_perimeter_AE"] for row in rows])
    out["pct_cells_lower_circularity_AE"] = float(np.mean(circ < 0) * 100.0) if circ.size else float("nan")
    out["pct_cells_lower_perimeter_AE"] = float(np.mean(perim < 0) * 100.0) if perim.size else float("nan")
    return out


def main() -> None:
    config = load_config(ROOT_DIR / "configs" / "pilot.yaml")
    data_dir = resolve_herlev_data_dir(config)
    device = torch.device(config["training"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lambda_boundary, lambda_payload = lock_lambda(config, data_dir, device)
    baseline_model, baseline_ckpt = load_model_from_checkpoint(config, device)
    lambda_payload["parameter_count"] = parameter_count(baseline_model)
    (OUT_DIR / "lambda_lock.json").write_text(json.dumps(lambda_payload, indent=2), encoding="utf-8")
    boundary_path, history, runtime, stopped = train_boundary(config, data_dir, device, lambda_boundary, lambda_payload)
    if stopped:
        print(json.dumps({"status": "STOPPED_RUNTIME_ESTIMATE", "history": history, "runtime_seconds": runtime}, indent=2))
        return
    baseline_rows = evaluate_model(config, data_dir, device, BASELINE_CHECKPOINT, "baseline")
    boundary_rows = evaluate_model(config, data_dir, device, boundary_path, "boundary_aware")
    all_rows = baseline_rows + boundary_rows
    pairs = paired_rows(all_rows)
    baseline_summary = summarize(baseline_rows)
    boundary_summary = summarize(boundary_rows)
    pair_summary = paired_summary(pairs)
    write_csv(OUT_DIR / "clean_validation_per_cell.csv", all_rows)
    write_csv(OUT_DIR / "clean_validation_paired_differences.csv", pairs)
    summary = {
        "status": "BOUNDARY_FINE_TUNE_PHASE1_COMPLETE",
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "baseline_checkpoint_sha256": lambda_payload["baseline_checkpoint_sha256"],
        "baseline_best_epoch": lambda_payload["baseline_best_epoch"],
        "baseline_validation_dice": lambda_payload["baseline_validation_dice"],
        "parameter_count": lambda_payload["parameter_count"],
        "lambda_boundary": lambda_boundary,
        "lambda_derivation": lambda_payload,
        "runtime_seconds": runtime,
        "completed_epochs": len(history),
        "baseline_clean_validation": baseline_summary,
        "boundary_aware_clean_validation": boundary_summary,
        "paired_differences": pair_summary,
        "no_corruption_used": True,
        "test_accessed": False,
        "additional_lambda_tested": False,
        "additional_training_run": False,
    }
    (OUT_DIR / "phase1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
