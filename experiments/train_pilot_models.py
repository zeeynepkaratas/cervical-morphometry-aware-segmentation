"""Train baseline and N/C-aware pilot models."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.losses.morphometry_loss import combined_segmentation_nc_loss
from src.data_prep.load_herlev import list_herlev_images, load_image_and_masks
from src.pilot.config_io import load_config, repo_commit_sha, resolve_herlev_data_dir
from src.pilot.metrics import (
    finite_mean,
    finite_median,
    foreground_dice,
    masks_to_target,
    restore_prediction_to_original_size,
    safe_morphometry,
)
from src.segmentation.train_unet import HerlevSegmentationDataset, preprocess_rgb_image
from src.segmentation.unet_model import UNet


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_name(lambda_nc: float) -> str:
    return f"baseline_lambda_0" if float(lambda_nc) == 0 else f"nc_lambda_{lambda_nc:g}".replace(".", "p")


def _load_resume_checkpoint(
    model: UNet,
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
    name: str,
    primary_rule: str,
    device: torch.device,
) -> tuple[int, float, float, list[dict]]:
    """Resume from last checkpoint, or from a best checkpoint if last is absent."""
    last_path = out_dir / f"{name}_last.pt"
    fallback_path = out_dir / f"{name}_{primary_rule}.pt"
    history_path = out_dir.parent / f"training_history_{name}.json"
    history: list[dict] = []
    if history_path.exists():
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        history = payload.get("history", [])

    path = last_path if last_path.exists() else fallback_path
    if not path.exists():
        return 1, -1.0, float("inf"), history

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_dice = float(checkpoint.get("best_val_dice", checkpoint.get("selection_value", -1.0)))
    best_composite = float(checkpoint.get("best_composite", float("inf")))
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    print(f"resuming {name} from {path.name} at epoch {start_epoch}")
    return start_epoch, best_dice, best_composite, history


def _evaluate_clean_validation(
    model: UNet,
    val_ids: list[str],
    data_dir: Path,
    image_size: tuple[int, int],
    device: torch.device,
) -> dict:
    """Evaluate clean validation images after restoring predictions to original size."""
    image_by_id = {path.stem: path for path in list_herlev_images(data_dir)}
    dice_values = []
    nc_errors = []
    invalid_nc = []
    model.eval()
    with torch.no_grad():
        for cell_id in val_ids:
            image_path = image_by_id[cell_id]
            image, gt_nucleus, gt_cytoplasm = load_image_and_masks(image_path)
            logits = model(preprocess_rgb_image(image, size=image_size).unsqueeze(0).to(device))
            pred_resized = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_target = restore_prediction_to_original_size(pred_resized, image.shape[:2], image_size)
            gt_target = masks_to_target(gt_nucleus, gt_cytoplasm)
            morphometry = safe_morphometry(pred_target == 2, pred_target == 1, gt_nucleus, gt_cytoplasm)
            dice_values.append(foreground_dice(pred_target, gt_target))
            nc_errors.append(morphometry["nc_absolute_error"])
            invalid_nc.append(bool(morphometry["invalid_nc_prediction"]))
    return {
        "val_foreground_dice": finite_mean(dice_values),
        "val_mean_nc_absolute_error": finite_mean(nc_errors),
        "val_median_nc_absolute_error": finite_median(nc_errors),
        "val_invalid_nc_rate": float(np.mean(invalid_nc)) if invalid_nc else float("nan"),
    }


def train_one(config: dict, data_dir: Path, lambda_nc: float, seed: int) -> dict:
    """Train one model and save best Dice and composite checkpoints."""
    _seed_everything(seed)
    device = torch.device(config["training"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    split = json.loads((ROOT_DIR / config["splits"]["pilot_split"]).read_text(encoding="utf-8"))
    train_ids = split["train_ids"]
    val_ids = split["validation_ids"]
    train_ds = HerlevSegmentationDataset(raw_dir=data_dir, image_ids=train_ids, image_size=tuple(config["image_size"]))
    val_ds = HerlevSegmentationDataset(raw_dir=data_dir, image_ids=val_ids, image_size=tuple(config["image_size"]))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Empty dataset: train={len(train_ds)} validation={len(val_ds)}")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=int(config["training"]["batch_size"]), shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0)
    model = UNet(base_channels=int(config["training"]["base_channels"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    out_dir = ROOT_DIR / config["outputs"]["pilot_dir"] / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{_model_name(lambda_nc)}_seed_{seed}"
    start_epoch, best_dice, best_composite, history = _load_resume_checkpoint(
        model,
        optimizer,
        out_dir,
        name,
        str(config["checkpoint"]["primary_rule"]),
        device,
    )
    patience_left = int(config["training"]["early_stopping_patience"])
    start = time.time()
    first_epoch_diag = None

    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        epoch_start = time.time()
        model.train()
        totals = {"loss": 0.0, "seg": 0.0, "nc": 0.0}
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            parts = combined_segmentation_nc_loss(model(images), targets, lambda_nc=float(lambda_nc))
            parts.total_loss.backward()
            optimizer.step()
            totals["loss"] += float(parts.total_loss.item())
            totals["seg"] += float(parts.segmentation_loss.item())
            totals["nc"] += float(parts.nc_loss.item())
        n_batches = max(1, len(train_loader))
        train_seg = totals["seg"] / n_batches
        train_nc = totals["nc"] / n_batches
        if epoch == 1:
            first_epoch_diag = {
                "model": name,
                "lambda_nc": lambda_nc,
                "seed": seed,
                "mean_segmentation_loss_epoch1": train_seg,
                "mean_nc_loss_epoch1": train_nc,
                "lambda_times_nc_epoch1": float(lambda_nc) * train_nc,
                "nc_component_fraction_epoch1": (float(lambda_nc) * train_nc) / max(1e-12, train_seg + float(lambda_nc) * train_nc),
            }

        clean_metrics = _evaluate_clean_validation(model, val_ids, data_dir, tuple(config["image_size"]), device)
        epoch_seconds = time.time() - epoch_start
        remaining_epochs = max(0, int(config["training"]["epochs"]) - epoch)
        row = {
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "estimated_remaining_seconds": remaining_epochs * epoch_seconds,
            "train_loss": totals["loss"] / n_batches,
            "train_segmentation_loss": train_seg,
            "train_nc_loss": train_nc,
            "val_dice": clean_metrics["val_foreground_dice"],
            "val_mean_nc_absolute_error": clean_metrics["val_mean_nc_absolute_error"],
            "val_median_nc_absolute_error": clean_metrics["val_median_nc_absolute_error"],
            "val_invalid_nc_rate": clean_metrics["val_invalid_nc_rate"],
            "lambda_nc": lambda_nc,
            "seed": seed,
            "device": str(device),
        }
        row["composite_validation"] = (1.0 - row["val_dice"]) + float(config["checkpoint"]["composite_nc_weight"]) * row["val_mean_nc_absolute_error"]
        history.append(row)
        print(
            f"{name} epoch {epoch:03d}/{int(config['training']['epochs']):03d} "
            f"epoch_seconds={epoch_seconds:.1f} eta_seconds={row['estimated_remaining_seconds']:.1f} "
            f"train_total={row['train_loss']:.4f} seg={train_seg:.4f} nc={train_nc:.4f} "
            f"val_dice={row['val_dice']:.4f} "
            f"val_mean_nc_abs={row['val_mean_nc_absolute_error']:.4f} "
            f"val_median_nc_abs={row['val_median_nc_absolute_error']:.4f}"
        )

        improved = False
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "lambda_nc": lambda_nc,
            "seed": seed,
            "config": {"base_channels": int(config["training"]["base_channels"]), "image_size": config["image_size"]},
            "commit_sha": repo_commit_sha(),
            "best_val_dice": best_dice,
            "best_composite": best_composite,
        }
        if row["val_dice"] > best_dice:
            best_dice = row["val_dice"]
            torch.save(payload | {"selection_metric": "best_val_dice", "selection_value": best_dice}, out_dir / f"{name}_best_val_dice.pt")
            improved = True
        if row["composite_validation"] < best_composite:
            best_composite = row["composite_validation"]
            torch.save(payload | {"selection_metric": "best_composite_validation", "selection_value": best_composite}, out_dir / f"{name}_best_composite_validation.pt")
            improved = True
        torch.save(payload | {"best_val_dice": best_dice, "best_composite": best_composite}, out_dir / f"{name}_last.pt")
        hist_path = ROOT_DIR / config["outputs"]["pilot_dir"] / f"training_history_{name}.json"
        hist_path.write_text(json.dumps({"history": history, "elapsed_seconds": time.time() - start}, indent=2), encoding="utf-8")
        patience_left = int(config["training"]["early_stopping_patience"]) if improved else patience_left - 1
        if patience_left <= 0:
            break

    hist_path = ROOT_DIR / config["outputs"]["pilot_dir"] / f"training_history_{name}.json"
    hist_path.write_text(json.dumps({"history": history, "elapsed_seconds": time.time() - start}, indent=2), encoding="utf-8")
    return {"name": name, "history_path": str(hist_path), "diagnostic": first_epoch_diag, "elapsed_seconds": time.time() - start}


def _summarize_quick_prescan(config: dict, runs: list[dict]) -> dict:
    """Write a quick clean-validation comparison table and preliminary decision."""
    rows = []
    for run in runs:
        history_payload = json.loads(Path(run["history_path"]).read_text(encoding="utf-8"))
        metric_rows = [
            row
            for row in history_payload.get("history", [])
            if "val_mean_nc_absolute_error" in row and np.isfinite(float(row["val_mean_nc_absolute_error"]))
        ]
        if not metric_rows:
            continue
        best = min(metric_rows, key=lambda row: float(row["composite_validation"]))
        rows.append(
            {
                "Model": run["name"],
                "En iyi epoch": int(best["epoch"]),
                "Foreground Dice": float(best["val_dice"]),
                "Mean N/C absolute error": float(best["val_mean_nc_absolute_error"]),
                "Median N/C absolute error": float(best["val_median_nc_absolute_error"]),
                "Invalid N/C rate": float(best["val_invalid_nc_rate"]),
                "Completed logged epochs": len(metric_rows),
                "Mean epoch seconds": float(np.mean([row["epoch_seconds"] for row in metric_rows])),
                "Last ETA seconds": float(metric_rows[-1]["estimated_remaining_seconds"]),
            }
        )

    baseline = next((row for row in rows if row["Model"].startswith("baseline_lambda_0")), None)
    candidate = next((row for row in rows if row["Model"].startswith("nc_lambda_0p1")), None)
    if baseline is None or candidate is None:
        decision = "UNCLEAR"
        reason = "Both model summaries are not available."
    else:
        nc_change = candidate["Mean N/C absolute error"] - baseline["Mean N/C absolute error"]
        dice_drop = baseline["Foreground Dice"] - candidate["Foreground Dice"]
        relative_nc_change = nc_change / baseline["Mean N/C absolute error"] if baseline["Mean N/C absolute error"] else float("nan")
        if relative_nc_change <= -0.10 and dice_drop < 0.01:
            decision = "PROMISING"
        elif relative_nc_change >= 0.0 or dice_drop >= 0.01:
            decision = "NOT_PROMISING"
        else:
            decision = "UNCLEAR"
        reason = f"relative_nc_change={relative_nc_change:.4f}, dice_drop={dice_drop:.4f}"

    out_dir = ROOT_DIR / config["outputs"]["pilot_dir"]
    csv_path = out_dir / "quick_prescan_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"rows": rows, "preliminary_decision": decision, "reason": reason}
    (out_dir / "quick_prescan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--quick-prescan", action="store_true")
    args = parser.parse_args()
    config = load_config(ROOT_DIR / args.config)
    if args.quick_prescan:
        config["loss"]["lambda_nc_values"] = [0.0, 0.10]
        config["training"]["epochs"] = 15
        config["training"]["early_stopping_patience"] = 4
    data_dir = resolve_herlev_data_dir(config, args.data_dir)
    split_path = ROOT_DIR / config["splits"]["pilot_split"]
    if not split_path.exists():
        raise FileNotFoundError(f"Pilot split not found. Run experiments/make_pilot_split.py first: {split_path}")
    out_dir = ROOT_DIR / config["outputs"]["pilot_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    runs = []
    for seed in config["training"]["seeds_initial"]:
        for lambda_nc in config["loss"]["lambda_nc_values"]:
            result = train_one(config, data_dir, float(lambda_nc), int(seed))
            runs.append(result)
            if result["diagnostic"] is not None:
                diagnostics.append(result["diagnostic"])
    if diagnostics:
        with (out_dir / "loss_scale_diagnostic.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0].keys()))
            writer.writeheader()
            writer.writerows(diagnostics)
    output = {"completed_runs": runs}
    if args.quick_prescan:
        output["quick_prescan_summary"] = _summarize_quick_prescan(config, runs)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
