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
from src.pilot.config_io import load_config, repo_commit_sha, resolve_herlev_data_dir
from src.segmentation.train_unet import HerlevSegmentationDataset, _batch_foreground_dice_iou
from src.segmentation.unet_model import UNet


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_name(lambda_nc: float) -> str:
    return f"baseline_lambda_0" if float(lambda_nc) == 0 else f"nc_lambda_{lambda_nc:g}".replace(".", "p")


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
    best_dice = -1.0
    best_composite = float("inf")
    patience_left = int(config["training"]["early_stopping_patience"])
    history = []
    start = time.time()
    first_epoch_diag = None

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
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

        val_loss = 0.0
        val_dice = []
        val_nc = []
        model.eval()
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                parts = combined_segmentation_nc_loss(logits, targets, lambda_nc=float(lambda_nc))
                val_loss += float(parts.total_loss.item())
                val_nc.append(float(parts.nc_loss.item()))
                pred = torch.argmax(logits, dim=1)
                dice, _ = _batch_foreground_dice_iou(pred, targets)
                val_dice.append(dice)
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / n_batches,
            "train_segmentation_loss": train_seg,
            "train_nc_loss": train_nc,
            "val_loss": val_loss / max(1, len(val_loader)),
            "val_dice": float(np.mean(val_dice)),
            "val_nc_loss": float(np.mean(val_nc)),
            "lambda_nc": lambda_nc,
            "seed": seed,
            "device": str(device),
        }
        row["composite_validation"] = (1.0 - row["val_dice"]) + float(config["checkpoint"]["composite_nc_weight"]) * row["val_nc_loss"]
        history.append(row)

        improved = False
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "lambda_nc": lambda_nc,
            "seed": seed,
            "config": {"base_channels": int(config["training"]["base_channels"]), "image_size": config["image_size"]},
            "commit_sha": repo_commit_sha(),
        }
        if row["val_dice"] > best_dice:
            best_dice = row["val_dice"]
            torch.save(payload | {"selection_metric": "best_val_dice", "selection_value": best_dice}, out_dir / f"{name}_best_val_dice.pt")
            improved = True
        if row["composite_validation"] < best_composite:
            best_composite = row["composite_validation"]
            torch.save(payload | {"selection_metric": "best_composite_validation", "selection_value": best_composite}, out_dir / f"{name}_best_composite_validation.pt")
            improved = True
        patience_left = int(config["training"]["early_stopping_patience"]) if improved else patience_left - 1
        if patience_left <= 0:
            break

    hist_path = ROOT_DIR / config["outputs"]["pilot_dir"] / f"training_history_{name}.json"
    hist_path.write_text(json.dumps({"history": history, "elapsed_seconds": time.time() - start}, indent=2), encoding="utf-8")
    return {"name": name, "history_path": str(hist_path), "diagnostic": first_epoch_diag, "elapsed_seconds": time.time() - start}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    config = load_config(ROOT_DIR / args.config)
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
            diagnostics.append(result["diagnostic"])
    with (out_dir / "loss_scale_diagnostic.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostics)
    print(json.dumps({"completed_runs": runs}, indent=2))


if __name__ == "__main__":
    main()
