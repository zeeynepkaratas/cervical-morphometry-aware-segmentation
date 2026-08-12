"""Plain continued-training control for the boundary pilot.

Runs exactly one new training control from the frozen baseline checkpoint:
same seed, clean train split, epochs, optimizer, LR, batch size, and
preprocessing as the boundary-aware Phase 1 run, but with segmentation loss
only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.run_boundary_pilot_phase1 import (  # noqa: E402
    BASELINE_CHECKPOINT,
    FINE_TUNE_LR,
    MAX_EPOCHS,
    OUT_DIR,
    SEED,
    evaluate_model,
    finite,
    load_model_from_checkpoint,
    load_split,
    paired_rows,
    parameter_count,
    sha256_file,
    summarize,
)
from src.pilot.config_io import load_config, resolve_herlev_data_dir  # noqa: E402
from src.pilot.fair_repeat import seed_everything, write_csv  # noqa: E402
from src.segmentation.train_unet import HerlevSegmentationDataset, segmentation_loss  # noqa: E402


PLAIN_CHECKPOINT = OUT_DIR / "plain_finetune_seed_20260803_last.pt"
BOUNDARY_CHECKPOINT = OUT_DIR / "boundary_aware_seed_20260803_last.pt"


def train_plain(config: dict, data_dir: Path, device: torch.device) -> tuple[list[dict], float]:
    split = load_split(config)
    image_size = tuple(config["image_size"])
    batch_size = int(config["training"]["batch_size"])
    train_ds = HerlevSegmentationDataset(raw_dir=data_dir, image_ids=split["train_ids"], image_size=image_size)
    model, _ = load_model_from_checkpoint(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
    history: list[dict] = []
    run_start = time.time()
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
        total_loss = 0.0
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        epoch_seconds = time.time() - epoch_start
        row = {
            "seed": SEED,
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "estimated_remaining_seconds": (MAX_EPOCHS - epoch) * epoch_seconds,
            "train_total_loss": total_loss / max(1, len(loader)),
            "train_segmentation_loss": total_loss / max(1, len(loader)),
            "train_boundary_loss": 0.0,
            "lambda_boundary": 0.0,
            "weighted_boundary_loss": 0.0,
            "learning_rate": FINE_TUNE_LR,
        }
        history.append(row)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "seed": SEED,
                "lambda_boundary": 0.0,
                "baseline_checkpoint": str(BASELINE_CHECKPOINT),
                "baseline_checkpoint_sha256": sha256_file(BASELINE_CHECKPOINT),
                "config": {
                    "image_size": config["image_size"],
                    "base_channels": int(config["training"]["base_channels"]),
                    "batch_size": batch_size,
                    "learning_rate": FINE_TUNE_LR,
                    "max_epochs": MAX_EPOCHS,
                    "loss": "segmentation_loss_only",
                },
            },
            PLAIN_CHECKPOINT,
        )
        (OUT_DIR / "plain_control_training_history.json").write_text(json.dumps({"history": history}, indent=2), encoding="utf-8")
        print(
            f"plain control epoch {epoch:03d}/{MAX_EPOCHS:03d} "
            f"epoch_seconds={epoch_seconds:.1f} eta_seconds={row['estimated_remaining_seconds']:.1f} "
            f"train_total={row['train_total_loss']:.4f}",
            flush=True,
        )
    runtime = time.time() - run_start
    (OUT_DIR / "plain_control_runtime.json").write_text(
        json.dumps({"runtime_seconds": runtime, "completed_epochs": len(history)}, indent=2),
        encoding="utf-8",
    )
    return history, runtime


def contrast_boundary_vs_plain(boundary_rows: list[dict], plain_rows: list[dict]) -> dict:
    by_cell: dict[str, dict[str, dict]] = {}
    for row in boundary_rows:
        by_cell.setdefault(str(row["cell_id"]), {})["boundary"] = row
    for row in plain_rows:
        by_cell.setdefault(str(row["cell_id"]), {})["plain"] = row
    specs = {
        "nucleus_dice": "higher",
        "foreground_dice": "higher",
        "circularity_AE": "lower",
        "perimeter_AE": "lower",
        "nc_AE": "lower",
        "area_AE": "lower",
    }
    out = {"n_cells": 0}
    for key, direction in specs.items():
        deltas = []
        better = []
        for models in by_cell.values():
            if set(models) != {"boundary", "plain"}:
                continue
            boundary = float(models["boundary"][key])
            plain = float(models["plain"][key])
            if not np.isfinite(boundary) or not np.isfinite(plain):
                continue
            delta = boundary - plain
            deltas.append(delta)
            better.append(delta > 0 if direction == "higher" else delta < 0)
        vals = finite(deltas)
        out["n_cells"] = max(out["n_cells"], int(vals.size))
        out[f"mean_delta_{key}"] = float(np.mean(vals)) if vals.size else float("nan")
        out[f"median_delta_{key}"] = float(np.median(vals)) if vals.size else float("nan")
        out[f"pct_cells_boundary_better_{key}"] = float(np.mean(better) * 100.0) if better else float("nan")
    return out


def main() -> None:
    config = load_config(ROOT_DIR / "configs" / "pilot.yaml")
    data_dir = resolve_herlev_data_dir(config)
    device = torch.device(config["training"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not BOUNDARY_CHECKPOINT.exists():
        raise FileNotFoundError(f"Boundary-aware checkpoint missing: {BOUNDARY_CHECKPOINT}")

    _, runtime = train_plain(config, data_dir, device)
    baseline_rows = evaluate_model(config, data_dir, device, BASELINE_CHECKPOINT, "baseline")
    plain_rows = evaluate_model(config, data_dir, device, PLAIN_CHECKPOINT, "plain_finetune")
    boundary_rows = evaluate_model(config, data_dir, device, BOUNDARY_CHECKPOINT, "boundary_aware")
    all_rows = baseline_rows + plain_rows + boundary_rows
    write_csv(OUT_DIR / "plain_control_clean_validation_per_cell.csv", all_rows)
    boundary_plain_pairs = contrast_boundary_vs_plain(boundary_rows, plain_rows)
    write_csv(OUT_DIR / "plain_control_boundary_vs_plain_paired_differences.csv", paired_rows(plain_rows + boundary_rows))

    model, baseline_checkpoint = load_model_from_checkpoint(config, device)
    summary = {
        "status": "PLAIN_FINE_TUNE_CONTROL_COMPLETE",
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "baseline_checkpoint_sha256": sha256_file(BASELINE_CHECKPOINT),
        "baseline_best_epoch": int(baseline_checkpoint["epoch"]),
        "baseline_validation_dice": float(baseline_checkpoint["val_dice"] if "val_dice" in baseline_checkpoint else baseline_checkpoint.get("best_val_dice", float("nan"))),
        "parameter_count": parameter_count(model),
        "plain_control_runtime_seconds": runtime,
        "completed_epochs": MAX_EPOCHS,
        "frozen_baseline": summarize(baseline_rows),
        "plain_10_epoch_finetune": summarize(plain_rows),
        "boundary_aware_10_epoch_finetune": summarize(boundary_rows),
        "primary_contrast_boundary_minus_plain": boundary_plain_pairs,
        "no_test_access": True,
        "no_corruption": True,
        "no_additional_lambda": True,
        "no_additional_seed": True,
    }
    (OUT_DIR / "plain_control_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
