"""Evaluate pilot checkpoints on clean, Gaussian blur, and Gaussian noise."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_prep.load_herlev import list_herlev_images, load_image_and_masks
from src.degradation.apply_degradations import apply_gaussian_blur, apply_gaussian_noise
from src.pilot.config_io import load_config, resolve_herlev_data_dir
from src.pilot.metrics import (
    dice_score,
    foreground_dice,
    iou_score,
    masks_to_target,
    restore_prediction_to_original_size,
    safe_morphometry,
)
from src.segmentation.train_unet import preprocess_rgb_image
from src.segmentation.unet_model import UNet


def _stable_seed(base_seed: int, *parts: str) -> int:
    key = "|".join([str(base_seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little", signed=False)


def _conditions(config: dict, image: np.ndarray, cell_id: str) -> list[tuple[str, str, np.ndarray]]:
    rows = [("clean", "clean", image)]
    for sigma in config["degradations"]["gaussian_blur"]:
        label = f"blur_sigma_{float(sigma):g}"
        rows.append(("blur", label, apply_gaussian_blur(image, float(sigma))))
    for std in config["degradations"]["gaussian_noise"]:
        label = f"noise_std_{float(std):g}"
        rows.append(
            (
                "noise",
                label,
                apply_gaussian_noise(image, float(std), seed=_stable_seed(int(config["seed"]), cell_id, label)),
            )
        )
    return rows


def _checkpoint_paths(config: dict) -> list[Path]:
    checkpoint_rule = config["checkpoint"]["primary_rule"]
    paths = []
    for seed in config["training"]["seeds_initial"]:
        for lambda_nc in config["loss"]["lambda_nc_values"]:
            if float(lambda_nc) == 0:
                model = "baseline_lambda_0"
            else:
                model = f"nc_lambda_{float(lambda_nc):g}".replace(".", "p")
            paths.append(ROOT_DIR / config["outputs"]["pilot_dir"] / "checkpoints" / f"{model}_seed_{seed}_{checkpoint_rule}.pt")
    return paths


def evaluate_checkpoint(config: dict, data_dir: Path, checkpoint_path: Path) -> list[dict]:
    """Evaluate one checkpoint and return per-cell rows."""
    device = torch.device(config["training"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = UNet(base_channels=int(checkpoint["config"]["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    split = json.loads((ROOT_DIR / config["splits"]["pilot_split"]).read_text(encoding="utf-8"))
    image_by_id = {path.stem: path for path in list_herlev_images(data_dir)}
    model_name = checkpoint_path.name.replace(f"_{config['checkpoint']['primary_rule']}.pt", "")
    rows = []
    with torch.no_grad():
        for cell_id in split["validation_ids"]:
            image_path = image_by_id[cell_id]
            image, gt_nucleus, gt_cytoplasm = load_image_and_masks(image_path)
            gt_target = masks_to_target(gt_nucleus, gt_cytoplasm)
            for family, condition, variant in _conditions(config, image, cell_id):
                logits = model(preprocess_rgb_image(variant, size=tuple(config["image_size"])).unsqueeze(0).to(device))
                pred_resized = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                pred_target = restore_prediction_to_original_size(pred_resized, image.shape[:2], tuple(config["image_size"]))
                pred_nucleus = pred_target == 2
                pred_cytoplasm = pred_target == 1
                row = {
                    "model": model_name,
                    "lambda_nc": checkpoint["lambda_nc"],
                    "seed": checkpoint["seed"],
                    "checkpoint_rule": config["checkpoint"]["primary_rule"],
                    "cell_id": cell_id,
                    "class": image_path.parent.name,
                    "condition_family": family,
                    "condition": condition,
                    "nucleus_dice": dice_score(pred_nucleus, gt_nucleus),
                    "cytoplasm_dice": dice_score(pred_cytoplasm, gt_cytoplasm),
                    "foreground_mean_dice": foreground_dice(pred_target, gt_target),
                    "nucleus_iou": iou_score(pred_nucleus, gt_nucleus),
                    "cytoplasm_iou": iou_score(pred_cytoplasm, gt_cytoplasm),
                }
                row.update(safe_morphometry(pred_nucleus, pred_cytoplasm, gt_nucleus, gt_cytoplasm))
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    config = load_config(ROOT_DIR / args.config)
    data_dir = resolve_herlev_data_dir(config, args.data_dir)
    rows: list[dict] = []
    for checkpoint_path in _checkpoint_paths(config):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint, refusing to skip silently: {checkpoint_path}")
        rows.extend(evaluate_checkpoint(config, data_dir, checkpoint_path))
    output = ROOT_DIR / config["outputs"]["pilot_dir"] / "per_cell_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
