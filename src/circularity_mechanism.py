"""Circularity mechanism analysis — Phase A (Direct perimeter).

Loads existing clean and noise prediction CSVs and enriches them with
DIRECT perimeter values (calculated explicitly via OpenCV contour length
from argmax predictions on the test set).

This completely avoids mathematically coupling the perimeter to circularity,
providing strict independent evidence.

Outputs:
    results/circularity_mechanism/real_prediction_area_perimeter.csv
    results/circularity_mechanism/real_prediction_deltas.csv
"""

from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FINAL_ANALYSIS_DIR = RESULTS_DIR / "final_analysis"
OUT_DIR = RESULTS_DIR / "circularity_mechanism"

DEGRADATION_CSV = FINAL_ANALYSIS_DIR / "degradation_per_cell_predictions.csv"
CONFIG_PATH = ROOT / "configs" / "pilot.yaml"
CHECKPOINT_DIR = RESULTS_DIR / "pilot" / "fair_repeat" / "checkpoints"
SPLIT_FILE = ROOT / "data" / "splits" / "original_herlev_group_split.json"

MODELS = {"baseline": 0.0, "nc_aware_lambda_0p10": 0.10}
FAIR_SEEDS = [20260803, 20260804, 20260805]
NOISE_LEVELS = [10.0, 20.0, 30.0]

# Import project utilities
from src.measurements.morphometry import relative_error
from src.pilot.metrics import restore_prediction_to_original_size
from src.segmentation.unet_model import UNet
from src.pilot.config_io import load_config, resolve_herlev_data_dir
from src.data_prep.load_herlev import load_image_and_masks
from src.data_prep.load_herlev import list_herlev_images
from src.degradation.apply_degradations import apply_gaussian_noise
from src.segmentation.train_unet import preprocess_rgb_image

def _stable_noise_seed(cell_id: str, severity: float) -> int:
    key = f"{cell_id}|gaussian_noise|{float(severity):g}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little", signed=False)

def _safe_perimeter(mask: np.ndarray) -> float:
    """OpenCV contour perimeter."""
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float("nan")
    contour = max(contours, key=cv2.contourArea)
    return float(cv2.arcLength(contour, True))

def _load_checkpoint(model_name: str, seed: int, config: dict, device: torch.device):
    path = CHECKPOINT_DIR / f"{model_name}_seed_{seed}_best_val_dice.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model = UNet(base_channels=int(config["training"]["base_channels"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def enrich_with_direct_perimeter(df: pd.DataFrame) -> pd.DataFrame:
    """Run deterministic re-inference over test split to get direct perimeters."""
    df = df.copy()
    config = load_config(CONFIG_PATH)
    herlev_dir = resolve_herlev_data_dir(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = tuple(config["image_size"])

    splits = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    # We only care about test set for Phase A evaluation
    test_ids = set(splits.get("test", []))
    image_index = {p.stem: p for p in list_herlev_images(herlev_dir)}

    # Add new columns
    df["direct_true_perimeter"] = float("nan")
    df["direct_predicted_perimeter"] = float("nan")

    # Cache true perimeters
    true_perims = {}

    print(f"Running direct perimeter inference on {len(test_ids)} test cells...")
    for seed in FAIR_SEEDS:
        for model_name in MODELS:
            try:
                model = _load_checkpoint(model_name, seed, config, device)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue

            for cell_id in test_ids:
                if cell_id not in image_index:
                    continue
                try:
                    image, gt_nucleus, _ = load_image_and_masks(image_index[cell_id])
                except Exception:
                    continue

                if cell_id not in true_perims:
                    true_perims[cell_id] = _safe_perimeter(gt_nucleus)

                for degradation, severity_str in [("clean", "clean")] + [("gaussian_noise", str(int(s))) for s in NOISE_LEVELS]:
                    if degradation == "clean":
                        img_input = image.copy()
                    else:
                        sev = float(severity_str)
                        ns = _stable_noise_seed(cell_id, sev)
                        img_input = apply_gaussian_noise(image, sev, seed=ns)

                    with torch.no_grad():
                        logits = model(
                            preprocess_rgb_image(img_input, size=image_size).unsqueeze(0).to(device)
                        )
                        pred_resized = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                    pred_target = restore_prediction_to_original_size(
                        pred_resized, image.shape[:2], image_size
                    )
                    pred_nucleus = pred_target == 2
                    
                    if not pred_nucleus.any():
                        pred_perim = float("nan")
                    else:
                        pred_perim = _safe_perimeter(pred_nucleus)

                    # Update dataframe
                    mask = (
                        (df["cell_id"] == cell_id) & 
                        (df["seed"] == seed) & 
                        (df["model"] == model_name) & 
                        (df["degradation"] == degradation) & 
                        (df["severity"] == severity_str)
                    )
                    df.loc[mask, "direct_true_perimeter"] = true_perims[cell_id]
                    df.loc[mask, "direct_predicted_perimeter"] = pred_perim

    # Compute errors
    df["direct_perimeter_absolute_error"] = (df["direct_predicted_perimeter"] - df["direct_true_perimeter"]).abs()
    df["direct_perimeter_relative_error"] = df.apply(
        lambda r: relative_error(r["direct_predicted_perimeter"], r["direct_true_perimeter"]) if pd.notna(r["direct_predicted_perimeter"]) else float("nan"),
        axis=1,
    )
    df["area_relative_error"] = df.apply(
        lambda r: relative_error(r["predicted_nucleus_area"], r["true_nucleus_area"]),
        axis=1,
    )
    df["circularity_relative_error"] = df.apply(
        lambda r: (
            relative_error(r["predicted_circularity"], r["true_circularity"])
            if not r.get("invalid_circularity", False)
            else float("inf")
        ),
        axis=1,
    )

    return df

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute clean->noise metric deltas per cell_id/seed/model/severity."""
    clean_df = df[df["degradation"] == "clean"].copy()
    noise_df = df[(df["degradation"] == "gaussian_noise")].copy()

    merge_keys = ["split", "cell_id", "class", "seed", "model", "lambda_nc"]

    clean_cols = {
        "nucleus_area_absolute_error": "clean_area_abs_err",
        "direct_perimeter_absolute_error": "clean_perim_abs_err",
        "direct_perimeter_relative_error": "clean_perim_rel_err",
        "area_relative_error": "clean_area_rel_err",
        "circularity_absolute_error": "clean_circ_abs_err",
        "circularity_relative_error": "clean_circ_rel_err",
        "nc_absolute_error": "clean_nc_abs_err",
        "nc_relative_error": "clean_nc_rel_err",
        "foreground_dice": "clean_fg_dice",
        "nucleus_dice": "clean_nuc_dice",
        "direct_true_perimeter": "direct_true_perimeter",
        "direct_predicted_perimeter": "clean_pred_perimeter",
        "true_nucleus_area": "true_nucleus_area",
        "predicted_nucleus_area": "clean_pred_nucleus_area",
        "true_circularity": "true_circularity",
        "predicted_circularity": "clean_pred_circularity",
        "true_nc": "true_nc",
        "predicted_nc": "clean_pred_nc",
    }
    clean_sub = clean_df[merge_keys + list(clean_cols.keys())].rename(columns=clean_cols)
    merged = noise_df.merge(clean_sub, on=merge_keys, how="inner")

    merged["delta_area_abs_err"] = merged["nucleus_area_absolute_error"] - merged["clean_area_abs_err"]
    merged["delta_area_rel_err"] = merged["area_relative_error"] - merged["clean_area_rel_err"]
    merged["delta_perim_abs_err"] = merged["direct_perimeter_absolute_error"] - merged["clean_perim_abs_err"]
    merged["delta_perim_rel_err"] = merged["direct_perimeter_relative_error"] - merged["clean_perim_rel_err"]
    merged["delta_circ_abs_err"] = merged["circularity_absolute_error"] - merged["clean_circ_abs_err"]
    merged["delta_circ_rel_err"] = merged["circularity_relative_error"] - merged["clean_circ_rel_err"]
    merged["delta_nc_abs_err"] = merged["nc_absolute_error"] - merged["clean_nc_abs_err"]
    merged["delta_nc_rel_err"] = merged["nc_relative_error"] - merged["clean_nc_rel_err"]
    merged["delta_fg_dice"] = merged["foreground_dice"] - merged["clean_fg_dice"]
    merged["delta_nuc_dice"] = merged["nucleus_dice"] - merged["clean_nuc_dice"]

    out_cols = (
        merge_keys + ["degradation", "severity"] +
        ["true_nucleus_area", "direct_true_perimeter", "true_circularity", "true_nc",
         "clean_pred_nucleus_area", "clean_pred_perimeter", "clean_pred_circularity", "clean_pred_nc",
         "clean_area_abs_err", "clean_perim_abs_err", "clean_circ_abs_err", "clean_nc_abs_err",
         "nucleus_area_absolute_error", "direct_perimeter_absolute_error",
         "circularity_absolute_error", "nc_absolute_error",
         "delta_area_abs_err", "delta_area_rel_err",
         "delta_perim_abs_err", "delta_perim_rel_err",
         "delta_circ_abs_err", "delta_circ_rel_err",
         "delta_nc_abs_err", "delta_nc_rel_err",
         "delta_fg_dice", "delta_nuc_dice",
         "invalid_circularity", "invalid_nc"]
    )
    return merged[[c for c in out_cols if c in merged.columns]]


def run(out_dir: Path = OUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading degradation predictions CSV ...")
    df = pd.read_csv(DEGRADATION_CSV, dtype={"severity": str})
    print(f"  {len(df):,} rows loaded.")

    print("Enriching with DIRECT perimeter and relative errors ...")
    enriched = enrich_with_direct_perimeter(df)

    ap_df = enriched[(enriched["degradation"].isin(["clean", "gaussian_noise"]))].copy()

    # Drop cells without direct perimeter calculation (e.g. non-test splits)
    ap_df = ap_df.dropna(subset=["direct_true_perimeter", "direct_predicted_perimeter"])

    out_ap = out_dir / "real_prediction_area_perimeter.csv"
    ap_df.to_csv(out_ap, index=False)
    print(f"  Written: {out_ap}  ({len(ap_df):,} rows)")

    print("Computing clean-->noise deltas ...")
    deltas = compute_deltas(ap_df)

    out_deltas = out_dir / "real_prediction_deltas.csv"
    deltas.to_csv(out_deltas, index=False)
    print(f"  Written: {out_deltas}  ({len(deltas):,} rows)")

    return ap_df, deltas


if __name__ == "__main__":
    run()
