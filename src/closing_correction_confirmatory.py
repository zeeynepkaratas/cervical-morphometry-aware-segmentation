"""Preregistered closing-correction confirmatory analysis.

This module runs no training and performs no tuning. It evaluates the locked
3x3 elliptical one-iteration morphological closing correction and a
deterministic perimeter-reduction-matched local null on the frozen test split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_prep.load_herlev import list_herlev_images, load_image_and_masks
from src.measurements.morphometry import compute_circularity, compute_nc_ratio
from src.pilot.config_io import load_config, resolve_herlev_data_dir
from src.pilot.metrics import dice_score, foreground_dice, masks_to_target, restore_prediction_to_original_size
from src.segmentation.train_unet import preprocess_rgb_image
from src.segmentation.unet_model import UNet


CONFIG_PATH = ROOT / "configs" / "pilot.yaml"
SPLIT_PATH = ROOT / "data" / "splits" / "original_herlev_group_split.json"
CHECKPOINT_DIR = ROOT / "results" / "pilot" / "fair_repeat" / "checkpoints"
OUT_DIR = ROOT / "results" / "closing_correction"
PREREG_PATH = ROOT / "docs" / "closing_correction_preregistration.md"

FAIR_SEEDS = [20260803, 20260804, 20260805]
MODELS = ["baseline", "nc_aware_lambda_0p10"]
PRIMARY_MODEL = "baseline"
SENSITIVITY_MODEL = "nc_aware_lambda_0p10"
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 20260812
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
NULL_TOLERANCE = 0.10


def git_sha(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_perimeter(mask: np.ndarray) -> float:
    mask_u8 = np.asarray(mask).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float("nan")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return float("nan")
    return float(cv2.arcLength(contour, True))


def valid_prediction(nucleus: np.ndarray, cytoplasm: np.ndarray) -> bool:
    if not np.asarray(nucleus).astype(bool).any():
        return False
    if not np.asarray(cytoplasm).astype(bool).any():
        return False
    return math.isfinite(safe_perimeter(nucleus))


def apply_locked_closing(raw_nucleus: np.ndarray, raw_cytoplasm: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    pred_cell = np.logical_or(raw_nucleus, raw_cytoplasm)
    closed = cv2.morphologyEx(raw_nucleus.astype(np.uint8), cv2.MORPH_CLOSE, KERNEL, iterations=1).astype(bool)
    closed = np.logical_and(closed, pred_cell)
    corrected_cytoplasm = np.logical_and(pred_cell, ~closed)
    if not valid_prediction(closed, corrected_cytoplasm):
        return raw_nucleus.copy(), raw_cytoplasm.copy(), True
    return closed, corrected_cytoplasm, False


def local_boundary_pixels(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return np.empty((0, 2), dtype=int)
    eroded = cv2.erode(m.astype(np.uint8), KERNEL, iterations=1).astype(bool)
    boundary = np.logical_and(m, ~eroded)
    return np.argwhere(boundary)


def deterministic_seed(cell_id: str, model: str, seed: int) -> int:
    key = f"{cell_id}|{model}|{seed}|closing_null_v1".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def perimeter_matched_null(
    raw_nucleus: np.ndarray,
    raw_cytoplasm: np.ndarray,
    closing_nucleus: np.ndarray,
    *,
    cell_id: str,
    model: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a local random-location perimeter-reduction-matched null.

    The edit removes one deterministic-random raw nucleus boundary pixel at a
    time when removal reduces OpenCV external-contour perimeter and leaves a
    valid non-empty nucleus plus non-zero predicted cytoplasm-only region. It
    never uses ground truth, prediction error direction, or closing-changed
    pixel locations. If +/-10% matching cannot be reached, the closest
    achievable valid mask is used. Invalid nulls fall back to raw prediction.
    """
    raw_p = safe_perimeter(raw_nucleus)
    closing_p = safe_perimeter(closing_nucleus)
    target_reduction = abs(closing_p - raw_p) if math.isfinite(raw_p) and math.isfinite(closing_p) else 0.0
    pred_cell = np.logical_or(raw_nucleus, raw_cytoplasm)
    if target_reduction <= 0 or not math.isfinite(target_reduction):
        return raw_nucleus.copy(), raw_cytoplasm.copy(), {
            "null_target_reduction": float(target_reduction),
            "null_achieved_reduction": 0.0,
            "null_matching_error_abs": 0.0,
            "null_matching_error_pct": 0.0,
            "null_raw_fallback": False,
            "null_within_tolerance": True,
        }

    rng = np.random.default_rng(deterministic_seed(cell_id, model, seed))
    candidates = local_boundary_pixels(raw_nucleus)
    if len(candidates) == 0:
        return raw_nucleus.copy(), raw_cytoplasm.copy(), _null_fallback_record(target_reduction)
    order = rng.permutation(len(candidates))
    current = raw_nucleus.copy().astype(bool)
    current_p = raw_p
    best = current.copy()
    best_reduction = 0.0
    best_error = abs(target_reduction)
    tolerance_abs = NULL_TOLERANCE * target_reduction

    for idx in order:
        y, x = candidates[idx]
        if not current[y, x]:
            continue
        trial = current.copy()
        trial[y, x] = False
        trial_cytoplasm = np.logical_and(pred_cell, ~trial)
        if not valid_prediction(trial, trial_cytoplasm):
            continue
        trial_p = safe_perimeter(trial)
        reduction = raw_p - trial_p
        if reduction <= 0 or not math.isfinite(reduction):
            continue
        current = trial
        current_p = trial_p
        error = abs(reduction - target_reduction)
        if error < best_error:
            best = trial.copy()
            best_reduction = float(reduction)
            best_error = float(error)
        if abs(reduction - target_reduction) <= tolerance_abs:
            best = trial.copy()
            best_reduction = float(reduction)
            best_error = float(abs(reduction - target_reduction))
            break
        if reduction > target_reduction * (1.0 + NULL_TOLERANCE):
            break

    null_cytoplasm = np.logical_and(pred_cell, ~best)
    if not valid_prediction(best, null_cytoplasm):
        return raw_nucleus.copy(), raw_cytoplasm.copy(), _null_fallback_record(target_reduction)
    pct = best_error / target_reduction if target_reduction else 0.0
    return best, null_cytoplasm, {
        "null_target_reduction": float(target_reduction),
        "null_achieved_reduction": float(best_reduction),
        "null_matching_error_abs": float(best_error),
        "null_matching_error_pct": float(pct),
        "null_raw_fallback": False,
        "null_within_tolerance": bool(pct <= NULL_TOLERANCE),
    }


def _null_fallback_record(target_reduction: float) -> dict:
    return {
        "null_target_reduction": float(target_reduction),
        "null_achieved_reduction": 0.0,
        "null_matching_error_abs": float(target_reduction),
        "null_matching_error_pct": 1.0 if target_reduction else 0.0,
        "null_raw_fallback": True,
        "null_within_tolerance": False if target_reduction else True,
    }


def metric_row(nucleus: np.ndarray, cytoplasm: np.ndarray, gt_nucleus: np.ndarray, gt_cytoplasm: np.ndarray) -> dict:
    gt_target = masks_to_target(gt_nucleus, gt_cytoplasm)
    pred_target = np.zeros(gt_target.shape, dtype=np.uint8)
    pred_target[np.logical_and(cytoplasm, ~nucleus)] = 1
    pred_target[nucleus] = 2
    gt_nc = compute_nc_ratio(gt_nucleus, gt_cytoplasm)
    gt_c = compute_circularity(gt_nucleus)
    gt_p = safe_perimeter(gt_nucleus)
    pred_p = safe_perimeter(nucleus)
    invalid = not valid_prediction(nucleus, cytoplasm)
    try:
        pred_nc = compute_nc_ratio(nucleus, cytoplasm)
        nc_ae = abs(pred_nc - gt_nc)
        invalid_nc = False
    except ValueError:
        pred_nc = float("nan")
        nc_ae = float("nan")
        invalid_nc = True
    try:
        pred_c = compute_circularity(nucleus)
        circ_ae = abs(pred_c - gt_c)
        invalid_c = False
    except ValueError:
        pred_c = float("nan")
        circ_ae = float("nan")
        invalid_c = True
    return {
        "nucleus_dice": dice_score(nucleus, gt_nucleus),
        "foreground_dice": foreground_dice(pred_target, gt_target),
        "gt_circularity": gt_c,
        "pred_circularity": pred_c,
        "circularity_ae": circ_ae,
        "gt_perimeter": gt_p,
        "pred_perimeter": pred_p,
        "perimeter_ae": abs(pred_p - gt_p) if math.isfinite(pred_p) and math.isfinite(gt_p) else float("nan"),
        "gt_nc": gt_nc,
        "pred_nc": pred_nc,
        "nc_ae": nc_ae,
        "gt_nucleus_area": int(np.count_nonzero(gt_nucleus)),
        "pred_nucleus_area": int(np.count_nonzero(nucleus)),
        "area_ae": abs(int(np.count_nonzero(nucleus)) - int(np.count_nonzero(gt_nucleus))),
        "invalid": bool(invalid or invalid_nc or invalid_c),
        "invalid_nc": bool(invalid_nc),
        "invalid_circularity": bool(invalid_c),
    }


def load_model(config: dict, model_name: str, seed: int, device: torch.device) -> UNet:
    path = CHECKPOINT_DIR / f"{model_name}_seed_{seed}_best_val_dice.pt"
    ckpt = torch.load(path, map_location=device)
    model = UNet(base_channels=int(config["training"]["base_channels"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_inference_rows(config: dict, data_dir: Path, cell_ids: list[str]) -> tuple[list[dict], list[dict]]:
    device = torch.device(config["training"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    image_size = tuple(config["image_size"])
    image_by_id = {p.stem: p for p in list_herlev_images(data_dir)}
    rows: list[dict] = []
    match_rows: list[dict] = []
    for model_name in MODELS:
        for seed in FAIR_SEEDS:
            print(f"loading {model_name} seed {seed}", flush=True)
            model = load_model(config, model_name, seed, device)
            with torch.no_grad():
                for cell_id in cell_ids:
                    path = image_by_id[cell_id]
                    image, gt_nucleus, gt_cytoplasm = load_image_and_masks(path)
                    logits = model(preprocess_rgb_image(image, size=image_size).unsqueeze(0).to(device))
                    pred_resized = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    pred_target = restore_prediction_to_original_size(pred_resized, image.shape[:2], image_size)
                    raw_nucleus = pred_target == 2
                    raw_cytoplasm = pred_target == 1
                    closing_nucleus, closing_cytoplasm, closing_fallback = apply_locked_closing(raw_nucleus, raw_cytoplasm)
                    null_nucleus, null_cytoplasm, null_info = perimeter_matched_null(
                        raw_nucleus,
                        raw_cytoplasm,
                        closing_nucleus,
                        cell_id=cell_id,
                        model=model_name,
                        seed=seed,
                    )
                    for processing, nucleus, cytoplasm, fallback in [
                        ("raw", raw_nucleus, raw_cytoplasm, False),
                        ("closing", closing_nucleus, closing_cytoplasm, closing_fallback),
                        ("null", null_nucleus, null_cytoplasm, bool(null_info["null_raw_fallback"])),
                    ]:
                        m = metric_row(nucleus, cytoplasm, gt_nucleus, gt_cytoplasm)
                        rows.append({
                            "cell_id": cell_id,
                            "class": path.parent.name,
                            "model": model_name,
                            "seed": seed,
                            "processing": processing,
                            "fallback_to_raw": fallback,
                            **m,
                        })
                    raw_p = safe_perimeter(raw_nucleus)
                    closing_p = safe_perimeter(closing_nucleus)
                    null_p = safe_perimeter(null_nucleus)
                    raw_area = int(np.count_nonzero(raw_nucleus))
                    closing_area = int(np.count_nonzero(closing_nucleus))
                    null_area = int(np.count_nonzero(null_nucleus))
                    match_rows.append({
                        "cell_id": cell_id,
                        "class": path.parent.name,
                        "model": model_name,
                        "seed": seed,
                        "raw_perimeter": raw_p,
                        "closing_perimeter": closing_p,
                        "null_perimeter": null_p,
                        "delta_perimeter_closing": closing_p - raw_p,
                        "delta_perimeter_null": null_p - raw_p,
                        "delta_area_closing": closing_area - raw_area,
                        "delta_area_null": null_area - raw_area,
                        "abs_delta_area_closing_minus_null": abs((closing_area - raw_area) - (null_area - raw_area)),
                        "closing_fallback_to_raw": closing_fallback,
                        **null_info,
                    })
    return rows, match_rows


def median_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def cell_medians(rows: list[dict], model_name: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["model"] == model_name:
            grouped[row["cell_id"]].append(row)
    out = []
    for cell_id, cell_rows in sorted(grouped.items()):
        result = {"cell_id": cell_id, "class": cell_rows[0]["class"], "model": model_name}
        for processing in ["raw", "closing", "null"]:
            subset = [r for r in cell_rows if r["processing"] == processing]
            for key in [
                "nucleus_dice", "foreground_dice", "circularity_ae", "perimeter_ae",
                "nc_ae", "area_ae", "pred_circularity", "gt_circularity",
                "pred_perimeter", "gt_perimeter", "pred_nucleus_area", "gt_nucleus_area",
            ]:
                result[f"{processing}_{key}"] = median_or_nan([r[key] for r in subset])
            result[f"{processing}_invalid_rate"] = float(np.mean([bool(r["invalid"]) for r in subset])) if subset else float("nan")
        out.append(result)
    return out


def paired_from_cell_medians(cell_rows: list[dict]) -> list[dict]:
    out = []
    for r in cell_rows:
        row = {"cell_id": r["cell_id"], "class": r["class"], "model": r["model"]}
        for metric in ["circularity_ae", "perimeter_ae", "nc_ae", "area_ae", "nucleus_dice", "foreground_dice"]:
            row[f"delta_closing_{metric}"] = r[f"closing_{metric}"] - r[f"raw_{metric}"]
            row[f"delta_null_{metric}"] = r[f"null_{metric}"] - r[f"raw_{metric}"]
            row[f"delta_specific_{metric}"] = row[f"delta_closing_{metric}"] - row[f"delta_null_{metric}"]
        row["closing_benefit_circularity_ae"] = r["raw_circularity_ae"] - r["closing_circularity_ae"]
        row["null_benefit_circularity_ae"] = r["raw_circularity_ae"] - r["null_circularity_ae"]
        row["raw_circularity_signed_bias"] = r["raw_pred_circularity"] - r["raw_gt_circularity"]
        row["roundness_group"] = "under_circular" if row["raw_circularity_signed_bias"] < 0 else "over_circular"
        row["raw_invalid_rate"] = r["raw_invalid_rate"]
        row["closing_invalid_rate"] = r["closing_invalid_rate"]
        row["null_invalid_rate"] = r["null_invalid_rate"]
        out.append(row)
    return out


def bootstrap_ci(values: list[float], repeats: int = BOOTSTRAP_REPEATS, seed: int = BOOTSTRAP_SEED) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(repeats, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n": int(arr.size),
    }


def summarize_metrics(cell_rows: list[dict], prefix: str) -> dict:
    out = {}
    for metric in ["circularity_ae", "perimeter_ae", "nc_ae", "area_ae", "nucleus_dice", "foreground_dice"]:
        vals = [r[f"{prefix}_{metric}"] for r in cell_rows]
        out[f"{prefix}_{metric}_mean"] = float(np.nanmean(vals))
        out[f"{prefix}_{metric}_median"] = float(np.nanmedian(vals))
    out[f"{prefix}_invalid_rate"] = float(np.nanmean([r[f"{prefix}_invalid_rate"] for r in cell_rows]))
    return out


def spearman_like(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    keep = np.isfinite(xa) & np.isfinite(ya)
    if keep.sum() < 3:
        return float("nan")
    xr = np.argsort(np.argsort(xa[keep])).astype(float)
    yr = np.argsort(np.argsort(ya[keep])).astype(float)
    return float(np.corrcoef(xr, yr)[0, 1])


def build_roundness_outputs(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    subgroup_rows = []
    for group in ["under_circular", "over_circular"]:
        subset = [r for r in pairs if r["roundness_group"] == group]
        vals = [r["delta_closing_circularity_ae"] for r in subset]
        ci = bootstrap_ci(vals)
        subgroup_rows.append({
            "roundness_group": group,
            "n": len(subset),
            "mean_delta_closing_circularity_ae": ci["mean"],
            "median_delta_closing_circularity_ae": ci["median"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "mean_closing_benefit": float(np.nanmean([r["closing_benefit_circularity_ae"] for r in subset])) if subset else float("nan"),
        })
    rho = spearman_like(
        [r["raw_circularity_signed_bias"] for r in pairs],
        [r["closing_benefit_circularity_ae"] for r in pairs],
    )
    continuous = []
    for r in pairs:
        continuous.append({
            "cell_id": r["cell_id"],
            "raw_circularity_signed_bias": r["raw_circularity_signed_bias"],
            "closing_benefit_circularity_ae": r["closing_benefit_circularity_ae"],
            "spearman_rho_all_cells": rho,
        })
    return subgroup_rows, continuous


def build_perimeter_mechanism(pairs: list[dict]) -> list[dict]:
    decomp_path = ROOT / "results" / "circularity_mechanism" / "log_decomposition_per_cell.csv"
    contribs: dict[str, list[float]] = defaultdict(list)
    with decomp_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == PRIMARY_MODEL:
                try:
                    contribs[row["cell_id"]].append(float(row["abs_perimeter_contribution"]))
                except (KeyError, ValueError):
                    pass
    rows = []
    xs, ys = [], []
    for r in pairs:
        c = median_or_nan(contribs.get(r["cell_id"], []))
        xs.append(c)
        ys.append(r["closing_benefit_circularity_ae"])
    rho = spearman_like(xs, ys)
    for r, c in zip(pairs, xs):
        rows.append({
            "cell_id": r["cell_id"],
            "abs_perimeter_contribution_median_existing": c,
            "closing_benefit_circularity_ae": r["closing_benefit_circularity_ae"],
            "spearman_rho_all_cells": rho,
        })
    return rows


def decomposition_bootstrap() -> dict:
    per_cell = ROOT / "results" / "circularity_mechanism" / "log_decomposition_per_cell.csv"
    reg = ROOT / "results" / "circularity_mechanism" / "area_perimeter_regression.csv"
    rows = list(csv.DictReader(per_cell.open(newline="", encoding="utf-8")))
    cells = sorted({r["cell_id"] for r in rows})
    by_cell = {c: [r for r in rows if r["cell_id"] == c] for c in cells}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    fractions = np.empty(BOOTSTRAP_REPEATS)
    for i in range(BOOTSTRAP_REPEATS):
        sample = rng.choice(cells, size=len(cells), replace=True)
        selected = [r for c in sample for r in by_cell[c]]
        fractions[i] = np.mean([r["dominant_component"] == "perimeter" for r in selected])
    r2_values = []
    with reg.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("scope") == "all":
                try:
                    r2_values.append(float(row["r2_full_model"]))
                except ValueError:
                    pass
    return {
        "perimeter_dominant_fraction": {
            "mean": float(np.mean([r["dominant_component"] == "perimeter" for r in rows])),
            "ci_low": float(np.percentile(fractions, 2.5)),
            "ci_high": float(np.percentile(fractions, 97.5)),
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "n_cells": len(cells),
        },
        "r2_existing_all_scope": {
            "values": r2_values,
            "min": float(np.min(r2_values)) if r2_values else float("nan"),
            "max": float(np.max(r2_values)) if r2_values else float("nan"),
            "note": "Existing all-scope R2 values are reported from the locked regression output; no new regression model is selected.",
        },
    }


def build_verdict(primary_pairs: list[dict]) -> dict:
    delta_closing = bootstrap_ci([r["delta_closing_circularity_ae"] for r in primary_pairs])
    delta_specific = bootstrap_ci([r["delta_specific_circularity_ae"] for r in primary_pairs])
    pct_improved = float(np.mean([r["delta_closing_circularity_ae"] < 0 for r in primary_pairs]) * 100.0)
    primary_pass = bool(delta_closing["ci_high"] < 0 and pct_improved > 50.0)
    safety = {}
    safety_failures = []
    specs = {
        "nucleus_dice": "lower_worse",
        "foreground_dice": "lower_worse",
        "nc_ae": "higher_worse",
        "area_ae": "higher_worse",
    }
    for metric, direction in specs.items():
        ci = bootstrap_ci([r[f"delta_closing_{metric}"] for r in primary_pairs])
        safety[metric] = ci
        if direction == "lower_worse" and ci["ci_high"] < 0:
            safety_failures.append(metric)
        if direction == "higher_worse" and ci["ci_low"] > 0:
            safety_failures.append(metric)
    raw_invalid = float(np.mean([r["raw_invalid_rate"] for r in primary_pairs]))
    closing_invalid = float(np.mean([r["closing_invalid_rate"] for r in primary_pairs]))
    invalid_non_increasing = bool(closing_invalid <= raw_invalid + 1e-12)
    specificity_pass = bool(delta_specific["ci_high"] < 0)
    if not primary_pass:
        verdict = "CORRECTION_NOT_SUPPORTED"
    elif safety_failures or not invalid_non_increasing:
        verdict = "CORRECTION_UNSAFE"
    elif specificity_pass:
        verdict = "CORRECTION_SUPPORTED"
    else:
        verdict = "CORRECTION_UNSPECIFIC"
    return {
        "verdict": verdict,
        "primary_pass": primary_pass,
        "primary_delta_closing": delta_closing,
        "pct_cells_improved": pct_improved,
        "safety": safety,
        "safety_failures": safety_failures,
        "invalid_rate_raw": raw_invalid,
        "invalid_rate_closing": closing_invalid,
        "invalid_non_increasing": invalid_non_increasing,
        "mechanism_specificity_pass": specificity_pass,
        "delta_specific": delta_specific,
    }


def summarize_null_matching(match_rows: list[dict], model_name: str) -> dict:
    rows = [r for r in match_rows if r["model"] == model_name]
    return {
        "mean_target_perimeter_reduction": float(np.nanmean([r["null_target_reduction"] for r in rows])),
        "mean_achieved_null_reduction": float(np.nanmean([r["null_achieved_reduction"] for r in rows])),
        "mean_matching_error_abs": float(np.nanmean([r["null_matching_error_abs"] for r in rows])),
        "mean_matching_error_pct": float(np.nanmean([r["null_matching_error_pct"] for r in rows])),
        "within_tolerance_rate": float(np.mean([bool(r["null_within_tolerance"]) for r in rows])),
        "raw_fallback_rate": float(np.mean([bool(r["null_raw_fallback"]) for r in rows])),
    }


def write_summary(summary: dict, path: Path) -> None:
    lines = [
        "# Closing Correction Confirmatory Analysis",
        "",
        f"Verdict: `{summary['verdict']['verdict']}`",
        f"Primary pass: `{summary['verdict']['primary_pass']}`",
        f"Mechanism specificity pass: `{summary['verdict']['mechanism_specificity_pass']}`",
        "",
        "## Primary Baseline",
        "",
        f"Raw circularity AE mean: `{summary['primary_raw']['raw_circularity_ae_mean']}`",
        f"Closing circularity AE mean: `{summary['primary_closing']['closing_circularity_ae_mean']}`",
        f"Delta closing mean: `{summary['verdict']['primary_delta_closing']['mean']}`",
        f"Delta closing CI: `{summary['verdict']['primary_delta_closing']['ci_low']}`, `{summary['verdict']['primary_delta_closing']['ci_high']}`",
        f"Percent improved: `{summary['verdict']['pct_cells_improved']}`",
        "",
        "No training, no new corruption, no lambda/kernel/operation sweep.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(CONFIG_PATH)
    data_dir = resolve_herlev_data_dir(config)
    splits = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    test_ids = list(splits["test"])
    prereg_commit = git_sha(["log", "--format=%H", "--", str(PREREG_PATH.relative_to(ROOT))]).splitlines()[0]
    metadata = {
        "repo": str(ROOT),
        "branch": git_sha(["rev-parse", "--abbrev-ref", "HEAD"]),
        "starting_base_commit": "ae811bbb458198dcb16b77d6c39ae5987bb640f2",
        "preregistration_commit": prereg_commit,
        "analysis_commit_at_runtime": git_sha(["rev-parse", "HEAD"]),
        "frozen_test_n": len(test_ids),
        "primary_model": PRIMARY_MODEL,
        "sensitivity_model": SENSITIVITY_MODEL,
        "training": "NO",
        "new_corruption": "NO",
        "kernel": "3x3 ellipse",
        "operation": "closing",
        "iterations": 1,
        "null_tolerance": NULL_TOLERANCE,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "split_sha256": sha256_file(SPLIT_PATH),
        "preregistration_sha256": sha256_file(PREREG_PATH),
    }
    (OUT_DIR / "protocol_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    seed_rows, match_rows = run_inference_rows(config, data_dir, test_ids)
    write_csv(OUT_DIR / "test_results_per_seed.csv", seed_rows)
    write_csv(OUT_DIR / "null_control_matching.csv", match_rows)
    write_csv(OUT_DIR / "area_delta_comparison.csv", match_rows)

    baseline_cells = cell_medians(seed_rows, PRIMARY_MODEL)
    sensitivity_cells = cell_medians(seed_rows, SENSITIVITY_MODEL)
    write_csv(OUT_DIR / "test_results_cell_median.csv", baseline_cells)
    primary_pairs = paired_from_cell_medians(baseline_cells)
    sensitivity_pairs = paired_from_cell_medians(sensitivity_cells)
    write_csv(OUT_DIR / "paired_differences.csv", primary_pairs)
    write_csv(OUT_DIR / "sensitivity_nc_aware.csv", sensitivity_pairs)

    roundness, continuous = build_roundness_outputs(primary_pairs)
    write_csv(OUT_DIR / "roundness_audit.csv", roundness)
    write_csv(OUT_DIR / "continuous_roundness_analysis.csv", continuous)
    write_csv(OUT_DIR / "perimeter_mechanism_analysis.csv", build_perimeter_mechanism(primary_pairs))
    null_comparison = [{
        "cell_id": r["cell_id"],
        "delta_closing_circularity_ae": r["delta_closing_circularity_ae"],
        "delta_null_circularity_ae": r["delta_null_circularity_ae"],
        "delta_specific_circularity_ae": r["delta_specific_circularity_ae"],
        "closing_benefit_circularity_ae": r["closing_benefit_circularity_ae"],
        "null_benefit_circularity_ae": r["null_benefit_circularity_ae"],
    } for r in primary_pairs]
    write_csv(OUT_DIR / "null_control_comparison.csv", null_comparison)

    decomp_ci = decomposition_bootstrap()
    (OUT_DIR / "decomposition_bootstrap_ci.json").write_text(json.dumps(decomp_ci, indent=2), encoding="utf-8")
    verdict = build_verdict(primary_pairs)
    (OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    primary_raw = summarize_metrics(baseline_cells, "raw")
    primary_closing = summarize_metrics(baseline_cells, "closing")
    primary_null = summarize_metrics(baseline_cells, "null")
    sensitivity_summary = {
        "raw": summarize_metrics(sensitivity_cells, "raw"),
        "closing": summarize_metrics(sensitivity_cells, "closing"),
        "paired_delta_closing_circularity": bootstrap_ci([r["delta_closing_circularity_ae"] for r in sensitivity_pairs]),
    }
    full_summary = {
        "metadata": metadata,
        "primary_raw": primary_raw,
        "primary_closing": primary_closing,
        "primary_null": primary_null,
        "primary_delta_perimeter": bootstrap_ci([r["delta_closing_perimeter_ae"] for r in primary_pairs]),
        "primary_delta_nc": bootstrap_ci([r["delta_closing_nc_ae"] for r in primary_pairs]),
        "primary_delta_area": bootstrap_ci([r["delta_closing_area_ae"] for r in primary_pairs]),
        "primary_delta_nucleus_dice": bootstrap_ci([r["delta_closing_nucleus_dice"] for r in primary_pairs]),
        "primary_delta_foreground_dice": bootstrap_ci([r["delta_closing_foreground_dice"] for r in primary_pairs]),
        "null_delta": bootstrap_ci([r["delta_null_circularity_ae"] for r in primary_pairs]),
        "null_matching": summarize_null_matching(match_rows, PRIMARY_MODEL),
        "roundness": roundness,
        "continuous_roundness_spearman": continuous[0]["spearman_rho_all_cells"] if continuous else float("nan"),
        "perimeter_mechanism_spearman": build_perimeter_mechanism(primary_pairs)[0]["spearman_rho_all_cells"] if primary_pairs else float("nan"),
        "sensitivity_nc_aware": sensitivity_summary,
        "decomposition_bootstrap_ci": decomp_ci,
        "verdict": verdict,
        "protocol_deviation": "NONE",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    write_summary(full_summary, OUT_DIR / "summary.md")
    print(json.dumps({
        "status": "CLOSING_CORRECTION_ANALYSIS_COMPLETE",
        "verdict": verdict["verdict"],
        "primary_delta": verdict["primary_delta_closing"],
        "pct_improved": verdict["pct_cells_improved"],
    }, indent=2))


if __name__ == "__main__":
    run()
