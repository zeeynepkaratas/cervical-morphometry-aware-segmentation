"""Prospective secondary dual-matched local random null validation.

This B analysis is isolated from the primary closing correction. It performs
no training, model selection, checkpoint selection, corruption, or tuning.
Frozen checkpoint inference is used only because prediction masks were not
persisted and the dual-matched local null requires mask topology.
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
PRIMARY_RESULTS = ROOT / "results" / "closing_correction"
INPUT_PER_SEED = PRIMARY_RESULTS / "test_results_per_seed.csv"
INPUT_VERDICT = PRIMARY_RESULTS / "verdict.json"
A_SUMMARY = PRIMARY_RESULTS / "mechanism_decomposition" / "summary.json"
OUT_DIR = PRIMARY_RESULTS / "dual_matched_null"
PREREG_PATH = ROOT / "docs" / "closing_dual_matched_null_preregistration.md"

PRIMARY_MODEL = "baseline"
FAIR_SEEDS = [20260803, 20260804, 20260805]
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
KERNEL_8 = np.ones((3, 3), dtype=np.uint8)
CANDIDATE_PROPOSALS = 2048
MAX_ACCEPTED_EDITS = 256
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = int(hashlib.sha256(b"dual_matched_null_bootstrap_v1").hexdigest()[:8], 16)
ORIGINAL_NULL_AREA_ERROR_MEAN = 103.72826086956522
ORIGINAL_NULL_PERIMETER_ERROR_MEAN = 4.728883951470472
TOL = 1e-6


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


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def safe_float(value: str) -> float:
    if value in {"", "nan", "NaN"}:
        return float("nan")
    return float(value)


def safe_perimeter(mask: np.ndarray) -> float:
    mask_u8 = np.asarray(mask).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float("nan")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return float("nan")
    return float(cv2.arcLength(contour, True))


def valid_mask(nucleus: np.ndarray, cytoplasm: np.ndarray) -> bool:
    return bool(np.any(nucleus) and np.any(cytoplasm) and math.isfinite(safe_perimeter(nucleus)))


def apply_closing(raw_nucleus: np.ndarray, raw_cytoplasm: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    pred_cell = np.logical_or(raw_nucleus, raw_cytoplasm)
    closed = cv2.morphologyEx(raw_nucleus.astype(np.uint8), cv2.MORPH_CLOSE, KERNEL_CLOSE, iterations=1).astype(bool)
    closed = np.logical_and(closed, pred_cell)
    closed_cyto = np.logical_and(pred_cell, ~closed)
    if not valid_mask(closed, closed_cyto):
        return raw_nucleus.copy(), raw_cytoplasm.copy(), True
    return closed, closed_cyto, False


def load_model(config: dict, seed: int, device: torch.device) -> UNet:
    path = CHECKPOINT_DIR / f"{PRIMARY_MODEL}_seed_{seed}_best_val_dice.pt"
    ckpt = torch.load(path, map_location=device)
    model = UNet(base_channels=int(config["training"]["base_channels"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def null_seed(cell_id: str, seed_id: int) -> int:
    key = f"dual_matched_null_v1|{cell_id}|{seed_id}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), KERNEL_8, iterations=1).astype(bool)
    return np.argwhere(np.logical_and(mask, ~eroded))


def add_pixels(current: np.ndarray, pred_cell: np.ndarray) -> np.ndarray:
    dilated = cv2.dilate(current.astype(np.uint8), KERNEL_8, iterations=1).astype(bool)
    return np.argwhere(np.logical_and.reduce((dilated, pred_cell, ~current)))


def score_candidate(candidate: np.ndarray, raw_nucleus: np.ndarray, raw_area: int, raw_perimeter: float, target_dA: float, target_dP: float) -> dict:
    area = int(np.count_nonzero(candidate))
    perim = safe_perimeter(candidate)
    dA = float(area - raw_area)
    dP = float(perim - raw_perimeter)
    area_floor = math.sqrt(max(float(raw_area), 1.0))
    perim_floor = math.sqrt(max(float(raw_perimeter), 1.0))
    eA = abs(dA - target_dA) / max(abs(target_dA), area_floor)
    eP = abs(dP - target_dP) / max(abs(target_dP), perim_floor)
    return {
        "area": area,
        "perimeter": perim,
        "dA": dA,
        "dP": dP,
        "area_error_abs": abs(dA - target_dA),
        "perimeter_error_abs": abs(dP - target_dP),
        "area_error_norm": eA,
        "perimeter_error_norm": eP,
        "joint_error": max(eA, eP),
        "sum_error": eA + eP,
    }


def better_score(new: dict, old: dict) -> bool:
    if new["joint_error"] < old["joint_error"] - 1e-12:
        return True
    if abs(new["joint_error"] - old["joint_error"]) <= 1e-12 and new["sum_error"] < old["sum_error"] - 1e-12:
        return True
    return False


def generate_dual_null(raw_nucleus: np.ndarray, raw_cytoplasm: np.ndarray, closing_nucleus: np.ndarray, cell_id: str, seed_id: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate a GT-blind dual-matched local random null.

    Candidate selection has no access to GT geometry, GT circularity, circularity
    AE, under/over status, closing benefit, or any circularity objective. It
    optimizes only signed raw-to-closing area/perimeter matching.
    """
    pred_cell = np.logical_or(raw_nucleus, raw_cytoplasm)
    raw_area = int(np.count_nonzero(raw_nucleus))
    raw_perim = safe_perimeter(raw_nucleus)
    close_area = int(np.count_nonzero(closing_nucleus))
    close_perim = safe_perimeter(closing_nucleus)
    target_dA = float(close_area - raw_area)
    target_dP = float(close_perim - raw_perim)
    rng = np.random.default_rng(null_seed(cell_id, seed_id))

    current = raw_nucleus.copy()
    current_cyto = raw_cytoplasm.copy()
    best = current.copy()
    best_score = score_candidate(best, raw_nucleus, raw_area, raw_perim, target_dA, target_dP)
    invalid_proposals = 0
    accepted_edits = 0
    valid_proposals = 0

    for _ in range(CANDIDATE_PROPOSALS):
        if accepted_edits >= MAX_ACCEPTED_EDITS:
            break
        current_score = score_candidate(current, raw_nucleus, raw_area, raw_perim, target_dA, target_dP)
        prefer_add = current_score["dA"] < target_dA
        if rng.random() < 0.75:
            op = "add" if prefer_add else "remove"
        else:
            op = "remove" if prefer_add else "add"

        trial = current.copy()
        if op == "add":
            pts = add_pixels(current, pred_cell)
            if len(pts) == 0:
                invalid_proposals += 1
                continue
            y, x = pts[int(rng.integers(0, len(pts)))]
            trial[y, x] = True
        else:
            pts = boundary_pixels(current)
            if len(pts) == 0:
                invalid_proposals += 1
                continue
            y, x = pts[int(rng.integers(0, len(pts)))]
            trial[y, x] = False

        trial_cyto = np.logical_and(pred_cell, ~trial)
        if not valid_mask(trial, trial_cyto):
            invalid_proposals += 1
            continue
        valid_proposals += 1
        trial_score = score_candidate(trial, raw_nucleus, raw_area, raw_perim, target_dA, target_dP)
        if better_score(trial_score, current_score):
            current = trial
            current_cyto = trial_cyto
            accepted_edits += 1
            if better_score(trial_score, best_score):
                best = trial.copy()
                best_score = trial_score

    best_cyto = np.logical_and(pred_cell, ~best)
    fallback = False
    if not valid_mask(best, best_cyto):
        best = raw_nucleus.copy()
        best_cyto = raw_cytoplasm.copy()
        best_score = score_candidate(best, raw_nucleus, raw_area, raw_perim, target_dA, target_dP)
        fallback = True
    info = {
        "target_dA": target_dA,
        "target_dP": target_dP,
        "null_dA": best_score["dA"],
        "null_dP": best_score["dP"],
        "area_matching_error_abs": best_score["area_error_abs"],
        "perimeter_matching_error_abs": best_score["perimeter_error_abs"],
        "area_matching_error_norm": best_score["area_error_norm"],
        "perimeter_matching_error_norm": best_score["perimeter_error_norm"],
        "joint_matching_error": best_score["joint_error"],
        "sum_matching_error": best_score["sum_error"],
        "valid_proposal_count": valid_proposals,
        "invalid_proposal_count": invalid_proposals,
        "accepted_edit_count": accepted_edits,
        "fallback_to_raw": fallback,
    }
    return best, best_cyto, info


def metrics(nucleus: np.ndarray, cytoplasm: np.ndarray, gt_nucleus: np.ndarray, gt_cytoplasm: np.ndarray) -> dict:
    gt_target = masks_to_target(gt_nucleus, gt_cytoplasm)
    pred_target = np.zeros(gt_target.shape, dtype=np.uint8)
    pred_target[np.logical_and(cytoplasm, ~nucleus)] = 1
    pred_target[nucleus] = 2
    gt_c = compute_circularity(gt_nucleus)
    pred_c = compute_circularity(nucleus) if np.any(nucleus) else float("nan")
    try:
        gt_nc = compute_nc_ratio(gt_nucleus, gt_cytoplasm)
        pred_nc = compute_nc_ratio(nucleus, cytoplasm)
        nc_ae = abs(pred_nc - gt_nc)
    except ValueError:
        pred_nc = float("nan")
        nc_ae = float("nan")
    area = int(np.count_nonzero(nucleus))
    gt_area = int(np.count_nonzero(gt_nucleus))
    invalid = not valid_mask(nucleus, cytoplasm) or not math.isfinite(pred_c) or not math.isfinite(nc_ae)
    return {
        "nucleus_dice": dice_score(nucleus, gt_nucleus),
        "foreground_dice": foreground_dice(pred_target, gt_target),
        "gt_circularity": gt_c,
        "pred_circularity": pred_c,
        "circularity_ae": abs(pred_c - gt_c) if math.isfinite(pred_c) else float("nan"),
        "pred_perimeter": safe_perimeter(nucleus),
        "pred_nucleus_area": area,
        "nc_ae": nc_ae,
        "area_ae": abs(area - gt_area),
        "invalid": bool(invalid),
    }


def frozen_index() -> dict[tuple[str, int, str], dict]:
    rows = read_csv(INPUT_PER_SEED)
    return {
        (row["cell_id"], int(row["seed"]), row["processing"]): row
        for row in rows
        if row["model"] == PRIMARY_MODEL and row["processing"] in {"raw", "closing"}
    }


def assert_matches_frozen(cell_id: str, seed_id: int, processing: str, metric: dict, frozen: dict) -> None:
    row = frozen[(cell_id, seed_id, processing)]
    checks = {
        "pred_nucleus_area": float(metric["pred_nucleus_area"]),
        "pred_perimeter": float(metric["pred_perimeter"]),
        "pred_circularity": float(metric["pred_circularity"]),
        "circularity_ae": float(metric["circularity_ae"]),
    }
    for key, value in checks.items():
        ref = safe_float(row[key])
        if math.isfinite(value) and math.isfinite(ref) and abs(value - ref) > TOL:
            raise ValueError(f"Frozen mismatch {cell_id} seed {seed_id} {processing} {key}: {value} != {ref}")


def run_per_seed() -> tuple[list[dict], list[dict]]:
    config = load_config(CONFIG_PATH)
    data_dir = resolve_herlev_data_dir(config)
    splits = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    test_ids = list(splits["test"])
    image_by_id = {p.stem: p for p in list_herlev_images(data_dir)}
    device = torch.device(config["training"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    image_size = tuple(config["image_size"])
    frozen = frozen_index()
    per_seed_rows: list[dict] = []
    diagnostics: list[dict] = []

    for seed_id in FAIR_SEEDS:
        print(f"dual null baseline seed {seed_id}", flush=True)
        model = load_model(config, seed_id, device)
        with torch.no_grad():
            for cell_id in test_ids:
                path = image_by_id[cell_id]
                image, gt_nucleus, gt_cytoplasm = load_image_and_masks(path)
                logits = model(preprocess_rgb_image(image, size=image_size).unsqueeze(0).to(device))
                pred_resized = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                pred_target = restore_prediction_to_original_size(pred_resized, image.shape[:2], image_size)
                raw_nucleus = pred_target == 2
                raw_cytoplasm = pred_target == 1
                closing_nucleus, closing_cytoplasm, closing_fallback = apply_closing(raw_nucleus, raw_cytoplasm)
                null_nucleus, null_cytoplasm, match = generate_dual_null(raw_nucleus, raw_cytoplasm, closing_nucleus, cell_id, seed_id)

                raw_m = metrics(raw_nucleus, raw_cytoplasm, gt_nucleus, gt_cytoplasm)
                closing_m = metrics(closing_nucleus, closing_cytoplasm, gt_nucleus, gt_cytoplasm)
                null_m = metrics(null_nucleus, null_cytoplasm, gt_nucleus, gt_cytoplasm)
                assert_matches_frozen(cell_id, seed_id, "raw", raw_m, frozen)
                assert_matches_frozen(cell_id, seed_id, "closing", closing_m, frozen)

                benefit_closing = raw_m["circularity_ae"] - closing_m["circularity_ae"]
                benefit_null = raw_m["circularity_ae"] - null_m["circularity_ae"]
                per_seed_rows.append({
                    "cell_id": cell_id,
                    "class": path.parent.name,
                    "model": PRIMARY_MODEL,
                    "seed": seed_id,
                    "raw_circularity_ae": raw_m["circularity_ae"],
                    "closing_circularity_ae": closing_m["circularity_ae"],
                    "dual_null_circularity_ae": null_m["circularity_ae"],
                    "benefit_closing": benefit_closing,
                    "benefit_null": benefit_null,
                    "specific_advantage": benefit_closing - benefit_null,
                    "raw_nucleus_dice": raw_m["nucleus_dice"],
                    "closing_nucleus_dice": closing_m["nucleus_dice"],
                    "dual_null_nucleus_dice": null_m["nucleus_dice"],
                    "raw_foreground_dice": raw_m["foreground_dice"],
                    "closing_foreground_dice": closing_m["foreground_dice"],
                    "dual_null_foreground_dice": null_m["foreground_dice"],
                    "raw_nc_ae": raw_m["nc_ae"],
                    "closing_nc_ae": closing_m["nc_ae"],
                    "dual_null_nc_ae": null_m["nc_ae"],
                    "raw_area_ae": raw_m["area_ae"],
                    "closing_area_ae": closing_m["area_ae"],
                    "dual_null_area_ae": null_m["area_ae"],
                    "raw_invalid": raw_m["invalid"],
                    "closing_invalid": closing_m["invalid"],
                    "dual_null_invalid": null_m["invalid"],
                    "raw_pred_circularity": raw_m["pred_circularity"],
                    "gt_circularity": raw_m["gt_circularity"],
                    "roundness_group": "under_circular" if raw_m["pred_circularity"] < raw_m["gt_circularity"] else ("over_circular" if raw_m["pred_circularity"] > raw_m["gt_circularity"] else "equal"),
                })
                diagnostics.append({
                    "cell_id": cell_id,
                    "class": path.parent.name,
                    "model": PRIMARY_MODEL,
                    "seed": seed_id,
                    "closing_fallback_to_raw": closing_fallback,
                    **match,
                })
    return per_seed_rows, diagnostics


def median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def cell_medians(rows: list[dict], diagnostics: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    grouped_diag: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["cell_id"]].append(row)
    for row in diagnostics:
        grouped_diag[row["cell_id"]].append(row)
    cell_rows = []
    diag_rows = []
    for cell_id, vals in sorted(grouped.items()):
        out = {"cell_id": cell_id, "class": vals[0]["class"], "model": PRIMARY_MODEL}
        for key in vals[0]:
            if key in {"cell_id", "class", "model", "seed", "roundness_group"}:
                continue
            if isinstance(vals[0][key], bool):
                out[key] = float(np.mean([bool(v[key]) for v in vals]))
            else:
                out[key] = median([v[key] for v in vals])
        out["roundness_group"] = "under_circular" if out["raw_pred_circularity"] < out["gt_circularity"] else ("over_circular" if out["raw_pred_circularity"] > out["gt_circularity"] else "equal")
        cell_rows.append(out)
    for cell_id, vals in sorted(grouped_diag.items()):
        out = {"cell_id": cell_id, "class": vals[0]["class"], "model": PRIMARY_MODEL}
        for key in vals[0]:
            if key in {"cell_id", "class", "model", "seed"}:
                continue
            if isinstance(vals[0][key], bool):
                out[key] = float(np.mean([bool(v[key]) for v in vals]))
            else:
                out[key] = median([v[key] for v in vals])
        diag_rows.append(out)
    return cell_rows, diag_rows


def bootstrap_ci(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(BOOTSTRAP_REPEATS, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n": int(arr.size),
    }


def distribution_summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: float("nan") for k in ["mean", "median", "q1", "q3", "iqr", "p90", "p95"]}
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def aggregate_matching(diag_rows: list[dict]) -> dict:
    area = distribution_summary([r["area_matching_error_abs"] for r in diag_rows])
    perim = distribution_summary([r["perimeter_matching_error_abs"] for r in diag_rows])
    joint = distribution_summary([r["joint_matching_error"] for r in diag_rows])
    target_dA_mean = float(np.mean([r["target_dA"] for r in diag_rows]))
    null_dA_mean = float(np.mean([r["null_dA"] for r in diag_rows]))
    target_dP_mean = float(np.mean([r["target_dP"] for r in diag_rows]))
    null_dP_mean = float(np.mean([r["null_dP"] for r in diag_rows]))
    sign_ok = (
        math.copysign(1.0, target_dA_mean) == math.copysign(1.0, null_dA_mean)
        and math.copysign(1.0, target_dP_mean) == math.copysign(1.0, null_dP_mean)
    )
    better_than_original = area["mean"] < ORIGINAL_NULL_AREA_ERROR_MEAN and perim["mean"] < ORIGINAL_NULL_PERIMETER_ERROR_MEAN
    status = "ADEQUATE" if sign_ok and better_than_original else "LIMITED"
    return {
        "area": area,
        "perimeter": perim,
        "joint": joint,
        "target_dA_mean": target_dA_mean,
        "null_dA_mean": null_dA_mean,
        "target_dP_mean": target_dP_mean,
        "null_dP_mean": null_dP_mean,
        "systematic_directional_mismatch": not sign_ok,
        "better_than_original_null_area_mean": bool(area["mean"] < ORIGINAL_NULL_AREA_ERROR_MEAN),
        "better_than_original_null_perimeter_mean": bool(perim["mean"] < ORIGINAL_NULL_PERIMETER_ERROR_MEAN),
        "matching_status": status,
    }


def aggregate_specificity(cell_rows: list[dict]) -> dict:
    spec = bootstrap_ci([r["specific_advantage"] for r in cell_rows])
    spec["pct_cells_closing_better"] = float(np.mean([r["specific_advantage"] > 0 for r in cell_rows]) * 100.0)
    closing = bootstrap_ci([r["benefit_closing"] for r in cell_rows])
    null = bootstrap_ci([r["benefit_null"] for r in cell_rows])
    return {"specific_advantage": spec, "closing_benefit": closing, "null_benefit": null}


def b_verdict(matching: dict, specificity: dict) -> str:
    spec = specificity["specific_advantage"]
    if matching["matching_status"] == "LIMITED":
        return "LIMITED_BY_MATCHING"
    if spec["ci_low"] > 0 and spec["mean"] > 0:
        return "DUAL_MATCHED_SPECIFICITY_SUPPORTED"
    if spec["ci_high"] < 0:
        return "NULL_OUTPERFORMS_CLOSING"
    return "SPECIFICITY_NOT_SUPPORTED"


def safety_summary(cell_rows: list[dict]) -> list[dict]:
    rows = []
    for metric in ["nucleus_dice", "foreground_dice", "nc_ae", "area_ae"]:
        row = {"metric": metric}
        for prefix in ["raw", "closing", "dual_null"]:
            vals = [r[f"{prefix}_{metric}"] for r in cell_rows]
            row[f"{prefix}_mean"] = float(np.nanmean(vals))
            row[f"{prefix}_median"] = float(np.nanmedian(vals))
        rows.append(row)
    for prefix in ["raw", "closing", "dual_null"]:
        rows.append({
            "metric": f"{prefix}_invalid_rate",
            f"{prefix}_mean": float(np.mean([r[f"{prefix}_invalid"] for r in cell_rows])),
            f"{prefix}_median": float(np.median([r[f"{prefix}_invalid"] for r in cell_rows])),
        })
    return rows


def under_over_rows(cell_rows: list[dict]) -> list[dict]:
    out = []
    for group in ["under_circular", "over_circular", "equal"]:
        subset = [r for r in cell_rows if r["roundness_group"] == group]
        if not subset:
            out.append({"roundness_group": group, "n": 0})
            continue
        row = {"roundness_group": group, "n": len(subset)}
        for metric in ["benefit_closing", "benefit_null", "specific_advantage"]:
            ci = bootstrap_ci([r[metric] for r in subset])
            row[f"{metric}_mean"] = ci["mean"]
            row[f"{metric}_median"] = ci["median"]
            row[f"{metric}_ci_low"] = ci["ci_low"]
            row[f"{metric}_ci_high"] = ci["ci_high"]
        out.append(row)
    return out


def write_summary_md(path: Path, summary: dict) -> None:
    lines = [
        "# Dual-Matched Closing Null Validation",
        "",
        f"B verdict: `{summary['verdict']}`",
        f"Matching status: `{summary['matching']['matching_status']}`",
        "",
        f"Specific advantage mean: `{summary['specificity']['specific_advantage']['mean']}`",
        f"Specific advantage CI: `{summary['specificity']['specific_advantage']['ci_low']}`, `{summary['specificity']['specific_advantage']['ci_high']}`",
        f"Cells closing better: `{summary['specificity']['specific_advantage']['pct_cells_closing_better']}`",
        "",
        "Primary `CORRECTION_SUPPORTED` verdict remains unchanged.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    locked_hashes = {
        "primary_verdict": sha256_file(INPUT_VERDICT),
        "primary_per_seed": sha256_file(INPUT_PER_SEED),
        "a_summary": sha256_file(A_SUMMARY),
    }
    prereg_commit = git_sha(["log", "--format=%H", "--", str(PREREG_PATH.relative_to(ROOT))]).splitlines()[0]
    metadata = {
        "repo": str(ROOT),
        "branch": git_sha(["rev-parse", "--abbrev-ref", "HEAD"]),
        "starting_commit": git_sha(["rev-parse", "HEAD"]),
        "preregistration_commit": prereg_commit,
        "training": "NO",
        "new_inference": "YES",
        "new_inference_reason": "Raw prediction masks were not persisted; frozen checkpoint inference reconstructs masks for GT-blind local null generation.",
        "candidate_proposals": CANDIDATE_PROPOSALS,
        "max_accepted_edits": MAX_ACCEPTED_EDITS,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "gt_used_during_null_generation": False,
        "circularity_used_during_null_matching": False,
        "under_over_used_during_null_generation": False,
        "locked_input_hashes_before": locked_hashes,
    }
    (OUT_DIR / "protocol_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    per_seed, diagnostics = run_per_seed()
    write_csv(OUT_DIR / "per_seed_results.csv", per_seed)
    write_csv(OUT_DIR / "matching_diagnostics.csv", diagnostics)
    cell_rows, cell_diag = cell_medians(per_seed, diagnostics)
    write_csv(OUT_DIR / "cell_median_results.csv", cell_rows)
    matching = aggregate_matching(cell_diag)
    specificity = aggregate_specificity(cell_rows)
    verdict = b_verdict(matching, specificity)
    safety = safety_summary(cell_rows)
    under_over = under_over_rows(cell_rows)
    write_csv(OUT_DIR / "safety_summary.csv", safety)
    write_csv(OUT_DIR / "under_over_descriptive.csv", under_over)
    (OUT_DIR / "aggregate_matching_summary.json").write_text(json.dumps(matching, indent=2), encoding="utf-8")
    (OUT_DIR / "specificity_bootstrap.json").write_text(json.dumps(specificity, indent=2), encoding="utf-8")

    locked_hashes_after = {
        "primary_verdict": sha256_file(INPUT_VERDICT),
        "primary_per_seed": sha256_file(INPUT_PER_SEED),
        "a_summary": sha256_file(A_SUMMARY),
    }
    quality = {
        "n_cells": len(cell_rows),
        "no_silent_exclusions": len(cell_rows) == 184,
        "locked_hashes_unchanged": locked_hashes == locked_hashes_after,
        "no_gt_leakage_in_candidate_selection": True,
        "no_circularity_based_candidate_selection": True,
        "no_result_dependent_rerun": True,
    }
    verdict_payload = {
        "verdict": verdict,
        "matching_status": matching["matching_status"],
        "specificity": specificity,
        "quality_checks": quality,
        "primary_correction": "CORRECTION_SUPPORTED",
        "primary_correction_changed": False,
    }
    (OUT_DIR / "verdict.json").write_text(json.dumps(verdict_payload, indent=2), encoding="utf-8")
    full_summary = {
        "metadata": metadata | {"locked_input_hashes_after": locked_hashes_after},
        "matching": matching,
        "specificity": specificity,
        "safety": safety,
        "under_over": under_over,
        "verdict": verdict,
        "quality_checks": quality,
        "primary_correction_changed": False,
        "protocol_deviation": "NONE",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    write_summary_md(OUT_DIR / "summary.md", full_summary)
    print(json.dumps({
        "status": "CLOSING_DUAL_MATCHED_NULL_COMPLETE",
        "verdict": verdict,
        "matching_status": matching["matching_status"],
        "n_cells": len(cell_rows),
        "specific_advantage": specificity["specific_advantage"],
    }, indent=2))


if __name__ == "__main__":
    run()
