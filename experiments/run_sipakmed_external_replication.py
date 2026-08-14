"""Run the locked SIPaKMeD external replication.

This runner executes the protocol frozen in
``docs/SIPAKMED_EXTERNAL_REPLICATION_FINAL_LOCK.md``. It intentionally performs
no training, fine-tuning, checkpoint selection, preprocessing tuning, sample
reselection, or post-processing parameter search.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.closing_dual_matched_null import apply_closing, generate_dual_null, safe_perimeter, valid_mask
from src.data_prep.load_sipakmed import (
    SipakmedSample,
    load_sipakmed_sample,
)
from src.measurements.morphometry import compute_circularity, compute_nc_ratio
from src.pilot.metrics import dice_score, foreground_dice, masks_to_target, restore_prediction_to_original_size
from src.segmentation.train_unet import preprocess_rgb_image
from src.segmentation.unet_model import UNet


LOCK_COMMIT = "b0cb7f6d727503861a36b0d754f18c5d999cffe8"
LOCK_TAG = "sipakmed-external-lock-20260813"
REPLICATION_BRANCH = "codex/sipakmed-external-replication"
MANIFEST_PATH = ROOT / "results" / "sipakmed_preflight" / "sipakmed_eligibility_manifest.csv"
OUT_DIR = ROOT / "results" / "sipakmed_external_replication"
PARTIAL_DIR = OUT_DIR / "_partial"
REPORT_PATH = ROOT / "docs" / "SIPAKMED_EXTERNAL_REPLICATION_REPORT.md"
CHECKPOINT_DIR = ROOT / "results" / "pilot" / "fair_repeat" / "checkpoints"
FAIR_SEEDS = [20260803, 20260804, 20260805]
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 20260813
IMAGE_SIZE = (128, 128)
INFERENCE_BATCH_SIZE = 8
PRIMARY_MODEL = "baseline"
EXPECTED_MANIFEST_SHA256 = "e86e99441866fb03d457e3867032ecee4e6792285c6647d93e35089b923de1cd"
EXPECTED_CHECKPOINT_SHA256 = {
    20260803: "8c52642452c674bc4b929a7b643c9f69f0fc600d2f7b5eafdf733287e98eed51",
    20260804: "d8bf421f71b0ba899a9d70700f6bdb1155110fd91af8617daff97325aed2b6af",
    20260805: "a8e1563e5dc86f67ed5b393a8f1fc78a96165b727cedd766aa8c1131bc5fdc26",
}
CODE_HASH_PATHS = [
    "docs/SIPAKMED_EXTERNAL_REPLICATION_FINAL_LOCK.md",
    "experiments/run_sipakmed_external_replication.py",
    "src/data_prep/load_sipakmed.py",
    "src/segmentation/unet_model.py",
    "src/segmentation/train_unet.py",
    "src/measurements/morphometry.py",
    "src/pilot/metrics.py",
    "src/closing_dual_matched_null.py",
]


def git_sha(args: list[str], check: bool = False) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError:
        if check:
            raise
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


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=True, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def dedupe_rows(rows: list[dict], key_fields: list[str]) -> list[dict]:
    deduped: dict[tuple, dict] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        deduped[key] = row
    return [deduped[key] for key in sorted(deduped)]


def finite_values(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def finite_mean(values: Iterable[float]) -> float:
    arr = finite_values(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def finite_median(values: Iterable[float]) -> float:
    arr = finite_values(values)
    return float(np.median(arr)) if arr.size else float("nan")


def assert_preflight_state() -> dict:
    head = git_sha(["rev-parse", "HEAD"], check=True)
    branch = git_sha(["rev-parse", "--abbrev-ref", "HEAD"], check=True)
    tag_target = git_sha(["rev-parse", f"{LOCK_TAG}^{{}}"], check=True)
    subprocess.check_call(["git", "merge-base", "--is-ancestor", LOCK_COMMIT, "HEAD"], cwd=ROOT)
    if branch != REPLICATION_BRANCH:
        raise RuntimeError(f"Expected branch {REPLICATION_BRANCH}, got {branch}")
    if tag_target != LOCK_COMMIT:
        raise RuntimeError(f"Tag {LOCK_TAG} points to {tag_target}, expected {LOCK_COMMIT}")
    manifest_sha = sha256_file(MANIFEST_PATH)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Manifest hash mismatch: {manifest_sha} != {EXPECTED_MANIFEST_SHA256}")
    return {
        "head": head,
        "branch": branch,
        "lock_commit": LOCK_COMMIT,
        "lock_tag": LOCK_TAG,
        "lock_tag_target": tag_target,
        "manifest_sha256": manifest_sha,
    }


def checkpoint_path(seed: int) -> Path:
    return CHECKPOINT_DIR / f"{PRIMARY_MODEL}_seed_{seed}_best_val_dice.pt"


def checkpoint_hashes() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for seed in FAIR_SEEDS:
        path = checkpoint_path(seed)
        digest = sha256_file(path)
        expected = EXPECTED_CHECKPOINT_SHA256[seed]
        if digest != expected:
            raise RuntimeError(f"Checkpoint hash mismatch for seed {seed}: {digest} != {expected}")
        out[str(seed)] = {"path": str(path.relative_to(ROOT)), "sha256": digest}
    return out


def load_eligible_samples() -> list[SipakmedSample]:
    rows = read_csv(MANIFEST_PATH)
    eligible = [row for row in rows if row["technical_valid"].strip().upper() == "TRUE"]
    if len(rows) != 4049 or len(eligible) != 4000:
        raise RuntimeError(f"Locked manifest count mismatch: total={len(rows)} eligible={len(eligible)}")
    samples: list[SipakmedSample] = []
    for row in eligible:
        samples.append(
            SipakmedSample(
                sample_id=row["sample_id"],
                parent_source_id=row["parent_source_id"],
                diagnostic_class=row["diagnostic_class_if_available"],
                image_path=ROOT / Path(row["image_path"]),
                nucleus_contour_path=ROOT / Path(row["nucleus_contour_path"]),
                cytoplasm_contour_path=ROOT / Path(row["cytoplasm_contour_path"]),
                native_width=int(row["width"]),
                native_height=int(row["height"]),
            )
        )
    return samples


def load_model(seed: int, device: torch.device) -> UNet:
    path = checkpoint_path(seed)
    ckpt = torch.load(path, map_location=device)
    model = UNet(base_channels=32).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def metric_row(nucleus: np.ndarray, cytoplasm: np.ndarray, gt_nucleus: np.ndarray, gt_cytoplasm: np.ndarray) -> dict:
    gt_target = masks_to_target(gt_nucleus, gt_cytoplasm)
    pred_target = np.zeros(gt_target.shape, dtype=np.uint8)
    pred_target[np.logical_and(cytoplasm, ~nucleus)] = 1
    pred_target[nucleus] = 2
    pred_cell = np.logical_or(nucleus, cytoplasm)
    gt_cell = np.logical_or(gt_nucleus, gt_cytoplasm)

    gt_area = int(np.count_nonzero(gt_nucleus))
    pred_area = int(np.count_nonzero(nucleus))
    gt_perim = safe_perimeter(gt_nucleus)
    pred_perim = safe_perimeter(nucleus)
    gt_circ = compute_circularity(gt_nucleus)
    gt_nc = compute_nc_ratio(gt_nucleus, gt_cytoplasm)

    invalid_prediction = not valid_mask(nucleus, cytoplasm)
    try:
        pred_circ = compute_circularity(nucleus)
        circ_ae = abs(pred_circ - gt_circ)
        invalid_circularity = False
    except ValueError:
        pred_circ = float("nan")
        circ_ae = float("nan")
        invalid_circularity = True
    try:
        pred_nc = compute_nc_ratio(nucleus, cytoplasm)
        nc_ae = abs(pred_nc - gt_nc)
        invalid_nc = False
    except ValueError:
        pred_nc = float("nan")
        nc_ae = float("nan")
        invalid_nc = True

    return {
        "nucleus_dice": dice_score(nucleus, gt_nucleus),
        "cytoplasm_dice": dice_score(cytoplasm, gt_cytoplasm),
        "foreground_dice": foreground_dice(pred_target, gt_target),
        "cell_dice": dice_score(pred_cell, gt_cell),
        "gt_nucleus_area": gt_area,
        "pred_nucleus_area": pred_area,
        "area_ae": abs(pred_area - gt_area),
        "gt_perimeter": gt_perim,
        "pred_perimeter": pred_perim,
        "perimeter_ae": abs(pred_perim - gt_perim) if math.isfinite(pred_perim) and math.isfinite(gt_perim) else float("nan"),
        "gt_circularity": gt_circ,
        "pred_circularity": pred_circ,
        "circularity_ae": circ_ae,
        "gt_nc": gt_nc,
        "pred_nc": pred_nc,
        "nc_ae": nc_ae,
        "invalid_prediction": bool(invalid_prediction),
        "invalid_circularity": bool(invalid_circularity),
        "invalid_nc": bool(invalid_nc),
    }


def partial_paths(seed: int) -> dict[str, Path]:
    return {
        "processing": PARTIAL_DIR / f"seed_{seed}_processing.jsonl",
        "raw": PARTIAL_DIR / f"seed_{seed}_raw.jsonl",
        "specificity": PARTIAL_DIR / f"seed_{seed}_specificity.jsonl",
    }


def run_inference_for_seed(samples: list[SipakmedSample], seed: int, device: torch.device) -> None:
    sample_count = len(samples)
    paths = partial_paths(seed)
    completed = {row["sample_id"] for row in read_jsonl(paths["specificity"])}
    remaining = [sample for sample in samples if sample.sample_id not in completed]
    print(
        f"[external] seed {seed}: completed={len(completed)} remaining={len(remaining)}",
        flush=True,
    )
    if not remaining:
        return

    print(f"[external] loading baseline seed {seed}", flush=True)
    model = load_model(seed, device)
    seed_start = time.time()
    with torch.no_grad():
        for batch_start in range(0, len(remaining), INFERENCE_BATCH_SIZE):
            batch = remaining[batch_start : batch_start + INFERENCE_BATCH_SIZE]
            loaded = [load_sipakmed_sample(sample) for sample in batch]
            tensors = [
                preprocess_rgb_image(item.image, size=IMAGE_SIZE)
                for item in loaded
            ]
            logits = model(torch.stack(tensors, dim=0).to(device))
            pred_resized_batch = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            batch_processing_rows: list[dict] = []
            batch_raw_rows: list[dict] = []
            batch_specificity_rows: list[dict] = []

            for item, pred_resized in zip(loaded, pred_resized_batch):
                pred_target = restore_prediction_to_original_size(pred_resized, item.image.shape[:2], IMAGE_SIZE)
                raw_nucleus = pred_target == 2
                raw_cytoplasm = pred_target == 1
                closing_nucleus, closing_cytoplasm, closing_fallback = apply_closing(raw_nucleus, raw_cytoplasm)
                null_nucleus, null_cytoplasm, null_info = generate_dual_null(
                    raw_nucleus,
                    raw_cytoplasm,
                    closing_nucleus,
                    item.sample_id,
                    seed,
                )

                metrics_by_processing: dict[str, dict] = {}
                for processing, nucleus, cytoplasm, fallback in [
                    ("raw", raw_nucleus, raw_cytoplasm, False),
                    ("closing", closing_nucleus, closing_cytoplasm, closing_fallback),
                    ("dual_matched_null", null_nucleus, null_cytoplasm, bool(null_info["fallback_to_raw"])),
                ]:
                    m = metric_row(nucleus, cytoplasm, item.nucleus_mask, item.cytoplasm_only_mask)
                    metrics_by_processing[processing] = m
                    base = {
                        "sample_id": item.sample_id,
                        "parent_source_id": item.parent_source_id,
                        "diagnostic_class": item.diagnostic_class,
                        "model": PRIMARY_MODEL,
                        "seed": seed,
                        "checkpoint_rule": "best_val_dice",
                        "processing": processing,
                        "fallback_to_raw": fallback,
                        "native_width": item.native_width,
                        "native_height": item.native_height,
                    }
                    row = {**base, **m}
                    batch_processing_rows.append(row)
                    if processing == "raw":
                        batch_raw_rows.append(row)

                raw_m = metrics_by_processing["raw"]
                closing_m = metrics_by_processing["closing"]
                null_m = metrics_by_processing["dual_matched_null"]
                batch_specificity_rows.append({
                    "sample_id": item.sample_id,
                    "parent_source_id": item.parent_source_id,
                    "diagnostic_class": item.diagnostic_class,
                    "model": PRIMARY_MODEL,
                    "seed": seed,
                    "raw_circularity_ae": raw_m["circularity_ae"],
                    "closing_circularity_ae": closing_m["circularity_ae"],
                    "null_circularity_ae": null_m["circularity_ae"],
                    "closing_benefit": raw_m["circularity_ae"] - closing_m["circularity_ae"],
                    "null_benefit": raw_m["circularity_ae"] - null_m["circularity_ae"],
                    "closing_specific_residual": (raw_m["circularity_ae"] - closing_m["circularity_ae"])
                    - (raw_m["circularity_ae"] - null_m["circularity_ae"]),
                    "raw_nucleus_dice": raw_m["nucleus_dice"],
                    "closing_nucleus_dice": closing_m["nucleus_dice"],
                    "null_nucleus_dice": null_m["nucleus_dice"],
                    "raw_foreground_dice": raw_m["foreground_dice"],
                    "closing_foreground_dice": closing_m["foreground_dice"],
                    "null_foreground_dice": null_m["foreground_dice"],
                    "raw_area_ae": raw_m["area_ae"],
                    "closing_area_ae": closing_m["area_ae"],
                    "null_area_ae": null_m["area_ae"],
                    "raw_perimeter_ae": raw_m["perimeter_ae"],
                    "closing_perimeter_ae": closing_m["perimeter_ae"],
                    "null_perimeter_ae": null_m["perimeter_ae"],
                    "raw_nc_ae": raw_m["nc_ae"],
                    "closing_nc_ae": closing_m["nc_ae"],
                    "null_nc_ae": null_m["nc_ae"],
                    "closing_fallback_to_raw": bool(closing_fallback),
                    **null_info,
                })

            append_jsonl(paths["processing"], batch_processing_rows)
            append_jsonl(paths["raw"], batch_raw_rows)
            append_jsonl(paths["specificity"], batch_specificity_rows)

            done_now = batch_start + len(batch)
            done_total = len(completed) + done_now
            if done_total % 400 == 0 or done_total == sample_count:
                elapsed = time.time() - seed_start
                rate = done_now / max(elapsed, 1e-9)
                eta = (len(remaining) - done_now) / max(rate, 1e-9)
                print(
                    f"[external] seed {seed}: {done_total}/{sample_count} samples, "
                    f"elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m",
                    flush=True,
                )
    print(f"[external] completed seed {seed} in {(time.time() - seed_start) / 60:.1f}m", flush=True)


def run_inference(samples: list[SipakmedSample], device: torch.device) -> tuple[list[dict], list[dict], list[dict]]:
    for seed in FAIR_SEEDS:
        run_inference_for_seed(samples, seed, device)
    processing_rows = []
    raw_rows = []
    specificity_rows = []
    for seed in FAIR_SEEDS:
        paths = partial_paths(seed)
        processing_rows.extend(read_jsonl(paths["processing"]))
        raw_rows.extend(read_jsonl(paths["raw"]))
        specificity_rows.extend(read_jsonl(paths["specificity"]))
    processing_rows = dedupe_rows(processing_rows, ["sample_id", "seed", "processing"])
    raw_rows = dedupe_rows(raw_rows, ["sample_id", "seed"])
    specificity_rows = dedupe_rows(specificity_rows, ["sample_id", "seed"])
    expected = len(samples) * len(FAIR_SEEDS)
    if len(raw_rows) != expected or len(specificity_rows) != expected:
        raise RuntimeError(
            f"Incomplete external inference rows: raw={len(raw_rows)} specificity={len(specificity_rows)} expected={expected}"
        )
    return processing_rows, raw_rows, specificity_rows


def seed_summary(raw_rows: list[dict]) -> list[dict]:
    out = []
    for seed in FAIR_SEEDS:
        subset = [row for row in raw_rows if int(row["seed"]) == seed]
        out.append({
            "model": PRIMARY_MODEL,
            "seed": seed,
            "n_evaluated": len(subset),
            "nucleus_dice_mean": finite_mean(row["nucleus_dice"] for row in subset),
            "foreground_dice_mean": finite_mean(row["foreground_dice"] for row in subset),
            "circularity_ae_mean": finite_mean(row["circularity_ae"] for row in subset),
            "area_ae_mean": finite_mean(row["area_ae"] for row in subset),
            "perimeter_ae_mean": finite_mean(row["perimeter_ae"] for row in subset),
            "nc_ae_mean": finite_mean(row["nc_ae"] for row in subset),
            "invalid_prediction_rate": finite_mean(float(row["invalid_prediction"]) for row in subset),
            "invalid_circularity_rate": finite_mean(float(row["invalid_circularity"]) for row in subset),
            "invalid_nc_rate": finite_mean(float(row["invalid_nc"]) for row in subset),
        })
    return out


def cell_median_rows(rows: list[dict], metrics: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    out = []
    for sample_id, vals in sorted(grouped.items()):
        base = {
            "sample_id": sample_id,
            "parent_source_id": vals[0]["parent_source_id"],
            "diagnostic_class": vals[0]["diagnostic_class"],
            "model": PRIMARY_MODEL,
            "seed_aggregation": "median_across_20260803_20260804_20260805",
            "n_seed_rows": len(vals),
        }
        for metric in metrics:
            base[metric] = finite_median(float(row[metric]) for row in vals)
        out.append(base)
    return out


def average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        avg = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg
        start = end
    return ranks


def spearman_rho(x_values: Iterable[float], y_values: Iterable[float]) -> float:
    x = np.asarray(list(x_values), dtype=float)
    y = np.asarray(list(y_values), dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(keep)) < 3:
        return float("nan")
    xr = average_rank(x[keep])
    yr = average_rank(y[keep])
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def parent_index(rows: list[dict]) -> tuple[list[str], np.ndarray]:
    parents = sorted({row["parent_source_id"] for row in rows})
    parent_to_idx = {parent: idx for idx, parent in enumerate(parents)}
    row_parent_idx = np.asarray([parent_to_idx[row["parent_source_id"]] for row in rows], dtype=np.int32)
    return parents, row_parent_idx


def bootstrap_result(point: float, values: np.ndarray, n_clusters: int, n_cells: int) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "estimate": float(point),
        "ci_low": float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
        "ci_high": float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cluster_unit": "parent_source_id",
        "n_clusters": int(n_clusters),
        "n_cells": int(n_cells),
        "finite_bootstrap_replicates": int(finite.size),
    }


def parent_bootstrap_mean_metric(rows: list[dict], metric: str) -> dict:
    parents, row_parent_idx = parent_index(rows)
    values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    finite = np.isfinite(values)
    point = finite_mean(values)
    cluster_sums = np.bincount(
        row_parent_idx[finite],
        weights=values[finite],
        minlength=len(parents),
    )
    cluster_counts = np.bincount(row_parent_idx[finite], minlength=len(parents)).astype(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(parents), size=(BOOTSTRAP_REPEATS, len(parents)), dtype=np.int32)
    sums = cluster_sums[sampled].sum(axis=1)
    counts = cluster_counts[sampled].sum(axis=1)
    boot = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)
    return bootstrap_result(point, boot, len(parents), len(rows))


def rank_group_metadata(values: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precompute tie groups in sorted order."""
    sorted_values = values[order]
    if sorted_values.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    lengths = np.diff(np.r_[starts, sorted_values.size])
    group_ids = np.repeat(np.arange(starts.size, dtype=np.int64), lengths)
    return starts.astype(np.int64), group_ids.astype(np.int64)


def weighted_rank_from_sorted_weights(
    weights: np.ndarray,
    order: np.ndarray,
    starts: np.ndarray,
    group_ids: np.ndarray,
) -> np.ndarray:
    """Tie-aware average ranks for a weighted bootstrap sample."""
    sorted_weights = weights[order]
    group_weights = np.add.reduceat(sorted_weights, starts)
    cumulative = np.cumsum(group_weights)
    average_ranks = cumulative - (group_weights - 1.0) / 2.0
    sorted_ranks = average_ranks[group_ids]
    ranks = np.empty_like(sorted_ranks, dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    keep = weights > 0
    if int(np.count_nonzero(keep)) < 3:
        return float("nan")
    wk = weights[keep]
    xk = x[keep]
    yk = y[keep]
    total = float(np.sum(wk))
    if total <= 0:
        return float("nan")
    x_mean = float(np.sum(wk * xk) / total)
    y_mean = float(np.sum(wk * yk) / total)
    xd = xk - x_mean
    yd = yk - y_mean
    cov = float(np.sum(wk * xd * yd))
    vx = float(np.sum(wk * xd * xd))
    vy = float(np.sum(wk * yd * yd))
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(cov / math.sqrt(vx * vy))


def parent_bootstrap_spearman_metric(rows: list[dict], x_metric: str, y_metric: str) -> dict:
    parents, row_parent_idx_all = parent_index(rows)
    x_all = np.asarray([float(row[x_metric]) for row in rows], dtype=float)
    y_all = np.asarray([float(row[y_metric]) for row in rows], dtype=float)
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    x = x_all[finite]
    y = y_all[finite]
    row_parent_idx = row_parent_idx_all[finite]
    point = spearman_rho(x, y)
    order_x = np.argsort(x, kind="mergesort")
    order_y = np.argsort(y, kind="mergesort")
    starts_x, group_ids_x = rank_group_metadata(x, order_x)
    starts_y, group_ids_y = rank_group_metadata(y, order_y)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(parents), size=(BOOTSTRAP_REPEATS, len(parents)), dtype=np.int32)
    boot = np.empty(BOOTSTRAP_REPEATS, dtype=float)
    for i, sample in enumerate(sampled):
        cluster_weights = np.bincount(sample, minlength=len(parents)).astype(float)
        row_weights = cluster_weights[row_parent_idx]
        rank_x = weighted_rank_from_sorted_weights(row_weights, order_x, starts_x, group_ids_x)
        rank_y = weighted_rank_from_sorted_weights(row_weights, order_y, starts_y, group_ids_y)
        boot[i] = weighted_corr(rank_x, rank_y, row_weights)
    return bootstrap_result(point, boot, len(parents), int(np.count_nonzero(finite)))


def parent_bootstrap(
    rows: list[dict],
    estimator: Callable[[list[dict]], float],
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_parent[row["parent_source_id"]].append(row)
    parents = sorted(by_parent)
    point = estimator(rows)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for i in range(repeats):
        sampled = rng.choice(parents, size=len(parents), replace=True)
        resampled = [row for parent in sampled for row in by_parent[parent]]
        values[i] = estimator(resampled)
    finite = values[np.isfinite(values)]
    return {
        "estimate": float(point),
        "ci_low": float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
        "ci_high": float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
        "bootstrap_repeats": repeats,
        "bootstrap_seed": seed,
        "cluster_unit": "parent_source_id",
        "n_clusters": len(parents),
        "n_cells": len(rows),
        "finite_bootstrap_replicates": int(finite.size),
    }


def mean_metric(metric: str) -> Callable[[list[dict]], float]:
    return lambda rows: finite_mean(float(row[metric]) for row in rows)


def spearman_metric(x_metric: str, y_metric: str) -> Callable[[list[dict]], float]:
    return lambda rows: spearman_rho((float(row[x_metric]) for row in rows), (float(row[y_metric]) for row in rows))


def build_bootstrap(raw_cell_rows: list[dict], specificity_cell_rows: list[dict]) -> tuple[list[dict], dict]:
    specs = [
        ("primary", "mean_nucleus_dice", "mean", "nucleus_dice", None),
        ("primary", "mean_circularity_ae", "mean", "circularity_ae", None),
        ("primary", "spearman_nucleus_dice_vs_circularity_ae", "spearman", "nucleus_dice", "circularity_ae"),
        ("secondary", "mean_area_ae", "mean", "area_ae", None),
        ("secondary", "mean_perimeter_ae", "mean", "perimeter_ae", None),
        ("secondary", "mean_nc_ae", "mean", "nc_ae", None),
        ("secondary", "mean_foreground_dice", "mean", "foreground_dice", None),
        ("secondary", "spearman_foreground_dice_vs_area_ae", "spearman", "foreground_dice", "area_ae"),
        ("secondary", "spearman_foreground_dice_vs_perimeter_ae", "spearman", "foreground_dice", "perimeter_ae"),
        ("secondary", "spearman_foreground_dice_vs_nc_ae", "spearman", "foreground_dice", "nc_ae"),
        ("secondary", "spearman_nucleus_dice_vs_area_ae", "spearman", "nucleus_dice", "area_ae"),
        ("secondary", "spearman_nucleus_dice_vs_perimeter_ae", "spearman", "nucleus_dice", "perimeter_ae"),
    ]
    out_rows = []
    nested: dict[str, dict] = defaultdict(dict)
    for section, metric_name, kind, metric_a, metric_b in specs:
        print(f"[external] bootstrap {section}/{metric_name}", flush=True)
        if kind == "mean":
            result = parent_bootstrap_mean_metric(raw_cell_rows, metric_a)
        else:
            assert metric_b is not None
            result = parent_bootstrap_spearman_metric(raw_cell_rows, metric_a, metric_b)
        row = {"section": section, "metric": metric_name, **result}
        out_rows.append(row)
        nested[section][metric_name] = result

    specificity_specs = [
        ("mean_raw_circularity_ae", "raw_circularity_ae"),
        ("mean_closing_circularity_ae", "closing_circularity_ae"),
        ("mean_null_circularity_ae", "null_circularity_ae"),
        ("mean_closing_benefit", "closing_benefit"),
        ("mean_null_benefit", "null_benefit"),
        ("mean_closing_specific_residual", "closing_specific_residual"),
    ]
    for metric_name, source_metric in specificity_specs:
        print(f"[external] bootstrap external_specificity/{metric_name}", flush=True)
        result = parent_bootstrap_mean_metric(specificity_cell_rows, source_metric)
        row = {"section": "external_specificity", "metric": metric_name, **result}
        out_rows.append(row)
        nested["external_specificity"][metric_name] = result
    return out_rows, nested


def build_aggregate_summary(
    raw_cell_rows: list[dict],
    specificity_cell_rows: list[dict],
    bootstrap_nested: dict,
    seed_summary_rows: list[dict],
    provenance: dict,
    runtime_failures: list[dict],
) -> dict:
    invalid = {
        "raw_invalid_prediction_rate": finite_mean(float(row["invalid_prediction"]) for row in raw_cell_rows),
        "raw_invalid_circularity_rate": finite_mean(float(row["invalid_circularity"]) for row in raw_cell_rows),
        "raw_invalid_nc_rate": finite_mean(float(row["invalid_nc"]) for row in raw_cell_rows),
    }
    return {
        "status": "SIPAKMED_EXTERNAL_REPLICATION_COMPLETE",
        "protocol_lock_commit": LOCK_COMMIT,
        "n_locked_eligible": 4000,
        "n_evaluated": len(raw_cell_rows),
        "all_4000_locked_eligible_samples_evaluated": len(raw_cell_rows) == 4000 and not runtime_failures,
        "seed_family_aggregation": "cell-level median across all three frozen baseline seeds",
        "primary_external_replication": bootstrap_nested["primary"],
        "secondary_morphometric_analyses": bootstrap_nested["secondary"],
        "external_closing_vs_matched_null_specificity": bootstrap_nested["external_specificity"],
        "invalid_rates": invalid,
        "seed_summary": seed_summary_rows,
        "runtime_failures": runtime_failures,
        "provenance": provenance,
    }


def write_figures(raw_cell_rows: list[dict], specificity_cell_rows: list[dict]) -> list[str]:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    x = [float(row["nucleus_dice"]) for row in raw_cell_rows]
    y = [float(row["circularity_ae"]) for row in raw_cell_rows]
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=8, alpha=0.35, linewidths=0)
    plt.xlabel("Nucleus Dice")
    plt.ylabel("Circularity AE")
    plt.title("SIPaKMeD primary association")
    plt.tight_layout()
    path = fig_dir / "primary_dice_vs_circularity_ae.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path.relative_to(ROOT)))

    data = [
        finite_values(float(row[key]) for row in specificity_cell_rows)
        for key in ["raw_circularity_ae", "closing_circularity_ae", "null_circularity_ae"]
    ]
    plt.figure(figsize=(6, 4))
    plt.boxplot(data, tick_labels=["Raw", "Closing", "Matched null"], showfliers=False)
    plt.ylabel("Circularity AE")
    plt.title("Closing specificity replication")
    plt.tight_layout()
    path = fig_dir / "closing_specificity_circularity_ae.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path.relative_to(ROOT)))

    labels = ["Area AE", "Perimeter AE", "N/C AE"]
    means = [
        finite_mean(float(row["area_ae"]) for row in raw_cell_rows),
        finite_mean(float(row["perimeter_ae"]) for row in raw_cell_rows),
        finite_mean(float(row["nc_ae"]) for row in raw_cell_rows),
    ]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, means)
    plt.ylabel("Mean absolute error")
    plt.title("Secondary morphometric errors")
    plt.tight_layout()
    path = fig_dir / "secondary_morphometry_mean_ae.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path.relative_to(ROOT)))
    return paths


def provenance_payload(started_at: str, elapsed_seconds: float, device: torch.device, figures: list[str]) -> dict:
    code_hashes = {
        rel: sha256_file(ROOT / rel)
        for rel in CODE_HASH_PATHS
        if (ROOT / rel).exists()
    }
    return {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "git": assert_preflight_state(),
        "checkpoints": checkpoint_hashes(),
        "code_hashes": code_hashes,
        "output_figures": figures,
        "protocol_unchanged": True,
        "training_run": False,
        "fine_tuning_run": False,
        "checkpoint_selection_from_sipakmed": False,
        "postprocessing_parameter_sweep": False,
    }


def fmt_ci(result: dict) -> str:
    return f"{result['estimate']:.6g} [{result['ci_low']:.6g}, {result['ci_high']:.6g}]"


def write_report(summary: dict, bootstrap_rows: list[dict], result_paths: list[str]) -> None:
    primary = summary["primary_external_replication"]
    secondary = summary["secondary_morphometric_analyses"]
    specificity = summary["external_closing_vs_matched_null_specificity"]
    lines = [
        "# SIPaKMeD External Replication Report",
        "",
        f"Protocol lock commit: `{LOCK_COMMIT}`",
        f"Run branch: `{REPLICATION_BRANCH}`",
        f"Locked eligible samples evaluated: `{summary['n_evaluated']}` / `4000`",
        f"All 4000 locked eligible samples evaluated: `{summary['all_4000_locked_eligible_samples_evaluated']}`",
        "",
        "This report follows the frozen outcome-blind protocol. No training, fine-tuning, checkpoint selection, preprocessing tuning, sample reselection, or post-processing parameter sweep was performed.",
        "",
        "## 1. Primary External Replication",
        "",
        "| Outcome | Estimate [parent-source cluster 95% CI] |",
        "| --- | ---: |",
        f"| Mean nucleus Dice | `{fmt_ci(primary['mean_nucleus_dice'])}` |",
        f"| Mean nucleus circularity AE | `{fmt_ci(primary['mean_circularity_ae'])}` |",
        f"| Spearman: nucleus Dice vs circularity AE | `{fmt_ci(primary['spearman_nucleus_dice_vs_circularity_ae'])}` |",
        "",
        "## 2. Secondary Morphometric Analyses",
        "",
        "| Outcome | Estimate [parent-source cluster 95% CI] |",
        "| --- | ---: |",
        f"| Mean nucleus area AE | `{fmt_ci(secondary['mean_area_ae'])}` |",
        f"| Mean nucleus perimeter AE | `{fmt_ci(secondary['mean_perimeter_ae'])}` |",
        f"| Mean N/C AE | `{fmt_ci(secondary['mean_nc_ae'])}` |",
        f"| Mean foreground Dice | `{fmt_ci(secondary['mean_foreground_dice'])}` |",
        f"| Spearman: foreground Dice vs area AE | `{fmt_ci(secondary['spearman_foreground_dice_vs_area_ae'])}` |",
        f"| Spearman: foreground Dice vs perimeter AE | `{fmt_ci(secondary['spearman_foreground_dice_vs_perimeter_ae'])}` |",
        f"| Spearman: foreground Dice vs N/C AE | `{fmt_ci(secondary['spearman_foreground_dice_vs_nc_ae'])}` |",
        f"| Spearman: nucleus Dice vs area AE | `{fmt_ci(secondary['spearman_nucleus_dice_vs_area_ae'])}` |",
        f"| Spearman: nucleus Dice vs perimeter AE | `{fmt_ci(secondary['spearman_nucleus_dice_vs_perimeter_ae'])}` |",
        "",
        "## 3. External Closing-vs-Matched-Null Specificity Replication",
        "",
        "| Outcome | Estimate [parent-source cluster 95% CI] |",
        "| --- | ---: |",
        f"| Mean raw circularity AE | `{fmt_ci(specificity['mean_raw_circularity_ae'])}` |",
        f"| Mean fixed-closing circularity AE | `{fmt_ci(specificity['mean_closing_circularity_ae'])}` |",
        f"| Mean matched-null circularity AE | `{fmt_ci(specificity['mean_null_circularity_ae'])}` |",
        f"| Mean closing benefit | `{fmt_ci(specificity['mean_closing_benefit'])}` |",
        f"| Mean matched-null benefit | `{fmt_ci(specificity['mean_null_benefit'])}` |",
        f"| Mean closing-specific residual | `{fmt_ci(specificity['mean_closing_specific_residual'])}` |",
        "",
        "## Output Artifacts",
        "",
    ]
    for rel in result_paths:
        lines.append(f"- `{rel}`")
    lines.extend([
        "",
        "## Runtime/Data Failures",
        "",
        f"Runtime/data failures: `{len(summary['runtime_failures'])}`",
        "",
        "## Outcome Discipline",
        "",
        "- Protocol changed after seeing SIPaKMeD outcomes: NO",
        "- Eligibility changed after seeing SIPaKMeD outcomes: NO",
        "- Checkpoints selected based on SIPaKMeD performance: NO",
        "- Post-processing parameters swept on SIPaKMeD: NO",
        "",
        "## Bootstrap Records",
        "",
        f"Bootstrap rows written: `{len(bootstrap_rows)}`",
        f"Bootstrap repeats: `{BOOTSTRAP_REPEATS}`",
        f"Bootstrap seed: `{BOOTSTRAP_SEED}`",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    start = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = assert_preflight_state()
    checkpoints = checkpoint_hashes()
    samples = load_eligible_samples()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"state": state, "n_eligible": len(samples), "device": str(device)}, indent=2), flush=True)

    runtime_failures: list[dict] = []
    processing_rows, raw_rows, specificity_rows = run_inference(samples, device)
    raw_metrics = [
        "nucleus_dice",
        "cytoplasm_dice",
        "foreground_dice",
        "cell_dice",
        "area_ae",
        "perimeter_ae",
        "circularity_ae",
        "nc_ae",
        "invalid_prediction",
        "invalid_circularity",
        "invalid_nc",
    ]
    specificity_metrics = [
        "raw_circularity_ae",
        "closing_circularity_ae",
        "null_circularity_ae",
        "closing_benefit",
        "null_benefit",
        "closing_specific_residual",
        "raw_nucleus_dice",
        "closing_nucleus_dice",
        "null_nucleus_dice",
        "raw_foreground_dice",
        "closing_foreground_dice",
        "null_foreground_dice",
        "raw_area_ae",
        "closing_area_ae",
        "null_area_ae",
        "raw_perimeter_ae",
        "closing_perimeter_ae",
        "null_perimeter_ae",
        "raw_nc_ae",
        "closing_nc_ae",
        "null_nc_ae",
    ]
    raw_cell_rows = cell_median_rows(raw_rows, raw_metrics)
    specificity_cell_rows = cell_median_rows(specificity_rows, specificity_metrics)
    seed_summary_rows = seed_summary(raw_rows)
    bootstrap_rows, bootstrap_nested = build_bootstrap(raw_cell_rows, specificity_cell_rows)
    figures = write_figures(raw_cell_rows, specificity_cell_rows)
    provenance = provenance_payload(started_at, time.time() - start, device, figures)
    provenance["checkpoints"] = checkpoints
    summary = build_aggregate_summary(
        raw_cell_rows,
        specificity_cell_rows,
        bootstrap_nested,
        seed_summary_rows,
        provenance,
        runtime_failures,
    )

    output_paths = [
        OUT_DIR / "cell_seed_processing_metrics.csv",
        OUT_DIR / "cell_seed_raw_metrics.csv",
        OUT_DIR / "cell_seed_specificity.csv",
        OUT_DIR / "cell_median_raw_metrics.csv",
        OUT_DIR / "cell_median_specificity.csv",
        OUT_DIR / "seed_summary.csv",
        OUT_DIR / "bootstrap_results.csv",
        OUT_DIR / "aggregate_summary.json",
        OUT_DIR / "runtime_environment.json",
        OUT_DIR / "checkpoint_hashes.json",
        OUT_DIR / "provenance_hashes.json",
    ]
    write_csv(output_paths[0], processing_rows)
    write_csv(output_paths[1], raw_rows)
    write_csv(output_paths[2], specificity_rows)
    write_csv(output_paths[3], raw_cell_rows)
    write_csv(output_paths[4], specificity_cell_rows)
    write_csv(output_paths[5], seed_summary_rows)
    write_csv(output_paths[6], bootstrap_rows)
    output_paths[7].write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    output_paths[8].write_text(json.dumps(provenance, indent=2, allow_nan=True), encoding="utf-8")
    output_paths[9].write_text(json.dumps(checkpoints, indent=2), encoding="utf-8")
    output_paths[10].write_text(json.dumps(provenance["code_hashes"], indent=2), encoding="utf-8")
    report_artifacts = [str(path.relative_to(ROOT)) for path in output_paths] + figures
    write_report(summary, bootstrap_rows, report_artifacts)
    if PARTIAL_DIR.exists():
        shutil.rmtree(PARTIAL_DIR)
    print(json.dumps({
        "status": "SIPAKMED_EXTERNAL_REPLICATION_COMPLETE",
        "n_evaluated": summary["n_evaluated"],
        "all_4000": summary["all_4000_locked_eligible_samples_evaluated"],
        "primary": summary["primary_external_replication"],
        "specificity": summary["external_closing_vs_matched_null_specificity"],
    }, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    run()
