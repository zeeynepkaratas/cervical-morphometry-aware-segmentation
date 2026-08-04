"""Core final analysis without retraining.

This script intentionally produces only the priority outputs:
loss-scale diagnostics, clean per-cell diagnostics, deterministic blur/noise
inference, basic summaries, frozen-clean-calibration conformal results, and a
short interim scientific decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_prep.load_herlev import list_herlev_images, load_image_and_masks
from src.degradation.apply_degradations import apply_gaussian_blur, apply_gaussian_noise
from src.measurements.morphometry import compute_circularity, compute_nc_ratio, relative_error
from src.pilot.config_io import load_config, resolve_herlev_data_dir
from src.pilot.fair_repeat import FAIR_SEEDS, summarize_rows, write_csv
from src.pilot.metrics import dice_score, foreground_dice, masks_to_target, restore_prediction_to_original_size
from src.segmentation.train_unet import preprocess_rgb_image
from src.segmentation.unet_model import UNet

MODELS = {"baseline": 0.0, "nc_aware_lambda_0p10": 0.10}
BLUR_LEVELS = [1.0, 2.0, 3.0]
NOISE_LEVELS = [10.0, 20.0, 30.0]
TARGET_COVERAGE = 0.90
BOOTSTRAP_SEED = 20260804
BOOTSTRAP_REPEATS = 5000


def stable_noise_seed(cell_id: str, degradation: str, severity: float) -> int:
    """Return deterministic noise seed independent of model/seed."""
    key = f"{cell_id}|{degradation}|{float(severity):g}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little", signed=False)


def conditions() -> list[tuple[str, str]]:
    """Locked clean/blur/noise conditions."""
    rows = [("clean", "clean")]
    rows += [("gaussian_blur", f"{level:g}") for level in BLUR_LEVELS]
    rows += [("gaussian_noise", f"{level:g}") for level in NOISE_LEVELS]
    return rows


def apply_condition(image: np.ndarray, cell_id: str, degradation: str, severity: str) -> np.ndarray:
    """Apply a locked condition to an RGB image; masks remain untouched."""
    if degradation == "clean":
        return image
    level = float(severity)
    if degradation == "gaussian_blur":
        return apply_gaussian_blur(image, level)
    if degradation == "gaussian_noise":
        return apply_gaussian_noise(image, level, seed=stable_noise_seed(cell_id, degradation, level))
    raise ValueError(f"Unexpected degradation: {degradation}")


def load_checkpoint(config: dict, seed: int, model_name: str, device: torch.device) -> tuple[UNet, dict]:
    """Load the best-val-Dice checkpoint only."""
    path = ROOT_DIR / "results" / "pilot" / "fair_repeat" / "checkpoints" / f"{model_name}_seed_{seed}_best_val_dice.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing best_val_dice checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    model = UNet(base_channels=int(config["training"]["base_channels"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint | {"checkpoint_path": str(path)}


def pred_metrics(pred_target: np.ndarray, gt_nucleus: np.ndarray, gt_cytoplasm: np.ndarray) -> dict:
    """Compute segmentation and morphometry metrics at original size."""
    gt_target = masks_to_target(gt_nucleus, gt_cytoplasm)
    pred_nucleus = pred_target == 2
    pred_cytoplasm = pred_target == 1
    gt_nc = compute_nc_ratio(gt_nucleus, gt_cytoplasm)
    gt_circ = compute_circularity(gt_nucleus)
    row = {
        "foreground_dice": foreground_dice(pred_target, gt_target),
        "nucleus_dice": dice_score(pred_nucleus, gt_nucleus),
        "cytoplasm_dice": dice_score(pred_cytoplasm, gt_cytoplasm),
        "true_nc": gt_nc,
        "true_circularity": gt_circ,
        "nucleus_area_absolute_error": abs(int(pred_nucleus.sum()) - int(gt_nucleus.sum())),
        "cytoplasm_area_absolute_error": abs(int(pred_cytoplasm.sum()) - int(gt_cytoplasm.sum())),
        "true_nucleus_area": int(gt_nucleus.sum()),
        "predicted_nucleus_area": int(pred_nucleus.sum()),
        "true_cytoplasm_area": int(gt_cytoplasm.sum()),
        "predicted_cytoplasm_area": int(pred_cytoplasm.sum()),
    }
    try:
        pred_nc = compute_nc_ratio(pred_nucleus, pred_cytoplasm)
        row.update(predicted_nc=pred_nc, nc_absolute_error=abs(pred_nc - gt_nc), nc_relative_error=relative_error(pred_nc, gt_nc), invalid_nc=False)
    except ValueError:
        row.update(predicted_nc=float("nan"), nc_absolute_error=float("inf"), nc_relative_error=float("inf"), invalid_nc=True)
    try:
        pred_circ = compute_circularity(pred_nucleus)
        row.update(predicted_circularity=pred_circ, circularity_absolute_error=abs(pred_circ - gt_circ), invalid_circularity=False)
    except ValueError:
        row.update(predicted_circularity=float("nan"), circularity_absolute_error=float("inf"), invalid_circularity=True)
    return row


def infer_rows(config: dict, data_dir: Path, split_name: str, cell_ids: list[str]) -> list[dict]:
    """Run deterministic inference for one split over all checkpoints/conditions."""
    device = torch.device(config["training"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    image_size = tuple(config["image_size"])
    image_by_id = {path.stem: path for path in list_herlev_images(data_dir)}
    rows = []
    for seed in FAIR_SEEDS:
        for model_name in MODELS:
            model, checkpoint = load_checkpoint(config, seed, model_name, device)
            for cell_id in cell_ids:
                image_path = image_by_id[cell_id]
                image, gt_nucleus, gt_cytoplasm = load_image_and_masks(image_path)
                for degradation, severity in conditions():
                    variant = apply_condition(image, cell_id, degradation, severity)
                    with torch.no_grad():
                        logits = model(preprocess_rgb_image(variant, size=image_size).unsqueeze(0).to(device))
                        pred_resized = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                    pred_target = restore_prediction_to_original_size(pred_resized, image.shape[:2], image_size)
                    row = {
                        "split": split_name,
                        "cell_id": cell_id,
                        "class": image_path.parent.name,
                        "seed": seed,
                        "model": model_name,
                        "lambda_nc": MODELS[model_name],
                        "degradation": degradation,
                        "severity": severity,
                        "best_epoch": int(checkpoint["epoch"]),
                        "checkpoint_rule": "best_val_dice",
                    }
                    row.update(pred_metrics(pred_target, gt_nucleus, gt_cytoplasm))
                    rows.append(row)
    return rows


def loss_scale(out_dir: Path) -> None:
    """Extract loss scale diagnostics from existing fair-repeat histories."""
    rows = []
    summaries = []
    history_dir = ROOT_DIR / "results" / "pilot" / "fair_repeat"
    for path in sorted(history_dir.glob("training_history_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = payload.get("history", [])
        fractions = []
        for row in history:
            seg = float(row["train_segmentation_loss"])
            raw = float(row["train_raw_nc_loss"])
            lam = float(row["lambda_nc"])
            weighted = lam * raw
            total = seg + weighted
            fraction = weighted / total if total else float("nan")
            fractions.append(fraction)
            rows.append(
                {
                    "seed": row["seed"],
                    "model": row["model"],
                    "epoch": row["epoch"],
                    "segmentation_loss": seg,
                    "raw_nc_loss": raw,
                    "lambda_nc": lam,
                    "weighted_nc_loss": weighted,
                    "total_loss": total,
                    "weighted_nc_fraction": fraction,
                }
            )
        finite = np.asarray([x for x in fractions if np.isfinite(x)], dtype=float)
        summaries.append(
            {
                "seed": payload["seed"],
                "model": payload["run_id"].replace(f"_seed_{payload['seed']}", ""),
                "first_logged_epoch_fraction": float(finite[0]) if finite.size else float("nan"),
                "median_epoch_fraction": float(np.median(finite)) if finite.size else float("nan"),
                "last_logged_epoch_fraction": float(finite[-1]) if finite.size else float("nan"),
                "minimum_fraction": float(np.min(finite)) if finite.size else float("nan"),
                "maximum_fraction": float(np.max(finite)) if finite.size else float("nan"),
                "mean_fraction": float(np.mean(finite)) if finite.size else float("nan"),
                "missing_epoch_count": max(0, 25 - len(history)),
            }
        )
    write_csv(out_dir / "loss_scale_diagnostic.csv", rows)
    write_csv(out_dir / "loss_scale_summary.csv", summaries)


def clean_diagnostics(out_dir: Path) -> None:
    """Create clean diagnostics from existing fair-repeat validation rows."""
    source = ROOT_DIR / "results" / "pilot" / "fair_repeat" / "per_cell_predictions.csv"
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    numeric = [
        "foreground_dice",
        "nucleus_dice",
        "cytoplasm_dice",
        "nc_absolute_error",
        "circularity_absolute_error",
        "nucleus_area_absolute_error",
        "cytoplasm_area_absolute_error",
    ]
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
        row["seed"] = int(row["seed"])
    dice_q3 = float(np.percentile([r["foreground_dice"] for r in rows], 75))
    nc_q3 = float(np.percentile([r["nc_absolute_error"] for r in rows if np.isfinite(r["nc_absolute_error"])], 75))
    circ_q3 = float(np.percentile([r["circularity_absolute_error"] for r in rows if np.isfinite(r["circularity_absolute_error"])], 75))
    diag = []
    for row in rows:
        out = dict(row)
        out["high_dice"] = row["foreground_dice"] >= dice_q3
        out["high_nc_error"] = row["nc_absolute_error"] >= nc_q3
        out["high_circularity_error"] = row["circularity_absolute_error"] >= circ_q3
        diag.append(out)
    write_csv(out_dir / "clean_per_cell_diagnostics.csv", diag)
    extremes = []
    paired_path = ROOT_DIR / "results" / "pilot" / "fair_repeat" / "paired_cell_differences.csv"
    paired = list(csv.DictReader(paired_path.open(newline="", encoding="utf-8")))
    for row in paired:
        for key in row:
            if key.startswith("delta_") or key.endswith("_error") or key.endswith("_dice"):
                try:
                    row[key] = float(row[key])
                except ValueError:
                    pass
    for label, key, reverse in [
        ("top_nc_improved", "delta_nc_error", False),
        ("top_nc_worsened", "delta_nc_error", True),
        ("top_circularity_improved", "delta_circularity_error", False),
        ("top_circularity_worsened", "delta_circularity_error", True),
    ]:
        for rank, row in enumerate(sorted(paired, key=lambda r: r[key], reverse=reverse)[:20], start=1):
            extremes.append({"case_type": label, "rank": rank, **row})
    high_dice_nc = [r for r in diag if r["high_dice"] and r["high_nc_error"]]
    high_dice_circ = [r for r in diag if r["high_dice"] and r["high_circularity_error"]]
    for label, subset in [("high_dice_high_nc_error", high_dice_nc), ("high_dice_high_circularity_error", high_dice_circ)]:
        for rank, row in enumerate(sorted(subset, key=lambda r: (r["nc_absolute_error"], r["circularity_absolute_error"]), reverse=True)[:50], start=1):
            extremes.append({"case_type": label, "rank": rank, **row})
    write_csv(out_dir / "clean_extreme_cases.csv", extremes)
    summarize_clean(diag, out_dir)


def summarize_clean(rows: list[dict], out_dir: Path) -> None:
    """Write clean seed/class summaries."""
    by_seed = []
    for (seed, model), members in group(rows, ["seed", "model"]).items():
        by_seed.append({"seed": seed, "model": model, **summary_metrics(members)})
    by_class = []
    for (cls, model), members in group(rows, ["class", "model"]).items():
        by_class.append({"class": cls, "model": model, "n": len(members), **summary_metrics(members)})
    write_csv(out_dir / "clean_metrics_by_seed.csv", by_seed)
    write_csv(out_dir / "clean_metrics_by_class.csv", by_class)


def group(rows: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row)
    return out


def summary_metrics(rows: list[dict]) -> dict:
    """Basic finite metric summary."""
    return {
        "foreground_dice": finite_mean([r["foreground_dice"] for r in rows]),
        "nucleus_dice": finite_mean([r["nucleus_dice"] for r in rows]),
        "cytoplasm_dice": finite_mean([r["cytoplasm_dice"] for r in rows]),
        "mean_nc_absolute_error": finite_mean([r["nc_absolute_error"] for r in rows]),
        "median_nc_absolute_error": finite_median([r["nc_absolute_error"] for r in rows]),
        "mean_circularity_absolute_error": finite_mean([r["circularity_absolute_error"] for r in rows]),
        "invalid_nc_rate": float(np.mean([str(r["invalid_nc"]) == "True" for r in rows])),
        "invalid_circularity_rate": float(np.mean([str(r["invalid_circularity"]) == "True" for r in rows])),
    }


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def leakage_audit(out_dir: Path, split: dict) -> None:
    """Audit split separation."""
    cal = set(split["calibration"])
    test = set(split["test"])
    pilot = set(json.loads((ROOT_DIR / "data" / "splits" / "pilot_train_validation_split.json").read_text())["validation_ids"])
    rows = [
        {"check": "calibration_test_overlap", "count": len(cal & test), "passed": len(cal & test) == 0},
        {"check": "pilot_validation_calibration_overlap", "count": len(pilot & cal), "passed": len(pilot & cal) == 0},
        {"check": "pilot_validation_test_overlap", "count": len(pilot & test), "passed": len(pilot & test) == 0},
        {"check": "calibration_count", "count": len(cal), "passed": len(cal) == 183},
        {"check": "test_count", "count": len(test), "passed": len(test) == 184},
    ]
    write_csv(out_dir / "data_leakage_audit.csv", rows)


def paired_deltas(rows: list[dict]) -> list[dict]:
    """Pair degraded rows against clean within cell/seed/model/split."""
    clean = {(r["split"], r["cell_id"], r["seed"], r["model"]): r for r in rows if r["degradation"] == "clean"}
    deltas = []
    for row in rows:
        if row["degradation"] == "clean":
            continue
        base = clean[(row["split"], row["cell_id"], row["seed"], row["model"])]
        out = {k: row[k] for k in ["split", "cell_id", "class", "seed", "model", "degradation", "severity"]}
        for metric in ["foreground_dice", "nucleus_dice", "cytoplasm_dice", "nc_absolute_error", "circularity_absolute_error", "nucleus_area_absolute_error", "cytoplasm_area_absolute_error"]:
            out[f"clean_{metric}"] = base[metric]
            out[f"degraded_{metric}"] = row[metric]
            out[f"delta_{metric}"] = row[metric] - base[metric]
        deltas.append(out)
    return deltas


def conformal(out_dir: Path, rows: list[dict]) -> None:
    """Frozen clean-calibration split conformal for N/C and circularity."""
    results = []
    per_cell = []
    for seed in FAIR_SEEDS:
        for model in MODELS:
            for target, pred_key, true_key, err_key, invalid_key in [
                ("nc", "predicted_nc", "true_nc", "nc_absolute_error", "invalid_nc"),
                ("circularity", "predicted_circularity", "true_circularity", "circularity_absolute_error", "invalid_circularity"),
            ]:
                cal_clean = [r for r in rows if r["split"] == "calibration" and r["seed"] == seed and r["model"] == model and r["degradation"] == "clean"]
                scores = np.asarray([r[err_key] for r in cal_clean if np.isfinite(r[err_key])], dtype=float)
                q_hat = conformal_quantile(scores, TARGET_COVERAGE)
                for degradation, severity in conditions():
                    test_rows = [r for r in rows if r["split"] == "test" and r["seed"] == seed and r["model"] == model and r["degradation"] == degradation and r["severity"] == severity]
                    covered = []
                    failure_covered = []
                    widths = []
                    invalids = []
                    for r in test_rows:
                        pred = r[pred_key]
                        true = r[true_key]
                        valid = bool(np.isfinite(pred) and not r[invalid_key])
                        lower = pred - q_hat if valid else float("nan")
                        upper = pred + q_hat if valid else float("nan")
                        is_covered = bool(valid and lower <= true <= upper)
                        covered.append(is_covered if valid else np.nan)
                        failure_covered.append(is_covered)
                        widths.append(2 * q_hat if valid else np.nan)
                        invalids.append(not valid)
                        per_cell.append(
                            {
                                "cell_id": r["cell_id"],
                                "seed": seed,
                                "model": model,
                                "degradation": degradation,
                                "severity": severity,
                                "target": target,
                                "point_prediction": pred,
                                "interval_lower": lower,
                                "interval_upper": upper,
                                "true_value": true,
                                "covered": is_covered,
                                "valid": valid,
                                "failure_aware_covered": is_covered,
                                "frozen_quantile": q_hat,
                                "calibration_source": "clean",
                            }
                        )
                    finite_cov = [x for x in covered if not (isinstance(x, float) and math.isnan(x))]
                    results.append(
                        {
                            "seed": seed,
                            "model": model,
                            "degradation": degradation,
                            "severity": severity,
                            "target": target,
                            "nominal_coverage": TARGET_COVERAGE,
                            "empirical_coverage": float(np.mean(finite_cov)) if finite_cov else float("nan"),
                            "coverage_gap": (float(np.mean(finite_cov)) - TARGET_COVERAGE) if finite_cov else float("nan"),
                            "conditional_coverage": float(np.mean(finite_cov)) if finite_cov else float("nan"),
                            "failure_aware_coverage": float(np.mean(failure_covered)) if failure_covered else float("nan"),
                            "interval_mean_width": finite_mean(widths),
                            "interval_median_width": finite_median(widths),
                            "invalid_rate": float(np.mean(invalids)) if invalids else float("nan"),
                            "failure_rate": float(np.mean(invalids)) if invalids else float("nan"),
                            "calibration_sample_size": int(scores.size),
                            "test_sample_size": len(test_rows),
                            "frozen_quantile": q_hat,
                            "calibration_source": "clean",
                        }
                    )
    write_csv(out_dir / "conformal_results.csv", results)
    write_csv(out_dir / "conformal_per_cell.csv", per_cell)


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    scores = np.sort(scores[np.isfinite(scores)])
    if scores.size == 0:
        return float("nan")
    k = int(math.ceil((scores.size + 1) * coverage))
    k = min(max(k, 1), scores.size)
    return float(scores[k - 1])


def aggregate_degradation(out_dir: Path, rows: list[dict], deltas: list[dict]) -> None:
    """Seed/severity summaries and simple bootstrap intervals."""
    summaries = []
    for key, members in group(rows, ["split", "seed", "model", "degradation", "severity"]).items():
        summaries.append(dict(zip(["split", "seed", "model", "degradation", "severity"], key)) | summary_metrics(members))
    write_csv(out_dir / "degradation_metrics_by_seed_severity.csv", summaries)
    boot = []
    for key, members in group([d for d in deltas if d["split"] == "test"], ["model", "degradation", "severity"]).items():
        for metric in ["delta_foreground_dice", "delta_nc_absolute_error", "delta_circularity_absolute_error"]:
            ci = cluster_bootstrap(members, metric)
            boot.append(dict(zip(["model", "degradation", "severity"], key)) | {"metric": metric, **ci})
    write_csv(out_dir / "cluster_bootstrap_intervals.csv", boot)


def cluster_bootstrap(rows: list[dict], metric: str) -> dict:
    """Cell-cluster bootstrap over all rows for selected cell IDs."""
    by_cell = group(rows, ["cell_id"])
    cells = list(by_cell)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(BOOTSTRAP_REPEATS):
        selected = rng.choice(len(cells), size=len(cells), replace=True)
        sample = [v for idx in selected for v in by_cell[cells[idx]]]
        vals = np.asarray([r[metric] for r in sample if np.isfinite(r[metric])], dtype=float)
        values.append(float(vals.mean()) if vals.size else float("nan"))
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    observed = np.asarray([r[metric] for r in rows if np.isfinite(r[metric])], dtype=float)
    return {
        "observed_mean": float(observed.mean()) if observed.size else float("nan"),
        "ci_low": float(np.percentile(arr, 2.5)) if arr.size else float("nan"),
        "ci_high": float(np.percentile(arr, 97.5)) if arr.size else float("nan"),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def decoupling(out_dir: Path, deltas: list[dict]) -> None:
    """Write cases where Dice barely changes but morphometry worsens strongly."""
    nc_q3 = np.percentile([d["delta_nc_absolute_error"] for d in deltas if np.isfinite(d["delta_nc_absolute_error"])], 75)
    circ_q3 = np.percentile([d["delta_circularity_absolute_error"] for d in deltas if np.isfinite(d["delta_circularity_absolute_error"])], 75)
    cases = []
    for d in deltas:
        small_dice = abs(d["delta_foreground_dice"]) <= 0.01
        if small_dice and (d["delta_nc_absolute_error"] >= nc_q3 or d["delta_circularity_absolute_error"] >= circ_q3):
            cases.append(d)
    write_csv(out_dir / "dice_morphometry_decoupling_cases.csv", cases)


def plots(out_dir: Path, rows: list[dict], deltas: list[dict]) -> None:
    """Four required core figures."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    test = [r for r in rows if r["split"] == "test"]
    for metric, name, ylabel in [
        ("foreground_dice", "severity_foreground_dice.png", "Foreground Dice"),
        ("nc_absolute_error", "severity_nc_error.png", "N/C absolute error"),
        ("circularity_absolute_error", "severity_circularity_error.png", "Circularity absolute error"),
    ]:
        summary = []
        for key, members in group(test, ["model", "degradation", "severity"]).items():
            summary.append((key, finite_mean([m[metric] for m in members])))
        plt.figure(figsize=(7, 4))
        for model in MODELS:
            xs = []
            ys = []
            labels = []
            for index, condition in enumerate(conditions()):
                vals = [v for (k, v) in summary if k == (model, condition[0], condition[1])]
                if vals:
                    xs.append(index)
                    ys.append(vals[0])
                    labels.append(f"{condition[0].replace('gaussian_', '')}:{condition[1]}")
            plt.plot(xs, ys, marker="o", label=model)
        plt.xticks(range(len(conditions())), [f"{d.replace('gaussian_', '')}\n{s}" for d, s in conditions()])
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=160)
        plt.close()
    plt.figure(figsize=(7, 4))
    vals = [d["delta_nc_absolute_error"] for d in deltas if d["split"] == "test" and np.isfinite(d["delta_nc_absolute_error"])]
    plt.hist(vals, bins=40)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Degraded - clean N/C absolute error")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(fig_dir / "degraded_clean_nc_delta_distribution.png", dpi=160)
    plt.close()


def interim_decision(out_dir: Path) -> None:
    """Create short interim scientific decision from core summaries."""
    boot = list(csv.DictReader((out_dir / "cluster_bootstrap_intervals.csv").open(newline="", encoding="utf-8")))
    conf = list(csv.DictReader((out_dir / "conformal_results.csv").open(newline="", encoding="utf-8")))
    nc_degraded = [float(r["observed_mean"]) for r in boot if r["metric"] == "delta_nc_absolute_error"]
    circ_degraded = [float(r["observed_mean"]) for r in boot if r["metric"] == "delta_circularity_absolute_error"]
    coverage_gaps = [float(r["coverage_gap"]) for r in conf if r["degradation"] != "clean" and r["coverage_gap"] not in {"", "nan"}]
    positive_nc = sum(v > 0 for v in nc_degraded)
    positive_circ = sum(v > 0 for v in circ_degraded)
    below_nominal = sum(v < 0 for v in coverage_gaps)
    if positive_nc >= len(nc_degraded) * 0.7 and positive_circ >= len(circ_degraded) * 0.7 and below_nominal >= len(coverage_gaps) * 0.7:
        decision = "GÜÇLÜ SİNYAL"
    elif positive_nc >= len(nc_degraded) * 0.5 or positive_circ >= len(circ_degraded) * 0.5 or below_nominal >= len(coverage_gaps) * 0.5:
        decision = "SINIRLI SİNYAL"
    else:
        decision = "SİSTEMATİK ÖRÜNTÜ YOK"
    summary = {
        "interim_decision": decision,
        "nc_degraded_positive_mean_count": positive_nc,
        "nc_degraded_condition_count": len(nc_degraded),
        "circularity_degraded_positive_mean_count": positive_circ,
        "circularity_degraded_condition_count": len(circ_degraded),
        "coverage_below_nominal_count": below_nominal,
        "coverage_condition_count": len(coverage_gaps),
        "scope": "core-only analysis; no retraining, no new lambda",
    }
    (out_dir / "final_analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "final_analysis_report.md").write_text(
        "# Core Interim Scientific Decision\n\n"
        f"Decision: **{decision}**\n\n"
        "This report is intentionally short because the task was narrowed to core outputs before producing the full figure/test/report set.\n\n"
        f"N/C degraded-positive conditions: {positive_nc}/{len(nc_degraded)}\n\n"
        f"Circularity degraded-positive conditions: {positive_circ}/{len(circ_degraded)}\n\n"
        f"Below-nominal conformal coverage conditions: {below_nominal}/{len(coverage_gaps)}\n",
        encoding="utf-8",
    )


def coerce_numeric(rows: list[dict]) -> None:
    """Convert metric fields to numeric in-place."""
    metric_keys = {
        "seed",
        "lambda_nc",
        "best_epoch",
        "foreground_dice",
        "nucleus_dice",
        "cytoplasm_dice",
        "true_nc",
        "predicted_nc",
        "nc_absolute_error",
        "nc_relative_error",
        "true_circularity",
        "predicted_circularity",
        "circularity_absolute_error",
        "nucleus_area_absolute_error",
        "cytoplasm_area_absolute_error",
        "true_nucleus_area",
        "predicted_nucleus_area",
        "true_cytoplasm_area",
        "predicted_cytoplasm_area",
    }
    for row in rows:
        for key in metric_keys & set(row):
            row[key] = float(row[key])
            if key in {"seed", "best_epoch", "true_nucleus_area", "predicted_nucleus_area", "true_cytoplasm_area", "predicted_cytoplasm_area"}:
                row[key] = int(row[key])
        row["invalid_nc"] = row["invalid_nc"] in {True, "True"}
        row["invalid_circularity"] = row["invalid_circularity"] in {True, "True"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    config = load_config(ROOT_DIR / args.config)
    data_dir = resolve_herlev_data_dir(config, args.data_dir)
    out_dir = ROOT_DIR / "results" / "final_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_scale(out_dir)
    clean_diagnostics(out_dir)
    split = json.loads((ROOT_DIR / "data" / "splits" / "original_herlev_group_split.json").read_text(encoding="utf-8"))
    leakage_audit(out_dir, split)
    predictions_path = out_dir / "degradation_per_cell_predictions.csv"
    if predictions_path.exists():
        rows = list(csv.DictReader(predictions_path.open(newline="", encoding="utf-8")))
        coerce_numeric(rows)
    else:
        rows = []
        rows.extend(infer_rows(config, data_dir, "calibration", split["calibration"]))
        rows.extend(infer_rows(config, data_dir, "test", split["test"]))
        write_csv(predictions_path, rows)
    deltas = paired_deltas(rows)
    write_csv(out_dir / "degradation_paired_deltas.csv", deltas)
    aggregate_degradation(out_dir, rows, deltas)
    decoupling(out_dir, deltas)
    conformal(out_dir, rows)
    plots(out_dir, rows, deltas)
    interim_decision(out_dir)
    print((out_dir / "final_analysis_summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
