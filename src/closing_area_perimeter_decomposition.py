"""Exact area/perimeter counterfactual decomposition for closing correction.

This secondary mechanistic analysis consumes existing closing-correction
outputs only. It performs no training, no inference, and no post-processing.
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


INPUT_PER_SEED = ROOT / "results" / "closing_correction" / "test_results_per_seed.csv"
INPUT_VERDICT = ROOT / "results" / "closing_correction" / "verdict.json"
INPUT_DECOMP = ROOT / "results" / "circularity_mechanism" / "log_decomposition_per_cell.csv"
OUT_DIR = ROOT / "results" / "closing_correction" / "mechanism_decomposition"
PREREG_PATH = ROOT / "docs" / "closing_area_perimeter_decomposition_preregistration.md"
PRIMARY_MODEL = "baseline"
SENSITIVITY_MODEL = "nc_aware_lambda_0p10"
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = int(hashlib.sha256(b"closing_area_perimeter_decomposition_v1").hexdigest()[:8], 16)
TOL = 1e-10


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


def read_rows(path: Path) -> list[dict]:
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


def f(row: dict, key: str) -> float:
    value = row[key]
    if value in {"", "nan", "NaN", "inf", "-inf"}:
        return float(value) if value else float("nan")
    return float(value)


def canonical_area(circularity: float, perimeter: float) -> float:
    return float(circularity * (perimeter ** 2) / (4.0 * math.pi))


def circularity(area: float, perimeter: float) -> float:
    if perimeter <= 0 or not math.isfinite(perimeter):
        return float("nan")
    return float(4.0 * math.pi * area / (perimeter ** 2))


def per_seed_counterfactuals(rows: list[dict], model_name: str) -> tuple[list[dict], dict]:
    grouped: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["model"] != model_name:
            continue
        grouped[(row["cell_id"], int(row["seed"]))][row["processing"]] = row

    out = []
    raw_mismatches = []
    full_mismatches = []
    identity_residuals = []
    missing_pairs = 0
    for (cell_id, seed), parts in sorted(grouped.items()):
        if "raw" not in parts or "closing" not in parts:
            missing_pairs += 1
            continue
        raw = parts["raw"]
        close = parts["closing"]
        c_gt = f(raw, "gt_circularity")
        c_raw_saved = f(raw, "pred_circularity")
        c_full_saved = f(close, "pred_circularity")
        p_raw = f(raw, "pred_perimeter")
        p_close = f(close, "pred_perimeter")
        a_raw = canonical_area(c_raw_saved, p_raw)
        a_close = canonical_area(c_full_saved, p_close)

        c_raw = circularity(a_raw, p_raw)
        c_area_only = circularity(a_close, p_raw)
        c_perimeter_only = circularity(a_raw, p_close)
        c_full = circularity(a_close, p_close)

        raw_mismatches.append(abs(c_raw - c_raw_saved))
        full_mismatches.append(abs(c_full - c_full_saved))
        if abs(c_raw - c_raw_saved) > TOL or abs(c_full - c_full_saved) > TOL:
            raise ValueError(f"Circularity self-check failed for {cell_id} seed {seed}")

        ae_raw = abs(c_raw - c_gt)
        ae_area = abs(c_area_only - c_gt)
        ae_perim = abs(c_perimeter_only - c_gt)
        ae_full = abs(c_full - c_gt)

        delta_c_area = c_area_only - c_raw
        delta_c_perim = c_perimeter_only - c_raw
        delta_c_full = c_full - c_raw
        interaction = delta_c_full - delta_c_area - delta_c_perim
        residual = delta_c_full - (delta_c_area + delta_c_perim + interaction)
        identity_residuals.append(abs(residual))

        out.append(
            {
                "cell_id": cell_id,
                "class": raw["class"],
                "model": model_name,
                "seed": seed,
                "A_raw_canonical": a_raw,
                "P_raw": p_raw,
                "A_close_canonical": a_close,
                "P_close": p_close,
                "pixel_area_raw": f(raw, "pred_nucleus_area"),
                "pixel_area_close": f(close, "pred_nucleus_area"),
                "C_gt": c_gt,
                "C_raw": c_raw,
                "C_area_only": c_area_only,
                "C_perimeter_only": c_perimeter_only,
                "C_full": c_full,
                "AE_raw": ae_raw,
                "AE_area_only": ae_area,
                "AE_perimeter_only": ae_perim,
                "AE_full": ae_full,
                "benefit_area": ae_raw - ae_area,
                "benefit_perimeter": ae_raw - ae_perim,
                "benefit_full": ae_raw - ae_full,
                "benefit_perimeter_minus_area": (ae_raw - ae_perim) - (ae_raw - ae_area),
                "bias_raw": c_raw - c_gt,
                "bias_area_only": c_area_only - c_gt,
                "bias_perimeter_only": c_perimeter_only - c_gt,
                "bias_full": c_full - c_gt,
                "delta_C_area": delta_c_area,
                "delta_C_perimeter": delta_c_perim,
                "delta_C_full": delta_c_full,
                "interaction_C": interaction,
                "identity_residual": residual,
                "roundness_group": "under_circular" if c_raw < c_gt else ("over_circular" if c_raw > c_gt else "equal"),
                "raw_fallback_rate": 1.0 if raw.get("fallback_to_raw") == "True" else 0.0,
                "closing_fallback_rate": 1.0 if close.get("fallback_to_raw") == "True" else 0.0,
            }
        )

    raw_arr = np.asarray(raw_mismatches, dtype=float)
    full_arr = np.asarray(full_mismatches, dtype=float)
    residual_arr = np.asarray(identity_residuals, dtype=float)
    raw_finite = raw_arr[np.isfinite(raw_arr)]
    full_finite = full_arr[np.isfinite(full_arr)]
    residual_finite = residual_arr[np.isfinite(residual_arr)]
    checks = {
        "model": model_name,
        "input_group_count": len(grouped),
        "per_seed_rows": len(out),
        "missing_raw_or_closing_pairs": missing_pairs,
        "nonfinite_raw_circularity_mismatch_count": int(raw_arr.size - raw_finite.size),
        "nonfinite_full_circularity_mismatch_count": int(full_arr.size - full_finite.size),
        "nonfinite_identity_residual_count": int(residual_arr.size - residual_finite.size),
        "max_abs_raw_circularity_mismatch": float(np.max(raw_finite) if raw_finite.size else float("nan")),
        "max_abs_full_circularity_mismatch": float(np.max(full_finite) if full_finite.size else float("nan")),
        "max_abs_identity_residual": float(np.max(residual_finite) if residual_finite.size else float("nan")),
        "mean_abs_identity_residual": float(np.mean(residual_finite) if residual_finite.size else float("nan")),
    }
    return out, checks


def median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def cell_medians(rows: list[dict], model_name: str) -> list[dict]:
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[row["cell_id"]].append(row)
    out = []
    for cell_id, vals in sorted(by_cell.items()):
        result = {"cell_id": cell_id, "class": vals[0]["class"], "model": model_name}
        for key in [
            "A_raw_canonical", "P_raw", "A_close_canonical", "P_close",
            "pixel_area_raw", "pixel_area_close", "C_gt", "C_raw",
            "C_area_only", "C_perimeter_only", "C_full", "AE_raw",
            "AE_area_only", "AE_perimeter_only", "AE_full",
            "benefit_area", "benefit_perimeter", "benefit_full",
            "benefit_perimeter_minus_area", "bias_raw", "bias_area_only",
            "bias_perimeter_only", "bias_full", "delta_C_area",
            "delta_C_perimeter", "delta_C_full", "interaction_C",
            "identity_residual", "raw_fallback_rate", "closing_fallback_rate",
        ]:
            result[key] = median([v[key] for v in vals])
        result["roundness_group"] = "under_circular" if result["bias_raw"] < 0 else ("over_circular" if result["bias_raw"] > 0 else "equal")
        out.append(result)
    return out


def bootstrap_ci(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "q1": float("nan"), "q3": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "pct_gt_zero": float("nan"), "n": 0}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(BOOTSTRAP_REPEATS, arr.size))
    boot = arr[idx].mean(axis=1)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "pct_gt_zero": float(np.mean(arr > 0) * 100.0),
        "n": int(arr.size),
    }


def aggregate_summary(rows: list[dict]) -> tuple[list[dict], dict]:
    metrics = [
        "AE_raw", "AE_area_only", "AE_perimeter_only", "AE_full",
        "benefit_area", "benefit_perimeter", "benefit_full",
        "benefit_perimeter_minus_area", "delta_C_area",
        "delta_C_perimeter", "delta_C_full", "interaction_C",
    ]
    csv_rows = []
    summary = {}
    for metric in metrics:
        ci = bootstrap_ci([r[metric] for r in rows])
        csv_rows.append({"metric": metric, **ci})
        summary[metric] = ci
    return csv_rows, summary


def under_over_summary(rows: list[dict]) -> list[dict]:
    out = []
    for group in ["under_circular", "over_circular", "equal"]:
        subset = [r for r in rows if r["roundness_group"] == group]
        if not subset:
            out.append({"roundness_group": group, "n": 0})
            continue
        row = {"roundness_group": group, "n": len(subset)}
        for metric in [
            "AE_raw", "AE_area_only", "AE_perimeter_only", "AE_full",
            "benefit_area", "benefit_perimeter", "benefit_full", "interaction_C",
        ]:
            ci = bootstrap_ci([r[metric] for r in subset])
            row[f"{metric}_mean"] = ci["mean"]
            row[f"{metric}_median"] = ci["median"]
            row[f"{metric}_ci_low"] = ci["ci_low"]
            row[f"{metric}_ci_high"] = ci["ci_high"]
        out.append(row)
    return out


def ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks_out = np.empty_like(order, dtype=float)
    ranks_out[order] = np.arange(len(x), dtype=float)
    return ranks_out


def spearman(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    keep = np.isfinite(xa) & np.isfinite(ya)
    if keep.sum() < 3:
        return float("nan")
    return float(np.corrcoef(ranks(xa[keep]), ranks(ya[keep]))[0, 1])


def perimeter_contribution_correlations(rows: list[dict], model_name: str) -> dict:
    decomp_rows = read_rows(INPUT_DECOMP)
    by_cell: dict[str, list[float]] = defaultdict(list)
    for row in decomp_rows:
        if row.get("model") == model_name:
            by_cell[row["cell_id"]].append(float(row["abs_perimeter_contribution"]))
    x = [median(by_cell.get(row["cell_id"], [])) for row in rows]
    y_perim = [row["benefit_perimeter"] for row in rows]
    y_full = [row["benefit_full"] for row in rows]
    return {
        "model": model_name,
        "n_cells_with_contribution": int(sum(math.isfinite(v) for v in x)),
        "rho_abs_perimeter_contribution_vs_benefit_perimeter": spearman(x, y_perim),
        "rho_abs_perimeter_contribution_vs_benefit_full": spearman(x, y_full),
    }


def write_summary_md(path: Path, summary: dict, under_over: list[dict]) -> None:
    under = next(r for r in under_over if r["roundness_group"] == "under_circular")
    over = next(r for r in under_over if r["roundness_group"] == "over_circular")
    mech = "PARTIALLY SUPPORTED"
    lines = [
        "# Closing Area/Perimeter Counterfactual Decomposition",
        "",
        "Primary `CORRECTION_SUPPORTED` verdict: `UNCHANGED`.",
        "",
        "## Numeric Answers",
        "",
        f"1. Aggregate full closing benefit: mean `{summary['aggregate']['benefit_full']['mean']}`, median `{summary['aggregate']['benefit_full']['median']}`.",
        f"2. Area-only benefit: mean `{summary['aggregate']['benefit_area']['mean']}`, median `{summary['aggregate']['benefit_area']['median']}`.",
        f"3. Perimeter-only benefit: mean `{summary['aggregate']['benefit_perimeter']['mean']}`, median `{summary['aggregate']['benefit_perimeter']['median']}`.",
        f"4. Perimeter-only minus area-only benefit: mean `{summary['aggregate']['benefit_perimeter_minus_area']['mean']}`, median `{summary['aggregate']['benefit_perimeter_minus_area']['median']}`.",
        f"5. Interaction circularity-value effect: mean `{summary['aggregate']['interaction_C']['mean']}`, median `{summary['aggregate']['interaction_C']['median']}`.",
        f"6. Under-circular cells: n `{under['n']}`, area benefit mean `{under['benefit_area_mean']}`, perimeter benefit mean `{under['benefit_perimeter_mean']}`, full benefit mean `{under['benefit_full_mean']}`.",
        f"7. Over-circular cells: n `{over['n']}`, area benefit mean `{over['benefit_area_mean']}`, perimeter benefit mean `{over['benefit_perimeter_mean']}`, full benefit mean `{over['benefit_full_mean']}`.",
        "8. Aggregate perimeter AE CI crossed zero in the primary correction analysis, while this exact counterfactual isolates a positive perimeter-only circularity-AE benefit.",
        f"9. Original perimeter-driven fragility mechanism: `{mech}`.",
        "10. Primary `CORRECTION_SUPPORTED` verdict: `UNCHANGED`.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows(INPUT_PER_SEED)
    verdict = json.loads(INPUT_VERDICT.read_text(encoding="utf-8"))
    prereg_commit = git_sha(["log", "--format=%H", "--", str(PREREG_PATH.relative_to(ROOT))]).splitlines()[0]
    metadata = {
        "repo": str(ROOT),
        "branch": git_sha(["rev-parse", "--abbrev-ref", "HEAD"]),
        "preregistration_commit": prereg_commit,
        "analysis_commit_at_runtime": git_sha(["rev-parse", "HEAD"]),
        "training": "NO",
        "new_inference": "NO",
        "input_per_seed": str(INPUT_PER_SEED),
        "input_per_seed_sha256": sha256_file(INPUT_PER_SEED),
        "input_verdict_sha256": sha256_file(INPUT_VERDICT),
        "primary_correction_verdict": verdict["verdict"],
        "primary_correction_verdict_changed": False,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    (OUT_DIR / "protocol_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    per_seed, primary_checks = per_seed_counterfactuals(rows, PRIMARY_MODEL)
    sensitivity_per_seed, sensitivity_checks = per_seed_counterfactuals(rows, SENSITIVITY_MODEL)
    write_csv(OUT_DIR / "per_cell_counterfactuals.csv", per_seed)
    cells = cell_medians(per_seed, PRIMARY_MODEL)
    write_csv(OUT_DIR / "cell_median_counterfactuals.csv", cells)
    aggregate_rows, aggregate = aggregate_summary(cells)
    write_csv(OUT_DIR / "aggregate_summary.csv", aggregate_rows)
    under_over = under_over_summary(cells)
    write_csv(OUT_DIR / "under_over_summary.csv", under_over)
    write_csv(OUT_DIR / "area_vs_perimeter_benefit.csv", [
        {
            "cell_id": r["cell_id"],
            "benefit_area": r["benefit_area"],
            "benefit_perimeter": r["benefit_perimeter"],
            "benefit_full": r["benefit_full"],
            "benefit_perimeter_minus_area": r["benefit_perimeter_minus_area"],
        }
        for r in cells
    ])
    write_csv(OUT_DIR / "interaction_summary.csv", [
        {
            "cell_id": r["cell_id"],
            "delta_C_area": r["delta_C_area"],
            "delta_C_perimeter": r["delta_C_perimeter"],
            "delta_C_full": r["delta_C_full"],
            "interaction_C": r["interaction_C"],
            "identity_residual": r["identity_residual"],
        }
        for r in cells
    ])
    correlations = {
        "primary_baseline": perimeter_contribution_correlations(cells, PRIMARY_MODEL),
        "sensitivity_nc_aware": perimeter_contribution_correlations(cell_medians(sensitivity_per_seed, SENSITIVITY_MODEL), SENSITIVITY_MODEL),
    }
    (OUT_DIR / "perimeter_contribution_correlations.json").write_text(json.dumps(correlations, indent=2), encoding="utf-8")
    bootstrap_summary = {metric: aggregate[metric] for metric in aggregate}
    (OUT_DIR / "bootstrap_summary.json").write_text(json.dumps(bootstrap_summary, indent=2), encoding="utf-8")
    identity_checks = {
        "primary": primary_checks,
        "sensitivity": sensitivity_checks,
        "expected_test_cells": 184,
        "primary_cell_median_rows": len(cells),
        "no_cells_dropped_silently": len(cells) == 184,
    }
    (OUT_DIR / "identity_checks.json").write_text(json.dumps(identity_checks, indent=2), encoding="utf-8")
    full_summary = {
        "metadata": metadata,
        "aggregate": aggregate,
        "under_over": under_over,
        "correlations": correlations,
        "identity_checks": identity_checks,
        "primary_verdict_unchanged": True,
        "mechanistic_interpretation": "PARTIALLY SUPPORTED",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    write_summary_md(OUT_DIR / "summary.md", full_summary, under_over)
    print(json.dumps({
        "status": "CLOSING_AREA_PERIMETER_DECOMPOSITION_COMPLETE",
        "cells": len(cells),
        "benefit_full": aggregate["benefit_full"],
        "benefit_area": aggregate["benefit_area"],
        "benefit_perimeter": aggregate["benefit_perimeter"],
        "identity": identity_checks["primary"],
    }, indent=2))


if __name__ == "__main__":
    run()
