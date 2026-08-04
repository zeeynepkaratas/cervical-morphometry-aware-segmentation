"""Selective completion of the core final analysis.

Consumes existing final_analysis CSVs only. It does not train models and does
not run new checkpoint inference.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import spearmanr
except ModuleNotFoundError:  # pragma: no cover
    spearmanr = None


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final_analysis"
FIG = OUT / "figures"
MODELS = ["baseline", "nc_aware_lambda_0p10"]
DEGRADATIONS = ["clean", "gaussian_blur", "gaussian_noise"]


def read_csv(path: Path) -> list[dict]:
    """Read CSV and coerce obvious numeric/boolean fields."""
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    for row in rows:
        for key, value in list(row.items()):
            if value in {"True", "False"}:
                row[key] = value == "True"
            else:
                try:
                    row[key] = float(value)
                    if key in {"seed", "best_epoch", "calibration_sample_size", "test_sample_size"}:
                        row[key] = int(row[key])
                except (TypeError, ValueError):
                    pass
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows with union fieldnames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def median(values) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def group(rows: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[key] for key in keys)].append(row)
    return out


def severity_key(value) -> float:
    return -1.0 if value == "clean" else float(value)


def coverage_gap_tables(conf: list[dict]) -> tuple[list[dict], list[dict]]:
    """Coverage gap vs clean plus interval/failure interpretation table."""
    clean = {
        (r["seed"], r["model"], r["target"]): r
        for r in conf
        if r["degradation"] == "clean"
    }
    gap_rows = []
    interval_rows = []
    for row in conf:
        key = (row["seed"], row["model"], row["target"])
        base = clean[key]
        coverage_delta = row["empirical_coverage"] - base["empirical_coverage"]
        gap_rows.append(
            {
                "seed": row["seed"],
                "model": row["model"],
                "target": row["target"],
                "degradation": row["degradation"],
                "severity": row["severity"],
                "nominal_coverage": row["nominal_coverage"],
                "clean_coverage": base["empirical_coverage"],
                "empirical_coverage": row["empirical_coverage"],
                "coverage_gap": row["coverage_gap"],
                "coverage_delta_vs_clean": coverage_delta,
                "below_nominal": row["coverage_gap"] < 0,
                "frozen_quantile": row["frozen_quantile"],
            }
        )
        interval_rows.append(
            {
                "seed": row["seed"],
                "model": row["model"],
                "target": row["target"],
                "degradation": row["degradation"],
                "severity": row["severity"],
                "empirical_coverage": row["empirical_coverage"],
                "coverage_gap": row["coverage_gap"],
                "coverage_delta_vs_clean": coverage_delta,
                "interval_mean_width": row["interval_mean_width"],
                "interval_width_delta_vs_clean": row["interval_mean_width"] - base["interval_mean_width"],
                "invalid_rate": row["invalid_rate"],
                "invalid_delta_vs_clean": row["invalid_rate"] - base["invalid_rate"],
                "failure_aware_coverage": row["failure_aware_coverage"],
                "calibration_source": row["calibration_source"],
            }
        )
    write_csv(OUT / "coverage_gap_analysis.csv", gap_rows)
    write_csv(OUT / "interval_width_failure_analysis.csv", interval_rows)
    return gap_rows, interval_rows


def trend_label(values: list[float], expected: str) -> str:
    """Classify monotonicity across severity levels."""
    if len(values) < 3 or any(not np.isfinite(v) for v in values):
        return "insufficient"
    diffs = np.diff(values)
    if expected == "increase":
        signs = diffs >= 0
        strict = diffs > 0
    else:
        signs = diffs <= 0
        strict = diffs < 0
    if bool(np.all(strict)):
        return "fully_monotonic"
    if int(np.sum(signs)) >= len(diffs) - 1:
        return "partially_monotonic"
    return "inconsistent"


def severity_trends(metrics: list[dict], conf: list[dict]) -> list[dict]:
    """Summarize directionality by seed/model/degradation."""
    rows = []
    metric_map = {
        "foreground_dice": "decrease",
        "mean_nc_absolute_error": "increase",
        "mean_circularity_absolute_error": "increase",
        "invalid_nc_rate": "increase",
    }
    for (split, seed, model, degradation), members in group(
        [r for r in metrics if r["split"] == "test" and r["degradation"] != "clean"],
        ["split", "seed", "model", "degradation"],
    ).items():
        members = sorted(members, key=lambda r: severity_key(r["severity"]))
        for metric, expected in metric_map.items():
            values = [m[metric] for m in members]
            rows.append(
                {
                    "family": "segmentation_morphometry",
                    "split": split,
                    "seed": seed,
                    "model": model,
                    "degradation": degradation,
                    "metric": metric,
                    "expected_direction": expected,
                    "values_by_severity": "|".join(f"{v:.6g}" for v in values),
                    "trend_label": trend_label(values, expected),
                }
            )
    for (seed, model, target, degradation), members in group(
        [r for r in conf if r["degradation"] != "clean"],
        ["seed", "model", "target", "degradation"],
    ).items():
        members = sorted(members, key=lambda r: severity_key(r["severity"]))
        for metric, expected in [
            ("empirical_coverage", "decrease"),
            ("coverage_gap", "decrease"),
            ("interval_mean_width", "increase"),
            ("invalid_rate", "increase"),
        ]:
            values = [m[metric] for m in members]
            rows.append(
                {
                    "family": "conformal",
                    "split": "test",
                    "seed": seed,
                    "model": model,
                    "degradation": degradation,
                    "target": target,
                    "metric": metric,
                    "expected_direction": expected,
                    "values_by_severity": "|".join(f"{v:.6g}" for v in values),
                    "trend_label": trend_label(values, expected),
                }
            )
    write_csv(OUT / "severity_trends.csv", rows)
    return rows


def nc_aware_comparison(metrics: list[dict]) -> list[dict]:
    """Short baseline vs N/C-aware table for degradation conditions."""
    base = {
        (r["split"], r["seed"], r["degradation"], r["severity"]): r
        for r in metrics
        if r["model"] == "baseline"
    }
    rows = []
    for row in metrics:
        if row["model"] != "nc_aware_lambda_0p10":
            continue
        key = (row["split"], row["seed"], row["degradation"], row["severity"])
        if key not in base or row["split"] != "test":
            continue
        b = base[key]
        rows.append(
            {
                "seed": row["seed"],
                "degradation": row["degradation"],
                "severity": row["severity"],
                "baseline_foreground_dice": b["foreground_dice"],
                "nc_aware_foreground_dice": row["foreground_dice"],
                "dice_delta_nc_aware_minus_baseline": row["foreground_dice"] - b["foreground_dice"],
                "baseline_mean_nc_error": b["mean_nc_absolute_error"],
                "nc_aware_mean_nc_error": row["mean_nc_absolute_error"],
                "mean_nc_error_delta_nc_aware_minus_baseline": row["mean_nc_absolute_error"] - b["mean_nc_absolute_error"],
                "baseline_circularity_error": b["mean_circularity_absolute_error"],
                "nc_aware_circularity_error": row["mean_circularity_absolute_error"],
                "circularity_error_delta_nc_aware_minus_baseline": row["mean_circularity_absolute_error"] - b["mean_circularity_absolute_error"],
            }
        )
    write_csv(OUT / "nc_aware_degradation_comparison.csv", rows)
    return rows


def spearman_table(per_cell: list[dict], deltas: list[dict]) -> list[dict]:
    """Dice-morphometry correlations and high-Dice worsening cases."""
    rows = []
    for key, members in group(
        [r for r in per_cell if r["split"] == "test"],
        ["model", "degradation", "severity"],
    ).items():
        dice = np.asarray([m["foreground_dice"] for m in members], dtype=float)
        for metric in ["nc_absolute_error", "circularity_absolute_error"]:
            err = np.asarray([m[metric] for m in members], dtype=float)
            keep = np.isfinite(dice) & np.isfinite(err)
            if keep.sum() >= 3 and spearmanr is not None:
                stat = spearmanr(dice[keep], err[keep])
                rho = float(stat.statistic)
                pvalue = float(stat.pvalue)
            else:
                rho = float("nan")
                pvalue = float("nan")
            rows.append(
                {
                    "model": key[0],
                    "degradation": key[1],
                    "severity": key[2],
                    "metric": metric,
                    "spearman_rho": rho,
                    "pvalue": pvalue,
                    "n": int(keep.sum()),
                }
            )
    write_csv(OUT / "dice_morphometry_spearman.csv", rows)

    clean_small = [d for d in deltas if d["split"] == "test" and abs(d["delta_foreground_dice"]) <= 0.01]
    nc_q3 = np.percentile([d["delta_nc_absolute_error"] for d in deltas if d["split"] == "test"], 75)
    circ_q3 = np.percentile([d["delta_circularity_absolute_error"] for d in deltas if d["split"] == "test"], 75)
    cases = [
        d
        for d in clean_small
        if d["delta_nc_absolute_error"] >= nc_q3 or d["delta_circularity_absolute_error"] >= circ_q3
    ]
    write_csv(OUT / "dice_morphometry_decoupling_selected_cases.csv", cases)
    return rows


def plots(gap_rows: list[dict], interval_rows: list[dict], per_cell: list[dict]) -> None:
    """Add three selective figures, keeping total figure count at seven."""
    FIG.mkdir(parents=True, exist_ok=True)
    for target, filename in [("nc", "severity_nc_coverage.png"), ("circularity", "severity_circularity_coverage.png")]:
        plt.figure(figsize=(7, 4))
        for model in MODELS:
            ys = []
            xs = []
            labels = []
            for i, cond in enumerate([("clean", "clean"), ("gaussian_blur", "1"), ("gaussian_blur", "2"), ("gaussian_blur", "3"), ("gaussian_noise", "10"), ("gaussian_noise", "20"), ("gaussian_noise", "30")]):
                vals = [
                    r["empirical_coverage"]
                    for r in gap_rows
                    if r["target"] == target and r["model"] == model and r["degradation"] == cond[0] and str(r["severity"]) == cond[1]
                ]
                if vals:
                    xs.append(i)
                    ys.append(mean(vals))
                    labels.append(f"{cond[0].replace('gaussian_', '')}\n{cond[1]}")
            plt.plot(xs, ys, marker="o", label=model)
        plt.axhline(0.90, color="black", linestyle="--", linewidth=1)
        plt.xticks(range(7), ["clean\nclean", "blur\n1", "blur\n2", "blur\n3", "noise\n10", "noise\n20", "noise\n30"])
        plt.ylabel(f"{target} empirical coverage")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / filename, dpi=160)
        plt.close()

    plt.figure(figsize=(6, 4))
    xs = [r["interval_mean_width"] for r in interval_rows if r["degradation"] != "clean"]
    ys = [r["coverage_gap"] for r in interval_rows if r["degradation"] != "clean"]
    plt.scatter(xs, ys, s=18, alpha=0.65)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Frozen interval mean width")
    plt.ylabel("Coverage gap")
    plt.tight_layout()
    plt.savefig(FIG / "coverage_gap_vs_interval_width.png", dpi=160)
    plt.close()

    test = [r for r in per_cell if r["split"] == "test"]
    dice_q3 = np.percentile([r["foreground_dice"] for r in test], 75)
    err_q3 = np.percentile([r["nc_absolute_error"] for r in test if np.isfinite(r["nc_absolute_error"])], 75)
    plt.figure(figsize=(6, 4))
    plt.scatter([r["foreground_dice"] for r in test], [r["nc_absolute_error"] for r in test], s=8, alpha=0.4)
    plt.axvline(dice_q3, color="black", linestyle="--", linewidth=1)
    plt.axhline(err_q3, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Foreground Dice")
    plt.ylabel("N/C absolute error")
    plt.tight_layout()
    plt.savefig(FIG / "high_dice_high_nc_error_scatter.png", dpi=160)
    plt.close()


def read_numeric_main() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    conf = read_csv(OUT / "conformal_results.csv")
    metrics = read_csv(OUT / "degradation_metrics_by_seed_severity.csv")
    per_cell = read_csv(OUT / "degradation_per_cell_predictions.csv")
    deltas = read_csv(OUT / "degradation_paired_deltas.csv")
    return conf, metrics, per_cell, deltas


def write_report(gap_rows: list[dict], trend_rows: list[dict], nc_rows: list[dict]) -> None:
    """Update final report with selective interpretation."""
    degraded_gaps = [r["coverage_delta_vs_clean"] for r in gap_rows if r["degradation"] != "clean"]
    strong_drop = [g for g in degraded_gaps if g <= -0.05]
    trend_counts = defaultdict(int)
    for row in trend_rows:
        trend_counts[(row["family"], row["metric"], row["trend_label"])] += 1
    nc_better = [r for r in nc_rows if r["mean_nc_error_delta_nc_aware_minus_baseline"] < 0]
    text = f"""# Selective Final Analysis Report

Decision: **ORTA AMA SAVUNULABİLİR**

The broad 17-figure expansion was intentionally not run. The selective
completion focuses on conformal coverage loss, severity trends, Dice-
morphometry decoupling, and the negative/unstable N/C-aware intervention.

## Coverage

Frozen clean-calibration intervals are below nominal in most degraded
conditions. Mean coverage delta versus clean across degraded rows is
`{mean(degraded_gaps):.4f}`; rows with at least 5 percentage-point drop:
`{len(strong_drop)}/{len(degraded_gaps)}`.

## Severity Trends

Trend labels are written to `severity_trends.csv`. Fully monotonic behavior is
not required for the story; the stronger observation is that coverage loss is
common while morphometric error trends are mixed.

## N/C-aware Model

N/C-aware has lower mean N/C error than baseline in `{len(nc_better)}/{len(nc_rows)}`
test seed-condition rows. This remains a negative or unstable intervention,
not a successful method claim.

## Recommended Scope

Keep the study centered on reliability: Dice overlap alone does not guarantee
morphometric measurement reliability, and frozen clean conformal intervals lose
coverage under distribution shift. Treat blur/noise morphometric error and the
N/C-aware loss as supporting, not central, evidence.
"""
    (OUT / "final_analysis_report.md").write_text(text, encoding="utf-8")
    summary = json.loads((OUT / "final_analysis_summary.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "user_scientific_classification": "ORTA AMA SAVUNULABİLİR",
            "mean_degraded_coverage_delta_vs_clean": mean(degraded_gaps),
            "coverage_drop_ge_5pp_rows": len(strong_drop),
            "coverage_degraded_rows": len(degraded_gaps),
            "nc_aware_better_mean_nc_rows": len(nc_better),
            "nc_aware_comparison_rows": len(nc_rows),
            "selective_completion": True,
        }
    )
    (OUT / "final_analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    conf, metrics, per_cell, deltas = read_numeric_main()
    gap_rows, interval_rows = coverage_gap_tables(conf)
    trend_rows = severity_trends(metrics, conf)
    nc_rows = nc_aware_comparison(metrics)
    spearman_table(per_cell, deltas)
    plots(gap_rows, interval_rows, per_cell)
    write_report(gap_rows, trend_rows, nc_rows)
    print((OUT / "final_analysis_summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
