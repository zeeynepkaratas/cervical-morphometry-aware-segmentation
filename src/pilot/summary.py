"""Summarization and GO/NO-GO decision helpers."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from src.pilot.metrics import finite_mean, finite_median


def summarize_per_cell(per_cell_csv: Path, output_dir: Path) -> dict:
    """Create model/condition summaries and a locked pilot decision."""
    rows = list(csv.DictReader(Path(per_cell_csv).open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"No per-cell rows found in {per_cell_csv}")
    for row in rows:
        for key in (
            "nucleus_dice",
            "cytoplasm_dice",
            "foreground_mean_dice",
            "nucleus_iou",
            "cytoplasm_iou",
            "nc_absolute_error",
            "nc_relative_error",
            "circularity_absolute_error",
        ):
            row[key] = float(row[key])
        row["invalid_nc_prediction"] = row["invalid_nc_prediction"] == "True"
        row["invalid_circularity_prediction"] = row["invalid_circularity_prediction"] == "True"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_model = _group_summary(rows, ["model"])
    by_condition = _group_summary(rows, ["model", "condition_family"])
    by_seed = _group_summary(rows, ["model", "seed"])
    _write_csv(output_dir / "metrics_by_model.csv", by_model)
    _write_csv(output_dir / "metrics_by_condition.csv", by_condition)
    _write_csv(output_dir / "metrics_by_seed.csv", by_seed)

    baseline = next((row for row in by_model if row["model"] == "baseline_lambda_0"), None)
    candidates = [row for row in by_model if row["model"] != "baseline_lambda_0"]
    best = min(candidates, key=lambda row: row["mean_nc_absolute_error"], default=None)
    decision = _decide(baseline, best, by_condition)
    summary = {"baseline": baseline, "best_candidate": best, "decision": decision}
    (output_dir / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "pilot_decision.md").write_text(_decision_markdown(summary), encoding="utf-8")
    return summary


def _group_summary(rows: list[dict], keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summaries = []
    for values, group_rows in sorted(grouped.items()):
        out = dict(zip(keys, values))
        out.update(
            n=len(group_rows),
            nucleus_dice=finite_mean(row["nucleus_dice"] for row in group_rows),
            cytoplasm_dice=finite_mean(row["cytoplasm_dice"] for row in group_rows),
            foreground_mean_dice=finite_mean(row["foreground_mean_dice"] for row in group_rows),
            nucleus_iou=finite_mean(row["nucleus_iou"] for row in group_rows),
            cytoplasm_iou=finite_mean(row["cytoplasm_iou"] for row in group_rows),
            mean_nc_absolute_error=finite_mean(row["nc_absolute_error"] for row in group_rows),
            median_nc_absolute_error=finite_median(row["nc_absolute_error"] for row in group_rows),
            mean_nc_relative_error=finite_mean(row["nc_relative_error"] for row in group_rows),
            invalid_nc_prediction_rate=sum(row["invalid_nc_prediction"] for row in group_rows) / len(group_rows),
            mean_circularity_absolute_error=finite_mean(row["circularity_absolute_error"] for row in group_rows),
            median_circularity_absolute_error=finite_median(row["circularity_absolute_error"] for row in group_rows),
            invalid_circularity_prediction_rate=sum(row["invalid_circularity_prediction"] for row in group_rows) / len(group_rows),
        )
        summaries.append(out)
    return summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _decide(baseline: dict | None, best: dict | None, by_condition: list[dict]) -> dict:
    if baseline is None or best is None:
        return {"class": "NO_GO", "reason": "Baseline or candidate summary missing."}
    nc_delta = best["mean_nc_absolute_error"] - baseline["mean_nc_absolute_error"]
    nc_relative_change = nc_delta / baseline["mean_nc_absolute_error"] if baseline["mean_nc_absolute_error"] else float("nan")
    dice_delta = best["foreground_mean_dice"] - baseline["foreground_mean_dice"]
    circ_delta = best["mean_circularity_absolute_error"] - baseline["mean_circularity_absolute_error"]
    families_improved = 0
    for family in ("clean", "blur", "noise"):
        b = next((row for row in by_condition if row["model"] == baseline["model"] and row["condition_family"] == family), None)
        c = next((row for row in by_condition if row["model"] == best["model"] and row["condition_family"] == family), None)
        if b and c and c["mean_nc_absolute_error"] < b["mean_nc_absolute_error"]:
            families_improved += 1
    checks = {
        "nc_error_reduced_at_least_10pct": nc_relative_change <= -0.10,
        "foreground_dice_drop_not_over_0p01": dice_delta >= -0.01,
        "invalid_nc_not_higher": best["invalid_nc_prediction_rate"] <= baseline["invalid_nc_prediction_rate"] + 0.01,
        "improves_at_least_two_condition_families": families_improved >= 2,
        "circularity_not_worse_over_10pct": (
            circ_delta <= 0.10 * baseline["mean_circularity_absolute_error"]
            if baseline["mean_circularity_absolute_error"]
            else True
        ),
    }
    passed = sum(checks.values())
    decision = "GO" if passed >= 5 else "CONDITIONAL_GO" if passed >= 3 else "NO_GO"
    return {
        "class": decision,
        "best_model": best["model"],
        "nc_relative_change": nc_relative_change,
        "foreground_dice_delta": dice_delta,
        "circularity_absolute_error_delta": circ_delta,
        "families_improved": families_improved,
        "checks": checks,
    }


def _decision_markdown(summary: dict) -> str:
    decision = summary["decision"]
    return (
        f"# Pilot Decision\n\n"
        f"Decision: **{decision['class']}**\n\n"
        f"Best model: `{decision.get('best_model', 'none')}`\n\n"
        f"N/C relative change: `{decision.get('nc_relative_change')}`\n\n"
        f"Foreground Dice delta: `{decision.get('foreground_dice_delta')}`\n\n"
        f"Circularity absolute-error delta: `{decision.get('circularity_absolute_error_delta')}`\n\n"
        "Thresholds were fixed before summarization in the pilot protocol.\n"
    )
