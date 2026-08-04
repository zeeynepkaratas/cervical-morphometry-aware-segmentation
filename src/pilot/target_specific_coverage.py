"""Target-specific conformal coverage fragility helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


TARGETS = ("nc", "circularity")
KEY_FIELDS = ("seed", "model", "degradation", "severity")


def severity_value(value) -> float:
    """Normalize severity values for sorting and exact condition matching."""
    return -1.0 if value == "clean" else float(value)


def condition_key(row: dict) -> tuple:
    """Return the within-condition key used to match N/C and circularity."""
    return (
        int(row["seed"]),
        row["model"],
        row["degradation"],
        severity_value(row["severity"]),
    )


def mean(values: Iterable[float]) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def median(values: Iterable[float]) -> float:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def trend_label(values: list[float], expected: str) -> str:
    """Classify monotonic behavior across ordered severities."""
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if len(vals) < 2:
        return "inconclusive"
    if expected == "decrease":
        steps = [b <= a for a, b in zip(vals, vals[1:])]
    elif expected == "increase":
        steps = [b >= a for a, b in zip(vals, vals[1:])]
    else:
        raise ValueError(f"Unexpected expected trend: {expected}")
    if all(steps):
        return "fully_monotonic"
    if sum(steps) >= max(1, len(steps) - 1):
        return "mostly_monotonic"
    return "inconsistent"


def pair_target_changes(gap_rows: list[dict]) -> list[dict]:
    """Pair N/C and circularity coverage changes within the same condition."""
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in gap_rows:
        if row["degradation"] == "clean":
            continue
        target = row["target"]
        if target in TARGETS:
            grouped[condition_key(row)][target] = row

    paired = []
    for key, by_target in sorted(grouped.items()):
        if not all(target in by_target for target in TARGETS):
            continue
        nc = by_target["nc"]
        circularity = by_target["circularity"]
        nc_delta = float(nc["coverage_delta_vs_clean"])
        circularity_delta = float(circularity["coverage_delta_vs_clean"])
        diff = circularity_delta - nc_delta
        paired.append(
            {
                "seed": key[0],
                "model": key[1],
                "degradation": key[2],
                "severity": key[3],
                "nc_clean_coverage": float(nc["clean_coverage"]),
                "nc_degraded_coverage": float(nc["empirical_coverage"]),
                "nc_coverage_change": nc_delta,
                "circularity_clean_coverage": float(circularity["clean_coverage"]),
                "circularity_degraded_coverage": float(circularity["empirical_coverage"]),
                "circularity_coverage_change": circularity_delta,
                "target_gap_difference": diff,
                "more_fragile_target": more_fragile_target(diff),
            }
        )
    return paired


def more_fragile_target(target_gap_difference: float, tolerance: float = 1e-12) -> str:
    """Interpret circularity_delta - nc_delta; negative means circularity fell more."""
    diff = float(target_gap_difference)
    if diff < -tolerance:
        return "circularity"
    if diff > tolerance:
        return "nc"
    return "equal"


def coverage_from_per_cell(rows: list[dict]) -> float:
    valid = [row for row in rows if bool(row["valid"])]
    if not valid:
        return float("nan")
    return mean(1.0 if bool(row["covered"]) else 0.0 for row in valid)


def target_gap_from_per_cell(rows: list[dict]) -> float:
    """Compute mean circularity-minus-N/C coverage-change from per-cell rows."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["target"] in TARGETS:
            grouped[condition_key(row) + (row["target"],)].append(row)

    clean_cov = {}
    for key, condition_rows in grouped.items():
        seed, model, degradation, _severity, target = key
        if degradation == "clean":
            clean_cov[(seed, model, target)] = coverage_from_per_cell(condition_rows)

    deltas: dict[tuple, dict[str, float]] = defaultdict(dict)
    for key, condition_rows in grouped.items():
        seed, model, degradation, severity, target = key
        if degradation == "clean":
            continue
        base = clean_cov.get((seed, model, target), float("nan"))
        cov = coverage_from_per_cell(condition_rows)
        if np.isfinite(base) and np.isfinite(cov):
            deltas[(seed, model, degradation, severity)][target] = cov - base

    diffs = []
    for by_target in deltas.values():
        if all(target in by_target for target in TARGETS):
            diffs.append(by_target["circularity"] - by_target["nc"])
    return mean(diffs)


def rows_by_cell_id(rows: list[dict]) -> dict[str, list[dict]]:
    """Group all repeated observations for each cell_id together."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cell_id"])].append(row)
    return dict(grouped)


def cluster_bootstrap_target_gap(
    per_cell_rows: list[dict],
    repeats: int = 5000,
    seed: int = 20260804,
) -> dict:
    """Bootstrap target-gap difference with cell_id as the sampling unit."""
    clusters = rows_by_cell_id(per_cell_rows)
    cell_ids = np.asarray(sorted(clusters))
    if cell_ids.size == 0:
        return {
            "metric": "circularity_minus_nc_coverage_change",
            "observed_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "bootstrap_repeats": 0,
            "bootstrap_seed": seed,
            "cluster_unit": "cell_id",
            "cluster_count": 0,
            "status": "failed_no_cell_ids",
        }

    condition_keys = sorted(
        {
            condition_key(row) + (row["target"],)
            for row in per_cell_rows
            if row["target"] in TARGETS
        }
    )
    key_to_index = {key: idx for idx, key in enumerate(condition_keys)}
    covered = np.zeros((cell_ids.size, len(condition_keys)), dtype=float)
    valid = np.zeros_like(covered)
    cell_to_index = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
    for row in per_cell_rows:
        if row["target"] not in TARGETS:
            continue
        cell_index = cell_to_index[str(row["cell_id"])]
        key_index = key_to_index[condition_key(row) + (row["target"],)]
        if bool(row["valid"]):
            valid[cell_index, key_index] += 1.0
            covered[cell_index, key_index] += 1.0 if bool(row["covered"]) else 0.0

    pair_indices = []
    degraded_conditions = sorted(
        {
            key[:4]
            for key in condition_keys
            if key[2] != "clean"
            and key_to_index.get(key[:4] + ("nc",)) is not None
            and key_to_index.get(key[:4] + ("circularity",)) is not None
        }
    )
    for seed_value, model, degradation, severity in degraded_conditions:
        nc_degraded = key_to_index.get((seed_value, model, degradation, severity, "nc"))
        circ_degraded = key_to_index.get((seed_value, model, degradation, severity, "circularity"))
        nc_clean = key_to_index.get((seed_value, model, "clean", severity_value("clean"), "nc"))
        circ_clean = key_to_index.get((seed_value, model, "clean", severity_value("clean"), "circularity"))
        if None not in {nc_degraded, circ_degraded, nc_clean, circ_clean}:
            pair_indices.append((nc_clean, nc_degraded, circ_clean, circ_degraded))

    def statistic(sample_indices: np.ndarray) -> float:
        cov_sum = covered[sample_indices].sum(axis=0)
        valid_sum = valid[sample_indices].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            coverage = cov_sum / valid_sum
        diffs = []
        for nc_clean, nc_degraded, circ_clean, circ_degraded in pair_indices:
            nc_delta = coverage[nc_degraded] - coverage[nc_clean]
            circ_delta = coverage[circ_degraded] - coverage[circ_clean]
            if np.isfinite(nc_delta) and np.isfinite(circ_delta):
                diffs.append(circ_delta - nc_delta)
        return mean(diffs)

    rng = np.random.default_rng(seed)
    all_indices = np.arange(cell_ids.size)
    observed = statistic(all_indices)
    samples = np.empty(repeats, dtype=float)
    for idx in range(repeats):
        samples[idx] = statistic(rng.choice(all_indices, size=cell_ids.size, replace=True))
    arr = samples[np.isfinite(samples)]
    if arr.size == 0:
        low = high = float("nan")
        status = "failed_no_finite_samples"
    else:
        low, high = np.percentile(arr, [2.5, 97.5])
        status = "ok"
    return {
        "metric": "circularity_minus_nc_coverage_change",
        "observed_mean": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_repeats": int(repeats),
        "bootstrap_seed": int(seed),
        "cluster_unit": "cell_id",
        "cluster_count": int(cell_ids.size),
        "status": status,
    }
