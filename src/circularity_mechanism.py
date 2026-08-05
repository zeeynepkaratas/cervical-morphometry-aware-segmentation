"""Circularity mechanism analysis — Phase A.

Loads existing clean and noise prediction CSVs and enriches them with
derived perimeter values (back-calculated from area and circularity),
relative errors, and clean→noise delta columns.

Perimeter derivation
--------------------
The project uses ``compute_circularity`` which is defined as::

    circularity = 4 * pi * area / perimeter**2

Rearranging::

    perimeter = sqrt(4 * pi * area / circularity)

This is mathematically exact *given* the area and circularity values that
are already stored in the CSV.  Both quantities come from the same OpenCV
contour (``cv2.contourArea`` and ``cv2.arcLength``), so the derived
perimeter is fully consistent with the project's locked measurement
convention.

Why not re-run inference?
    Prediction masks are not stored on disk.  Re-running inference would
    require the Herlev raw images and the GPU checkpoints; this is allowed
    only if existing data are insufficient.  Since area and circularity are
    both available, the algebraic inversion is preferred: it avoids all
    new model passes and produces numerically identical perimeter values.

Output files (written to ``results/circularity_mechanism/``)
-------------------------------------------------------------
real_prediction_area_perimeter.csv
    Per-row enriched table (all conditions, all cell_id/seed/model).
real_prediction_deltas.csv
    Clean→noise deltas for every cell_id/seed/model/severity triple.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FINAL_ANALYSIS_DIR = RESULTS_DIR / "final_analysis"
OUT_DIR = RESULTS_DIR / "circularity_mechanism"

DEGRADATION_CSV = FINAL_ANALYSIS_DIR / "degradation_per_cell_predictions.csv"
SPLIT_FILE = ROOT / "data" / "splits" / "original_herlev_group_split.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RE_EPSILON = 1e-6  # matches src/utils/config.py


# ---------------------------------------------------------------------------
# Perimeter back-calculation
# ---------------------------------------------------------------------------

def derive_perimeter(area: float, circularity: float) -> float:
    """Derive perimeter from area and circularity using the locked formula.

    Formula: circularity = 4 * pi * area / perimeter^2
    => perimeter = sqrt(4 * pi * area / circularity)

    Returns NaN if circularity <= 0 or area <= 0 (invalid cell).
    """
    if circularity <= 0 or area <= 0:
        return float("nan")
    return math.sqrt(4.0 * math.pi * area / circularity)


def relative_error(predicted: float, reference: float, eps: float = RE_EPSILON) -> float:
    """Relative error = |pred - ref| / (|ref| + eps)."""
    return abs(predicted - reference) / (abs(reference) + eps)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split_ids() -> dict[str, list[str]]:
    """Return {split_name: [cell_id, ...]} from the original Herlev split."""
    data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    return {
        "unet_train": data["unet_train"],
        "unet_val": data["unet_val"],
        "calibration": data["calibration"],
        "test": data["test"],
    }


def load_degradation_csv() -> pd.DataFrame:
    """Load the existing per-cell predictions CSV."""
    df = pd.read_csv(DEGRADATION_CSV, dtype={"severity": str})
    return df


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def enrich_with_perimeter(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived perimeter columns and relative errors to the data frame.

    Adds columns:
        true_perimeter          — from true_nucleus_area + true_circularity
        predicted_perimeter     — from predicted_nucleus_area + predicted_circularity
        perimeter_absolute_error
        perimeter_relative_error
        area_relative_error
        nc_relative_error       (already present as nc_relative_error)
        circularity_relative_error  — new
    """
    df = df.copy()

    # True perimeter
    df["true_perimeter"] = df.apply(
        lambda r: derive_perimeter(r["true_nucleus_area"], r["true_circularity"]),
        axis=1,
    )

    # Predicted perimeter (invalid if predicted_circularity is NaN or 0)
    df["predicted_perimeter"] = df.apply(
        lambda r: (
            derive_perimeter(r["predicted_nucleus_area"], r["predicted_circularity"])
            if not r.get("invalid_circularity", False)
            else float("nan")
        ),
        axis=1,
    )

    # Perimeter absolute / relative error
    df["perimeter_absolute_error"] = (df["predicted_perimeter"] - df["true_perimeter"]).abs()
    df["perimeter_relative_error"] = df.apply(
        lambda r: relative_error(r["predicted_perimeter"], r["true_perimeter"]),
        axis=1,
    )

    # Area relative error
    df["area_relative_error"] = df.apply(
        lambda r: relative_error(r["predicted_nucleus_area"], r["true_nucleus_area"]),
        axis=1,
    )

    # Circularity relative error (|pred - true| / |true| + eps)
    df["circularity_relative_error"] = df.apply(
        lambda r: (
            relative_error(r["predicted_circularity"], r["true_circularity"])
            if not r.get("invalid_circularity", False)
            else float("inf")
        ),
        axis=1,
    )

    return df


# ---------------------------------------------------------------------------
# Delta computation (clean → noise)
# ---------------------------------------------------------------------------

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute clean→noise metric deltas per cell_id/seed/model/severity.

    Returns a DataFrame with columns:
        cell_id, seed, model, degradation, severity,
        delta_area_error, delta_perimeter_error,
        delta_circularity_error, delta_nc_error, delta_dice
        (and relative variants)
    """
    # Separate clean rows
    clean_df = df[df["degradation"] == "clean"].copy()
    noise_df = df[(df["degradation"] == "gaussian_noise")].copy()

    # Key for merging
    merge_keys = ["split", "cell_id", "class", "seed", "model", "lambda_nc"]

    # Rename clean columns for merge
    clean_cols = {
        "nucleus_area_absolute_error": "clean_area_abs_err",
        "perimeter_absolute_error": "clean_perim_abs_err",
        "perimeter_relative_error": "clean_perim_rel_err",
        "area_relative_error": "clean_area_rel_err",
        "circularity_absolute_error": "clean_circ_abs_err",
        "circularity_relative_error": "clean_circ_rel_err",
        "nc_absolute_error": "clean_nc_abs_err",
        "nc_relative_error": "clean_nc_rel_err",
        "foreground_dice": "clean_fg_dice",
        "nucleus_dice": "clean_nuc_dice",
        "true_perimeter": "true_perimeter",
        "predicted_perimeter": "clean_pred_perimeter",
        "true_nucleus_area": "true_nucleus_area",
        "predicted_nucleus_area": "clean_pred_nucleus_area",
        "true_circularity": "true_circularity",
        "predicted_circularity": "clean_pred_circularity",
        "true_nc": "true_nc",
        "predicted_nc": "clean_pred_nc",
    }
    clean_sub = clean_df[merge_keys + list(clean_cols.keys())].rename(columns=clean_cols)

    # Merge noise rows with clean baseline
    merged = noise_df.merge(clean_sub, on=merge_keys, how="inner")

    # Delta = degraded_value - clean_value
    merged["delta_area_abs_err"] = (
        merged["nucleus_area_absolute_error"] - merged["clean_area_abs_err"]
    )
    merged["delta_area_rel_err"] = (
        merged["area_relative_error"] - merged["clean_area_rel_err"]
    )
    merged["delta_perim_abs_err"] = (
        merged["perimeter_absolute_error"] - merged["clean_perim_abs_err"]
    )
    merged["delta_perim_rel_err"] = (
        merged["perimeter_relative_error"] - merged["clean_perim_rel_err"]
    )
    merged["delta_circ_abs_err"] = (
        merged["circularity_absolute_error"] - merged["clean_circ_abs_err"]
    )
    merged["delta_circ_rel_err"] = (
        merged["circularity_relative_error"] - merged["clean_circ_rel_err"]
    )
    merged["delta_nc_abs_err"] = (
        merged["nc_absolute_error"] - merged["clean_nc_abs_err"]
    )
    merged["delta_nc_rel_err"] = (
        merged["nc_relative_error"] - merged["clean_nc_rel_err"]
    )
    merged["delta_fg_dice"] = merged["foreground_dice"] - merged["clean_fg_dice"]
    merged["delta_nuc_dice"] = merged["nucleus_dice"] - merged["clean_nuc_dice"]

    # Select output columns
    out_cols = (
        merge_keys
        + ["degradation", "severity"]
        + [
            "true_nucleus_area", "true_perimeter", "true_circularity", "true_nc",
            "clean_pred_nucleus_area", "clean_pred_perimeter",
            "clean_pred_circularity", "clean_pred_nc",
            "clean_area_abs_err", "clean_perim_abs_err", "clean_circ_abs_err", "clean_nc_abs_err",
            "nucleus_area_absolute_error", "perimeter_absolute_error",
            "circularity_absolute_error", "nc_absolute_error",
            "delta_area_abs_err", "delta_area_rel_err",
            "delta_perim_abs_err", "delta_perim_rel_err",
            "delta_circ_abs_err", "delta_circ_rel_err",
            "delta_nc_abs_err", "delta_nc_rel_err",
            "delta_fg_dice", "delta_nuc_dice",
            "invalid_circularity", "invalid_nc",
        ]
    )
    return merged[[c for c in out_cols if c in merged.columns]]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(out_dir: Path = OUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the area–perimeter decomposition and write outputs.

    Returns (enriched_df, deltas_df).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading degradation predictions CSV ...")
    df = load_degradation_csv()
    print(f"  {len(df):,} rows loaded.")

    print("Enriching with derived perimeter and relative errors ...")
    enriched = enrich_with_perimeter(df)

    # Filter to noise-only + clean conditions (exclude blur for the main analysis)
    ap_df = enriched[
        (enriched["degradation"].isin(["clean", "gaussian_noise"]))
    ].copy()

    out_ap = out_dir / "real_prediction_area_perimeter.csv"
    ap_df.to_csv(out_ap, index=False)
    print(f"  Written: {out_ap}  ({len(ap_df):,} rows)")

    print("Computing clean-->noise deltas ...")
    deltas = compute_deltas(enriched)

    out_deltas = out_dir / "real_prediction_deltas.csv"
    deltas.to_csv(out_deltas, index=False)
    print(f"  Written: {out_deltas}  ({len(deltas):,} rows)")

    return ap_df, deltas


if __name__ == "__main__":
    run()
