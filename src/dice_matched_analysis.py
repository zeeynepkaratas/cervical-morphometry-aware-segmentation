"""Dice-matched perturbation analysis — Phase B continuation.

Groups perturbation results by Dice band and compares circularity vs N/C
changes across perturbation families at matched Dice loss.

Output: results/circularity_mechanism/dice_matched_perturbation_analysis.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "circularity_mechanism"

# Dice bands as specified in the protocol
DICE_BANDS = {
    "mild":     (0.95, 0.98),
    "moderate": (0.90, 0.95),
}

# Perturbation families for grouping
BOUNDARY_TYPES = {"jitter", "protrusion", "indentation"}
SHAPE_TYPES = {"erosion", "dilation"}


def classify_perturbation(pert: str) -> str:
    if pert in BOUNDARY_TYPES:
        return "boundary"
    return "shape"


def run(
    perturbations_path: Path | None = None,
    out_dir: Path = OUT_DIR,
) -> pd.DataFrame:
    """Run dice-matched analysis and write output CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if perturbations_path is None:
        perturbations_path = out_dir / "controlled_mask_perturbations.csv"

    print(f"Loading perturbation results from {perturbations_path} ...")
    df = pd.read_csv(perturbations_path)

    # Filter valid rows only
    valid_df = df[df["valid"] == True].copy()
    valid_df["perturbation_family"] = valid_df["perturbation"].map(classify_perturbation)

    rows = []

    for band_name, (lo, hi) in DICE_BANDS.items():
        band_df = valid_df[
            (valid_df["nucleus_dice"] >= lo) & (valid_df["nucleus_dice"] < hi)
        ].copy()

        if len(band_df) == 0:
            print(f"  WARNING: No perturbations in Dice band '{band_name}' ({lo}–{hi}).")
            # Record that the band was attempted but yielded no results
            rows.append({
                "dice_band": band_name,
                "dice_lo": lo,
                "dice_hi": hi,
                "perturbation_type": "all",
                "perturbation_family": "all",
                "n_cells": 0,
                "n_perturbation_instances": 0,
                "mean_nucleus_dice": float("nan"),
                "mean_circularity_change": float("nan"),
                "mean_abs_circularity_change": float("nan"),
                "median_circularity_change": float("nan"),
                "mean_nc_change": float("nan"),
                "mean_abs_nc_change": float("nan"),
                "median_nc_change": float("nan"),
                "mean_area_change": float("nan"),
                "mean_perimeter_change": float("nan"),
                "achieved_dice_range": f"{lo:.2f}–{hi:.2f}",
                "band_populated": False,
            })
            continue

        # Per perturbation type within band
        for pert_type in sorted(valid_df["perturbation"].unique()):
            sub = band_df[band_df["perturbation"] == pert_type]
            if len(sub) == 0:
                continue

            family = classify_perturbation(pert_type)
            rows.append({
                "dice_band": band_name,
                "dice_lo": lo,
                "dice_hi": hi,
                "perturbation_type": pert_type,
                "perturbation_family": family,
                "n_cells": sub["cell_id"].nunique(),
                "n_perturbation_instances": len(sub),
                "mean_nucleus_dice": float(sub["nucleus_dice"].mean()),
                "mean_circularity_change": float(sub["circularity_change"].mean()),
                "mean_abs_circularity_change": float(sub["circularity_change"].abs().mean()),
                "median_circularity_change": float(sub["circularity_change"].median()),
                "std_circularity_change": float(sub["circularity_change"].std()),
                "mean_nc_change": float(sub["nc_change"].mean()),
                "mean_abs_nc_change": float(sub["nc_change"].abs().mean()),
                "median_nc_change": float(sub["nc_change"].median()),
                "std_nc_change": float(sub["nc_change"].std()),
                "mean_area_change": float(sub["area_change"].mean()),
                "mean_area_change_rel": float(sub["area_change_rel"].mean()),
                "mean_perimeter_change": float(sub["perimeter_change"].mean()),
                "mean_perimeter_change_rel": float(sub["perimeter_change_rel"].mean()),
                "achieved_dice_min": float(sub["nucleus_dice"].min()),
                "achieved_dice_max": float(sub["nucleus_dice"].max()),
                "band_populated": True,
            })

        # Summary by family within band
        for family_name, family_set in [("boundary", BOUNDARY_TYPES), ("shape", SHAPE_TYPES)]:
            sub = band_df[band_df["perturbation"].isin(family_set)]
            if len(sub) == 0:
                continue
            rows.append({
                "dice_band": band_name,
                "dice_lo": lo,
                "dice_hi": hi,
                "perturbation_type": f"ALL_{family_name.upper()}",
                "perturbation_family": family_name,
                "n_cells": sub["cell_id"].nunique(),
                "n_perturbation_instances": len(sub),
                "mean_nucleus_dice": float(sub["nucleus_dice"].mean()),
                "mean_circularity_change": float(sub["circularity_change"].mean()),
                "mean_abs_circularity_change": float(sub["circularity_change"].abs().mean()),
                "median_circularity_change": float(sub["circularity_change"].median()),
                "std_circularity_change": float(sub["circularity_change"].std()),
                "mean_nc_change": float(sub["nc_change"].mean()),
                "mean_abs_nc_change": float(sub["nc_change"].abs().mean()),
                "median_nc_change": float(sub["nc_change"].median()),
                "std_nc_change": float(sub["nc_change"].std()),
                "mean_area_change": float(sub["area_change"].mean()),
                "mean_area_change_rel": float(sub["area_change_rel"].mean()),
                "mean_perimeter_change": float(sub["perimeter_change"].mean()),
                "mean_perimeter_change_rel": float(sub["perimeter_change_rel"].mean()),
                "achieved_dice_min": float(sub["nucleus_dice"].min()),
                "achieved_dice_max": float(sub["nucleus_dice"].max()),
                "band_populated": True,
            })

        # Report the actual Dice distribution for this band
        print(f"  Band '{band_name}' ({lo}–{hi}): {len(band_df)} valid instances "
              f"from {band_df['cell_id'].nunique()} cells.")

    result_df = pd.DataFrame(rows)
    out_path = out_dir / "dice_matched_perturbation_analysis.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Written: {out_path}  ({len(result_df)} rows)")
    return result_df


if __name__ == "__main__":
    run()
