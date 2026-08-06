import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.counterfactual_decomposition import run_decomposition

OUT_DIR = Path("results/circularity_mechanism")

def clustered_bootstrap_ci(data, cell_ids, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    unique_cells = np.unique(cell_ids)
    n_cells = len(unique_cells)
    
    boot_means = np.zeros(n_boot)
    for i in range(n_boot):
        # Sample cell IDs with replacement
        boot_cells = rng.choice(unique_cells, size=n_cells, replace=True)
        
        # Build bootstrap sample by grabbing all instances for each sampled cell
        # This is slow if done naively, so we use indexing
        # Create a mapping from cell_id to indices
        boot_indices = []
        # Pre-compute indices per cell for faster lookup
        # (Assuming data and cell_ids are aligned arrays)
        pass # implemented efficiently below

    # Efficient implementation:
    cell_to_idx = {cid: np.where(cell_ids == cid)[0] for cid in unique_cells}
    for i in range(n_boot):
        boot_cells = rng.choice(unique_cells, size=n_cells, replace=True)
        boot_idx = np.concatenate([cell_to_idx[cid] for cid in boot_cells])
        boot_means[i] = np.mean(data[boot_idx])
        
    return np.percentile(boot_means, [2.5, 97.5])

def run_counterfactual_comparison():
    print("Running decomposition...")
    run_decomposition()
    print("Decomposition completed.")
    
    print("Loading perturbation results...")
    df = pd.read_csv(OUT_DIR / "controlled_mask_perturbations.csv")
    
    # Eligibility filters
    df["pixel_area_change"] = df["perturbed_area"] - df["gt_area"]
    # Contour area relative change (gt_area is pixel area, use it for denom for consistency with pixel preservation)
    # Actually, area_change_rel is already in the CSV and is contour area based. 
    # Let's ensure both are met for counterfactuals.
    
    is_cf = df["perturbation"] == "area_preserving_boundary"
    cf_eligible = (df["valid"] == True) & (df["pixel_area_change"] == 0) & (df["area_change_rel"].abs() <= 0.02)
    
    df.loc[is_cf, "valid"] = cf_eligible[is_cf]
    
    valid_df = df[df["valid"] == True].copy()
    
    # Save eligibility stats
    total_cf = len(df[is_cf])
    eligible_cf = len(df[is_cf & cf_eligible])
    pd.DataFrame([{
        "total_counterfactual_attempts": total_cf,
        "eligible_counterfactuals": eligible_cf,
        "eligibility_rate": eligible_cf / total_cf if total_cf > 0 else 0
    }]).to_csv(OUT_DIR / "counterfactual_eligibility.csv", index=False)
    
    # Restrict to Mild Dice band
    band_df = valid_df[(valid_df["nucleus_dice"] >= 0.95) & (valid_df["nucleus_dice"] <= 0.98)].copy()
    
    shape_df = band_df[band_df["perturbation"].isin(["erosion", "dilation"])]
    counter_df = band_df[band_df["perturbation"] == "area_preserving_boundary"]
    
    # Strict matching within cell
    common_cells = set(shape_df["cell_id"]).intersection(set(counter_df["cell_id"]))
    
    pairs = []
    for cid in common_cells:
        cell_shapes = shape_df[shape_df["cell_id"] == cid].copy()
        cell_cfs = counter_df[counter_df["cell_id"] == cid].copy()
        
        # Sort shape instances to allow deterministic matching
        cell_shapes = cell_shapes.sort_values(["perturbation", "severity"]).reset_index()
        used_shape_indices = set()
        
        for _, cf_row in cell_cfs.iterrows():
            best_shape_idx = -1
            min_dice_diff = float("inf")
            
            for i, shape_row in cell_shapes.iterrows():
                if i in used_shape_indices:
                    continue
                diff = abs(cf_row["nucleus_dice"] - shape_row["nucleus_dice"])
                if diff < min_dice_diff:
                    min_dice_diff = diff
                    best_shape_idx = i
                    
            if best_shape_idx != -1:
                used_shape_indices.add(best_shape_idx)
                s_row = cell_shapes.iloc[best_shape_idx]
                pairs.append({
                    "cell_id": cid,
                    "cf_severity": cf_row["severity"],
                    "shape_perturbation": s_row["perturbation"],
                    "shape_severity": s_row["severity"],
                    "cf_dice": cf_row["nucleus_dice"],
                    "shape_dice": s_row["nucleus_dice"],
                    "cf_abs_circ_change": abs(cf_row["circularity_change"]),
                    "shape_abs_circ_change": abs(s_row["circularity_change"]),
                    "circ_change_diff": abs(cf_row["circularity_change"]) - abs(s_row["circularity_change"]),
                    "cf_abs_nc_change": abs(cf_row["nc_change"]),
                    "shape_abs_nc_change": abs(s_row["nc_change"]),
                    "cf_perimeter_change": cf_row["perimeter_change"]
                })

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(OUT_DIR / "area_preserving_counterfactual_pairs.csv", index=False)
    
    if len(pairs_df) == 0:
        print("No matched pairs found.")
        return
        
    # Bootstrap
    diff_data = pairs_df["circ_change_diff"].values
    cell_ids = pairs_df["cell_id"].values
    ci_lower, ci_upper = clustered_bootstrap_ci(diff_data, cell_ids)
    
    mean_diff = np.mean(diff_data)
    median_diff = np.median(diff_data)
    
    # Cell level summary
    cell_summary = pairs_df.groupby("cell_id")["circ_change_diff"].mean().reset_index()
    cell_summary.to_csv(OUT_DIR / "area_preserving_counterfactual_cell_summary.csv", index=False)
    
    pos_fraction = (cell_summary["circ_change_diff"] > 0).mean()
    
    summary = {
        "n_matched_pairs": len(pairs_df),
        "n_unique_cells": len(pairs_df["cell_id"].unique()),
        "mean_paired_circ_diff": mean_diff,
        "median_paired_circ_diff": median_diff,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "positive_diff_cell_fraction": pos_fraction,
        "mean_cf_abs_nc_change": pairs_df["cf_abs_nc_change"].mean()
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "area_preserving_counterfactual_summary.csv", index=False)
    
    # Pre-registered Decision Logic
    if ci_lower > 0 and summary["mean_cf_abs_nc_change"] < 0.02:
        decision = "STRONG_ADDITIONAL_SUPPORT"
    elif mean_diff > 0:
        decision = "PARTIAL_ADDITIONAL_SUPPORT"
    else:
        decision = "ADDITIONAL_SUPPORT_NOT_OBSERVED"
        
    interp = {
        "decision": decision,
        "reasoning": f"Bootstrap 95% CI for paired absolute circularity change difference was [{ci_lower:.5f}, {ci_upper:.5f}] with mean {mean_diff:.5f}.",
        "metrics": summary
    }
    
    with open(OUT_DIR / "counterfactual_interpretation.json", "w") as f:
        json.dump(interp, f, indent=2)
        
    print(f"Decision: {decision}")
    print(f"95% CI: [{ci_lower:.5f}, {ci_upper:.5f}]")

if __name__ == "__main__":
    run_counterfactual_comparison()
