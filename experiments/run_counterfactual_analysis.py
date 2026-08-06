import pandas as pd
import numpy as np
import json
import sys
import hashlib
import platform
import cv2
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.counterfactual_decomposition import run_decomposition

OUT_DIR = Path("results/circularity_mechanism")

def bootstrap_cell_means(cell_differences, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n_cells = len(cell_differences)
    if n_cells == 0:
        return 0, 0
    # Resample the 1D array of cell differences
    boot_samples = rng.choice(cell_differences, size=(n_boot, n_cells), replace=True)
    boot_means = np.mean(boot_samples, axis=1)
    return np.percentile(boot_means, [2.5, 97.5])

def get_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def run_counterfactual_comparison():
    print("Running decomposition...")
    run_decomposition()
    print("Decomposition completed.")
    
    print("Loading perturbation results...")
    input_file = OUT_DIR / "controlled_mask_perturbations.csv"
    df = pd.read_csv(input_file)
    
    # Eligibility filters based on Explicit Area
    # Requirements: pixel_count_change == 0 AND abs(contour_area_change_rel) <= 0.02
    is_cf = df["perturbation"] == "area_preserving_boundary"
    cf_eligible = (df["valid"] == True) & (df["pixel_count_change"] == 0) & (df["contour_area_change_rel"].abs() <= 0.02)
    
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
    
    common_cells = sorted(list(set(shape_df["cell_id"]).intersection(set(counter_df["cell_id"]))))
    
    severity_order = {"low": 1, "medium": 2, "high": 3, "r1": 1, "r2": 2}
    
    pairs = []
    unmatched_cf_count = 0
    
    for cid in common_cells:
        cell_shapes = shape_df[shape_df["cell_id"] == cid].copy()
        cell_cfs = counter_df[counter_df["cell_id"] == cid].copy()
        
        # Sort counterfactuals by severity order
        cell_cfs["sev_order"] = cell_cfs["severity"].map(severity_order)
        cell_cfs = cell_cfs.sort_values("sev_order").reset_index(drop=True)
        
        used_shape_indices = set()
        
        for _, cf_row in cell_cfs.iterrows():
            best_shape_idx = -1
            
            # Create comparable array of available shapes
            available_shapes = []
            for i, shape_row in cell_shapes.iterrows():
                if i in used_shape_indices:
                    continue
                diff = abs(cf_row["nucleus_dice"] - shape_row["nucleus_dice"])
                available_shapes.append({
                    "idx": i,
                    "diff": diff,
                    "pert": shape_row["perturbation"],
                    "sev": severity_order.get(shape_row["severity"], 99)
                })
            
            if not available_shapes:
                unmatched_cf_count += 1
                continue
                
            # Tie break rules: 1. min diff, 2. pert name alphabetical, 3. severity order
            available_shapes.sort(key=lambda x: (x["diff"], x["pert"], x["sev"]))
            
            best_shape_idx = available_shapes[0]["idx"]
            used_shape_indices.add(best_shape_idx)
            
            s_row = cell_shapes.loc[best_shape_idx]
            pairs.append({
                "cell_id": cid,
                "cf_severity": cf_row["severity"],
                "shape_perturbation": s_row["perturbation"],
                "shape_severity": s_row["severity"],
                "cf_dice": cf_row["nucleus_dice"],
                "shape_dice": s_row["nucleus_dice"],
                "absolute_dice_difference": available_shapes[0]["diff"],
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
        
    # Cell level aggregation
    cell_summary = pairs_df.groupby("cell_id")["circ_change_diff"].mean().reset_index()
    cell_summary.to_csv(OUT_DIR / "area_preserving_counterfactual_cell_summary.csv", index=False)
    
    # Bootstrap over cell means only
    cell_differences = cell_summary["circ_change_diff"].values
    n_boot = 10000
    boot_seed = 42
    ci_lower, ci_upper = bootstrap_cell_means(cell_differences, n_boot=n_boot, seed=boot_seed)
    
    mean_diff = np.mean(cell_differences)
    median_diff = np.median(cell_differences)
    pos_fraction = (cell_differences > 0).mean()
    
    summary = {
        "n_matched_pairs": len(pairs_df),
        "n_unique_cells": len(cell_differences),
        "unmatched_cf_instances": unmatched_cf_count,
        "mean_paired_circ_diff": mean_diff,
        "median_paired_circ_diff": median_diff,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "positive_diff_cell_fraction": pos_fraction,
        "mean_cf_abs_nc_change": pairs_df["cf_abs_nc_change"].mean()
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "area_preserving_counterfactual_summary.csv", index=False)
    
    # Pre-registered Decision Logic strictly using the newly cell-bootstrapped CI
    if ci_lower > 0 and summary["mean_cf_abs_nc_change"] < 0.02:
        decision = "STRONG_ADDITIONAL_SUPPORT"
    elif mean_diff > 0:
        decision = "PARTIAL_ADDITIONAL_SUPPORT"
    else:
        decision = "ADDITIONAL_SUPPORT_NOT_OBSERVED"
        
    interp = {
        "decision": decision,
        "reasoning": f"Cell-level bootstrap 95% CI for paired absolute circularity change difference was [{ci_lower:.5f}, {ci_upper:.5f}] with mean {mean_diff:.5f}.",
        "metrics": summary
    }
    
    with open(OUT_DIR / "counterfactual_interpretation.json", "w") as f:
        json.dump(interp, f, indent=2)
        
    print(f"Decision: {decision}")
    print(f"95% CI: [{ci_lower:.5f}, {ci_upper:.5f}]")
    
    # Manifest Generation
    import subprocess
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8").strip()
    except Exception:
        git_sha = "unknown"
        git_branch = "unknown"

    manifest = {
        "input_file": str(input_file.name),
        "input_sha256": get_file_sha256(input_file),
        "input_rows": len(df),
        "input_cols": len(df.columns),
        "git_branch": git_branch,
        "git_commit": git_sha,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "bootstrap_seed": boot_seed,
        "bootstrap_iterations": n_boot
    }
    
    with open(OUT_DIR / "counterfactual_evidence_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    run_counterfactual_comparison()
