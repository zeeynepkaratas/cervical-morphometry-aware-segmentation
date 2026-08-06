import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.counterfactual_decomposition import run_decomposition

OUT_DIR = Path("results/circularity_mechanism")

def run_counterfactual_comparison():
    print("Running decomposition...")
    run_decomposition()
    print("Decomposition completed.")
    
    print("Loading perturbation results...")
    df = pd.read_csv(OUT_DIR / "controlled_mask_perturbations.csv")
    
    print(f"Total perturbation instances loaded: {len(df)}")
    
    # Filter only valid
    valid_df = df[df["valid"] == True].copy()
    print(f"Valid perturbation instances: {len(valid_df)}")
    
    # Restrict to Mild Dice band
    band_df = valid_df[(valid_df["nucleus_dice"] >= 0.95) & (valid_df["nucleus_dice"] <= 0.98)].copy()
    print(f"Instances in 0.95-0.98 Dice band: {len(band_df)}")
    
    # Group by perturbation types
    shape_df = band_df[band_df["perturbation"].isin(["erosion", "dilation"])]
    counter_df = band_df[band_df["perturbation"] == "area_preserving_boundary"]
    
    # Need overlapping cells to do a paired-like comparison
    common_cells = set(shape_df["cell_id"]).intersection(set(counter_df["cell_id"]))
    print(f"Cells with both shape and counterfactual in the dice band: {len(common_cells)}")
    
    if len(common_cells) < 10:
        print("WARNING: Insufficient cells for meaningful comparison.")
        
    shape_matched = shape_df[shape_df["cell_id"].isin(common_cells)].copy()
    counter_matched = counter_df[counter_df["cell_id"].isin(common_cells)].copy()
    
    shape_matched.to_csv(OUT_DIR / "area_preserving_counterfactual_per_cell_shape.csv", index=False)
    counter_matched.to_csv(OUT_DIR / "area_preserving_counterfactual_per_cell_boundary.csv", index=False)
    
    # Summarize Mean absolute circularity change
    shape_matched["abs_circ_change"] = shape_matched["circularity_change"].abs()
    counter_matched["abs_circ_change"] = counter_matched["circularity_change"].abs()
    
    shape_matched["abs_nc_change"] = shape_matched["nc_change"].abs()
    counter_matched["abs_nc_change"] = counter_matched["nc_change"].abs()
    
    shape_mean = shape_matched["abs_circ_change"].mean()
    counter_mean = counter_matched["abs_circ_change"].mean()
    
    shape_nc = shape_matched["abs_nc_change"].mean()
    counter_nc = counter_matched["abs_nc_change"].mean()
    
    diff = counter_mean - shape_mean
    
    print(f"Shape abs_circ_change: {shape_mean:.5f}")
    print(f"Counterfactual abs_circ_change: {counter_mean:.5f}")
    print(f"Difference: {diff:.5f}")
    
    summary = {
        "n_cells": len(common_cells),
        "shape_instances": len(shape_matched),
        "counterfactual_instances": len(counter_matched),
        "mean_abs_circ_shape": float(shape_mean),
        "mean_abs_circ_counterfactual": float(counter_mean),
        "diff_abs_circ": float(diff),
        "mean_abs_nc_shape": float(shape_nc),
        "mean_abs_nc_counterfactual": float(counter_nc)
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "area_preserving_counterfactual_summary.csv", index=False)
    
    # Interpretation
    if len(common_cells) > 20 and diff > 0.05 and counter_nc < 0.05:
        decision = "STRONG_ADDITIONAL_SUPPORT"
    elif diff > 0:
        decision = "PARTIAL_ADDITIONAL_SUPPORT"
    else:
        decision = "ADDITIONAL_SUPPORT_NOT_OBSERVED"
        
    interp = {
        "decision": decision,
        "reasoning": f"Counterfactual difference was {diff:.4f} across {len(common_cells)} matched cells.",
        "metrics": summary
    }
    
    with open(OUT_DIR / "counterfactual_interpretation.json", "w") as f:
        json.dump(interp, f, indent=2)
        
    print(f"Decision: {decision}")

if __name__ == "__main__":
    run_counterfactual_comparison()
