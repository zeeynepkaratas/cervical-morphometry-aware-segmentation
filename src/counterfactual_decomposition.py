import pandas as pd
import numpy as np

def run_decomposition():
    df = pd.read_csv("results/circularity_mechanism/real_prediction_area_perimeter.csv")
    
    # Needs to match clean vs corrupted for the exact same cell_id, model, seed
    # Filter to valid predictions
    df = df[(df["predicted_nucleus_area"] > 0) & (df["direct_predicted_perimeter"] > 0)].copy()
    
    # Calculate unclipped raw direct circularity
    df["raw_direct_circularity"] = (4 * np.pi * df["predicted_nucleus_area"]) / (df["direct_predicted_perimeter"] ** 2)
    
    clean_df = df[df["severity"] == "clean"].copy()
    corr_df = df[df["severity"] != "clean"].copy()
    
    # Merge on cell_id, model, seed
    merged = pd.merge(
        corr_df, clean_df, 
        on=["cell_id", "model", "seed"], 
        suffixes=("_corr", "_clean")
    )
    
    # Exact Decomposition (Corrupted / Clean)
    merged["delta_log_circularity"] = np.log(merged["raw_direct_circularity_corr"] / merged["raw_direct_circularity_clean"])
    merged["area_contribution"] = np.log(merged["predicted_nucleus_area_corr"] / merged["predicted_nucleus_area_clean"])
    merged["perimeter_contribution"] = -2 * np.log(merged["direct_predicted_perimeter_corr"] / merged["direct_predicted_perimeter_clean"])
    
    merged["identity_residual"] = merged["delta_log_circularity"] - (merged["area_contribution"] + merged["perimeter_contribution"])
    
    merged["abs_area_contribution"] = merged["area_contribution"].abs()
    merged["abs_perimeter_contribution"] = merged["perimeter_contribution"].abs()
    merged["dominant_component"] = np.where(merged["abs_perimeter_contribution"] > merged["abs_area_contribution"], "perimeter", "area")
    
    # Output detailed CSV
    out_cols = [
        "cell_id", "model", "seed", "severity_corr", "degradation_corr",
        "raw_direct_circularity_clean", "raw_direct_circularity_corr",
        "delta_log_circularity", "area_contribution", "perimeter_contribution",
        "abs_area_contribution", "abs_perimeter_contribution",
        "dominant_component", "identity_residual"
    ]
    merged[out_cols].to_csv("results/circularity_mechanism/log_decomposition_per_cell.csv", index=False)
    
    # Generate summaries
    summary_list = []
    
    def summarize_group(group, name, sev="all", model="all", seed="all"):
        return {
            "group": name,
            "corruption": name,
            "severity": sev,
            "model": model,
            "seed": seed,
            "n_pairs": len(group),
            "mean_abs_area_contrib": group["abs_area_contribution"].mean(),
            "mean_abs_perimeter_contrib": group["abs_perimeter_contribution"].mean(),
            "median_abs_area_contrib": group["abs_area_contribution"].median(),
            "median_abs_perimeter_contrib": group["abs_perimeter_contribution"].median(),
            "mean_perimeter_minus_area": (group["abs_perimeter_contribution"] - group["abs_area_contribution"]).mean(),
            "perimeter_dominant_fraction": (group["dominant_component"] == "perimeter").mean(),
            "max_identity_residual": group["identity_residual"].abs().max()
        }

    summary_list.append(summarize_group(merged, "Overall"))
    
    for (deg, sev), group in merged.groupby(["degradation_corr", "severity_corr"]):
        summary_list.append(summarize_group(group, deg, sev=sev))
        
    for model, group in merged.groupby("model"):
        summary_list.append(summarize_group(group, "Overall", model=model))
        
    for seed, group in merged.groupby("seed"):
        summary_list.append(summarize_group(group, "Overall", seed=seed))
        
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv("results/circularity_mechanism/log_decomposition_summary.csv", index=False)

if __name__ == "__main__":
    run_decomposition()
