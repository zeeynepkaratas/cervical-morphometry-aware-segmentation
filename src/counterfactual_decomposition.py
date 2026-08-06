import pandas as pd
import numpy as np

def bootstrap_perimeter_dominant_fraction(dominant_component_array, cell_id_array, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    unique_cells = np.unique(cell_id_array)
    n_cells = len(unique_cells)
    
    # Precompute indices
    cell_to_idx = {cid: np.where(cell_id_array == cid)[0] for cid in unique_cells}
    is_perimeter = (dominant_component_array == "perimeter")
    
    boot_fractions = np.zeros(n_boot)
    for i in range(n_boot):
        boot_cells = rng.choice(unique_cells, size=n_cells, replace=True)
        boot_idx = np.concatenate([cell_to_idx[cid] for cid in boot_cells])
        boot_fractions[i] = np.mean(is_perimeter[boot_idx])
        
    return np.percentile(boot_fractions, [2.5, 97.5])

def run_decomposition():
    df = pd.read_csv("results/circularity_mechanism/real_prediction_area_perimeter.csv")
    input_row_count = len(df)
    
    # Filter to valid predictions
    valid_df = df[(df["predicted_nucleus_area"] > 0) & (df["direct_predicted_perimeter"] > 0)].copy()
    excluded_count = input_row_count - len(valid_df)
    
    valid_df["raw_direct_circularity"] = (4 * np.pi * valid_df["predicted_nucleus_area"]) / (valid_df["direct_predicted_perimeter"] ** 2)
    
    clean_df = valid_df[valid_df["severity"] == "clean"].copy()
    corr_df = valid_df[valid_df["severity"] != "clean"].copy()
    
    # Audit 1: Clean Key Uniqueness
    clean_keys = clean_df["cell_id"].astype(str) + "_" + clean_df["model"].astype(str) + "_" + clean_df["seed"].astype(str)
    if clean_keys.duplicated().any():
        raise ValueError("Duplicate keys found in clean predictions.")
        
    # Audit 2: Corrupted Key Uniqueness
    corr_keys = corr_df["cell_id"].astype(str) + "_" + corr_df["model"].astype(str) + "_" + corr_df["seed"].astype(str) + "_" + corr_df["degradation"].astype(str) + "_" + corr_df["severity"].astype(str)
    if corr_keys.duplicated().any():
        raise ValueError("Duplicate keys found in corrupted predictions.")
    
    merged = pd.merge(
        corr_df, clean_df, 
        on=["cell_id", "model", "seed"], 
        suffixes=("_corr", "_clean")
    )
    
    matched_pairs = len(merged)
    
    merged["delta_log_circularity"] = np.log(merged["raw_direct_circularity_corr"] / merged["raw_direct_circularity_clean"])
    merged["area_contribution"] = np.log(merged["predicted_nucleus_area_corr"] / merged["predicted_nucleus_area_clean"])
    merged["perimeter_contribution"] = -2 * np.log(merged["direct_predicted_perimeter_corr"] / merged["direct_predicted_perimeter_clean"])
    
    merged["identity_residual"] = merged["delta_log_circularity"] - (merged["area_contribution"] + merged["perimeter_contribution"])
    
    max_res = merged["identity_residual"].abs().max()
    if max_res > 1e-9:
        raise ValueError(f"Identity residual {max_res} exceeds 1e-9 tolerance.")
    
    merged["abs_area_contribution"] = merged["area_contribution"].abs()
    merged["abs_perimeter_contribution"] = merged["perimeter_contribution"].abs()
    merged["dominant_component"] = np.where(merged["abs_perimeter_contribution"] > merged["abs_area_contribution"], "perimeter", "area")
    
    out_cols = [
        "cell_id", "model", "seed", "severity_corr", "degradation_corr",
        "raw_direct_circularity_clean", "raw_direct_circularity_corr",
        "delta_log_circularity", "area_contribution", "perimeter_contribution",
        "abs_area_contribution", "abs_perimeter_contribution",
        "dominant_component", "identity_residual"
    ]
    merged[out_cols].to_csv("results/circularity_mechanism/log_decomposition_per_cell.csv", index=False)
    
    # Calculate clustered CI for the overall metric
    ci_lower, ci_upper = bootstrap_perimeter_dominant_fraction(
        merged["dominant_component"].values,
        merged["cell_id"].values
    )
    
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
            "mean_perimeter_minus_area": (group["abs_perimeter_contribution"] - group["abs_area_contribution"]).mean(),
            "perimeter_dominant_fraction": (group["dominant_component"] == "perimeter").mean(),
            "max_identity_residual": group["identity_residual"].abs().max()
        }

    overall_dict = summarize_group(merged, "Overall")
    overall_dict["perimeter_dominant_fraction_ci_lower"] = ci_lower
    overall_dict["perimeter_dominant_fraction_ci_upper"] = ci_upper
    summary_list.append(overall_dict)
    
    for (deg, sev), group in merged.groupby(["degradation_corr", "severity_corr"]):
        summary_list.append(summarize_group(group, deg, sev=sev))
        
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv("results/circularity_mechanism/log_decomposition_summary.csv", index=False)

    # Output audit metrics
    print(f"Input rows: {input_row_count}")
    print(f"Excluded non-finite: {excluded_count}")
    print(f"Valid clean rows: {len(clean_df)}")
    print(f"Valid corrupted rows: {len(corr_df)}")
    print(f"Matched pairs: {matched_pairs}")
    print(f"Max identity residual: {max_res}")
    print(f"Overall perimeter dominant fraction: {overall_dict['perimeter_dominant_fraction']:.4f} CI:[{ci_lower:.4f}, {ci_upper:.4f}]")

if __name__ == "__main__":
    run_decomposition()
