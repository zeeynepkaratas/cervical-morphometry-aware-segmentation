import pandas as pd
import numpy as np

def run_decomposition():
    df = pd.read_csv("results/circularity_mechanism/real_prediction_area_perimeter.csv")
    
    # Exclude invalid predictions
    mask = (
        (df["predicted_nucleus_area"] > 0) & 
        (df["direct_predicted_perimeter"] > 0) & 
        (df["predicted_circularity"].notna())
    )
    df = df[mask].copy()

    # Calculate exact log-ratio decomposition
    df["delta_log_circularity"] = np.log(df["predicted_circularity"] / df["true_circularity"])
    df["area_contribution"] = np.log(df["predicted_nucleus_area"] / df["true_nucleus_area"])
    df["perimeter_contribution"] = -2 * np.log(df["direct_predicted_perimeter"] / df["direct_true_perimeter"])
    df["identity_residual"] = df["delta_log_circularity"] - (df["area_contribution"] + df["perimeter_contribution"])
    df["abs_area_contribution"] = df["area_contribution"].abs()
    df["abs_perimeter_contribution"] = df["perimeter_contribution"].abs()
    df["dominant_component"] = np.where(df["abs_perimeter_contribution"] > df["abs_area_contribution"], "perimeter", "area")

    df.to_csv("results/circularity_mechanism/log_decomposition_per_cell.csv", index=False)

    summary = {
        "mean_identity_residual": df["identity_residual"].mean(),
        "perimeter_dominant_fraction": (df["dominant_component"] == "perimeter").mean(),
        "mean_abs_perimeter_contribution": df["abs_perimeter_contribution"].mean(),
        "mean_abs_area_contribution": df["abs_area_contribution"].mean()
    }
    pd.DataFrame([summary]).to_csv("results/circularity_mechanism/log_decomposition_summary.csv", index=False)
    
if __name__ == "__main__":
    run_decomposition()
