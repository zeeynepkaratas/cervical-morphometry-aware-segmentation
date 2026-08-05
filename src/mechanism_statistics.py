"""Statistical analysis for the circularity mechanism study.

Computes:
    1. Spearman correlations between delta metrics
    2. Cell-ID-clustered bootstrap 95% CIs (5000 repeats)
    3. Cell-level aggregated OLS regression (target = delta_circ_abs_err)
    4. Stratified breakdowns by seed, model, severity

All outputs written to ``results/circularity_mechanism/``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "circularity_mechanism"
BOOTSTRAP_SEED = 20260806
BOOTSTRAP_REPS = 5_000
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Clustered bootstrap helpers
# ---------------------------------------------------------------------------

from joblib import Parallel, delayed

def _cluster_bootstrap_mean(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    n_reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
    ci: float = 0.95,
) -> dict:
    """Bootstrap the mean of ``values`` by resampling whole clusters.

    All rows with the same ``cluster_id`` are treated as a single unit
    (i.e., they are always sampled together or not at all).

    Returns dict with keys: mean, ci_lo, ci_hi, n_clusters, n_obs.
    """
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    # Build mapping cluster → row indices
    cluster_map: dict = {}
    for cid in unique_clusters:
        cluster_map[cid] = np.where(cluster_ids == cid)[0]

    def _one_rep(_):
        sampled = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_map[c] for c in sampled])
        return np.nanmean(values[idx])

    boot_means = np.array(Parallel(n_jobs=-1, prefer="threads")(
        delayed(_one_rep)(i) for i in range(n_reps)
    ))

    alpha = 1.0 - ci
    return {
        "mean": float(np.nanmean(values)),
        "ci_lo": float(np.nanpercentile(boot_means, 100 * alpha / 2)),
        "ci_hi": float(np.nanpercentile(boot_means, 100 * (1 - alpha / 2))),
        "n_clusters": n_clusters,
        "n_obs": len(values),
    }


def _spearman_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
    n_reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Spearman correlation with clustered bootstrap CI."""
    rng = np.random.default_rng(seed)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, cluster_ids = x[finite], y[finite], cluster_ids[finite]

    rho, pval = stats.spearmanr(x, y)

    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    cluster_map = {cid: np.where(cluster_ids == cid)[0] for cid in unique_clusters}

    def _one_rep_spearman(_):
        sampled = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_map[c] for c in sampled])
        try:
            return stats.spearmanr(x[idx], y[idx]).statistic
        except Exception:
            return float("nan")

    boot_rhos = np.array(Parallel(n_jobs=-1, prefer="threads")(
        delayed(_one_rep_spearman)(i) for i in range(n_reps)
    ))

    return {
        "spearman_rho": float(rho),
        "p_value": float(pval),
        "ci_lo": float(np.nanpercentile(boot_rhos, 2.5)),
        "ci_hi": float(np.nanpercentile(boot_rhos, 97.5)),
        "n_clusters": n_clusters,
        "n_obs": int(len(x)),
    }


# ---------------------------------------------------------------------------
# Cell-level aggregation for regression
# ---------------------------------------------------------------------------

def _cell_level_means(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Aggregate to one row per cell_id by taking nanmean across seeds/severities."""
    return df.groupby("cell_id")[cols].apply(
        lambda g: g.apply(lambda s: np.nanmean(s[np.isfinite(s)]))
    ).reset_index()


# ---------------------------------------------------------------------------
# OLS regression helpers (no sklearn required)
# ---------------------------------------------------------------------------

def _ols_standardised(
    y: np.ndarray, X: np.ndarray, feature_names: list[str]
) -> list[dict]:
    """Standardised OLS coefficients via numpy.

    Standardises both y and each column of X to zero mean / unit variance,
    then fits the normal equations.  Returns list of dicts with
    {feature, std_coef, r2}.
    """
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[finite], X[finite]

    y_s = (y - y.mean()) / (y.std() + 1e-12)
    X_s = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    # Normal equations: coefficients = (X'X)^-1 X'y
    try:
        coefs = np.linalg.lstsq(X_s, y_s, rcond=None)[0]
    except np.linalg.LinAlgError:
        coefs = np.full(X_s.shape[1], float("nan"))

    y_pred = X_s @ coefs
    ss_res = np.sum((y_s - y_pred) ** 2)
    ss_tot = np.sum((y_s - y_s.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return [
        {"feature": name, "std_coef": float(c), "r2_full_model": float(r2)}
        for name, c in zip(feature_names, coefs)
    ]


# ---------------------------------------------------------------------------
# Main analysis functions
# ---------------------------------------------------------------------------

def compute_correlations(deltas: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations between delta metrics, with clustered bootstrap CIs.

    Pairs analysed:
        1. delta_circ_abs_err  ↔  delta_perim_abs_err
        2. delta_circ_abs_err  ↔  delta_area_abs_err
        3. delta_nc_abs_err    ↔  delta_area_abs_err
        4. delta_nc_abs_err    ↔  delta_perim_abs_err
        (all severity levels pooled; then also per-seed breakdown)
    """
    # Keep only valid rows
    df = deltas[~deltas["invalid_circularity"].fillna(False)].copy()
    df = df[~df["invalid_nc"].fillna(False)].copy()
    cluster_ids = df["cell_id"].values

    pairs = [
        ("delta_circ_abs_err", "delta_perim_abs_err", "circ_vs_perim"),
        ("delta_circ_abs_err", "delta_area_abs_err",  "circ_vs_area"),
        ("delta_nc_abs_err",   "delta_area_abs_err",  "nc_vs_area"),
        ("delta_nc_abs_err",   "delta_perim_abs_err", "nc_vs_perim"),
        ("delta_perim_abs_err","delta_area_abs_err",  "perim_vs_area"),
    ]

    rows = []
    for x_col, y_col, label in pairs:
        for scope, sub_df in _scope_splits(df):
            r = _spearman_with_ci(
                sub_df[x_col].values,
                sub_df[y_col].values,
                sub_df["cell_id"].values,
            )
            rows.append({"pair": label, "scope": scope, **r})

    return pd.DataFrame(rows)


def _scope_splits(df: pd.DataFrame):
    """Yield (scope_label, sub_dataframe) for pooled, per-seed, per-model, per-severity."""
    yield "all", df
    for seed in sorted(df["seed"].unique()):
        yield f"seed_{seed}", df[df["seed"] == seed]
    for model in sorted(df["model"].unique()):
        yield f"model_{model}", df[df["model"] == model]
    for sev in sorted(df["severity"].unique()):
        yield f"severity_{sev}", df[df["severity"] == sev]


def compute_regression(deltas: pd.DataFrame) -> pd.DataFrame:
    """Cell-level aggregated OLS: target = delta_circ_abs_err.

    Predictors: delta_area_abs_err, delta_perim_abs_err, delta_fg_dice.
    Run on cell-aggregated means (one row per cell_id).
    """
    df = deltas[~deltas["invalid_circularity"].fillna(False)].copy()
    df = df[~deltas["invalid_nc"].fillna(False)].copy()

    feature_cols = ["delta_area_abs_err", "delta_perim_abs_err", "delta_fg_dice"]
    target_col = "delta_circ_abs_err"

    rows = []
    for scope, sub_df in _scope_splits(df):
        agg = _cell_level_means(sub_df, [target_col] + feature_cols)
        y = agg[target_col].values
        X = agg[feature_cols].values
        res = _ols_standardised(y, X, feature_cols)
        for r in res:
            rows.append({"scope": scope, **r})

    return pd.DataFrame(rows)


def compute_bootstrap_intervals(deltas: pd.DataFrame) -> pd.DataFrame:
    """Clustered bootstrap 95% CIs for key delta quantities."""
    df = deltas[~deltas["invalid_circularity"].fillna(False)].copy()
    df = df[~deltas["invalid_nc"].fillna(False)].copy()

    metrics = [
        ("delta_circ_abs_err",  "noise_circularity_error_change"),
        ("delta_perim_abs_err", "noise_perimeter_error_change"),
        ("delta_area_abs_err",  "noise_area_error_change"),
        ("delta_nc_abs_err",    "noise_nc_error_change"),
        ("delta_fg_dice",       "noise_dice_change"),
    ]

    # Filter to gaussian noise only
    noise_df = df[df["degradation"] == "gaussian_noise"].copy()

    rows = []
    for col, label in metrics:
        for scope, sub_df in _scope_splits(noise_df):
            r = _cluster_bootstrap_mean(
                sub_df[col].values,
                sub_df["cell_id"].values,
            )
            rows.append({"metric": label, "scope": scope, **r})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(deltas_path: Path | None = None, out_dir: Path = OUT_DIR) -> None:
    """Run all statistical analyses and write CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if deltas_path is None:
        deltas_path = out_dir / "real_prediction_deltas.csv"

    print(f"Loading deltas from {deltas_path} ...")
    deltas = pd.read_csv(deltas_path, dtype={"severity": str})
    print(f"  {len(deltas):,} rows.")

    # Correlations
    print("Computing Spearman correlations ...")
    corr_df = compute_correlations(deltas)
    corr_path = out_dir / "area_perimeter_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"  Written: {corr_path}")

    # Regression
    print("Computing cell-level regression ...")
    reg_df = compute_regression(deltas)
    reg_path = out_dir / "area_perimeter_regression.csv"
    reg_df.to_csv(reg_path, index=False)
    print(f"  Written: {reg_path}")

    # Bootstrap intervals
    print("Computing clustered bootstrap intervals ...")
    boot_df = compute_bootstrap_intervals(deltas)
    boot_path = out_dir / "area_perimeter_bootstrap.csv"
    boot_df.to_csv(boot_path, index=False)
    print(f"  Written: {boot_path}")


if __name__ == "__main__":
    run()
