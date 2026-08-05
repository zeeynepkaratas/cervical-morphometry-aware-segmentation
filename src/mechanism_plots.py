"""Plots for the circularity mechanism analysis — Phase D.

Generates the 10 required figures:
    1.  Noise area error vs perimeter error change (scatter/density)
    2.  Perimeter error delta vs circularity error delta
    3.  Area error delta vs circularity error delta
    4.  Dice-matched perturbation types: circularity change
    5.  Dice-matched perturbation types: N/C change
    6.  Perturbation types: area/perimeter change summary
    7.  Raw vs post-processed circularity error
    8.  Raw vs post-processed Dice and N/C trade-off
    9.  Top-5 most-improved cells (post-processing)
    10. Top-5 most-worsened cells (post-processing)

All figures saved to results/circularity_mechanism/figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "circularity_mechanism"
FIG_DIR = OUT_DIR / "figures"

COLORS = {
    "boundary": "#E05C3A",
    "shape":    "#3A7EC0",
    "raw":      "#888888",
    "pp":       "#2CA02C",
    "clean":    "#1F77B4",
    "noise":    "#D62728",
}

PERT_COLORS = {
    "erosion":     "#3A7EC0",
    "dilation":    "#5BA3E0",
    "jitter":      "#E05C3A",
    "protrusion":  "#C0392B",
    "indentation": "#E67E22",
}


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Figure 1: Area error vs perimeter error change under noise
# ---------------------------------------------------------------------------

def fig_area_vs_perimeter_change(deltas_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(deltas_path)
    df = df[df["degradation"] == "gaussian_noise"]
    df = df[~df["invalid_circularity"].fillna(False)]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        df["delta_area_abs_err"],
        df["delta_perim_abs_err"],
        c=df["severity"].astype(float),
        cmap="plasma",
        alpha=0.35,
        s=10,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Noise severity")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.axvline(0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel("Delta nucleus area absolute error (noise - clean)")
    ax.set_ylabel("Delta nucleus perimeter absolute error (noise - clean)")
    ax.set_title("Fig 1: Area vs Perimeter Error Change under Gaussian Noise")
    _save(fig, "fig01_area_vs_perimeter_change")


# ---------------------------------------------------------------------------
# Figure 2: Perimeter error delta vs circularity error delta
# ---------------------------------------------------------------------------

def fig_perim_vs_circ_delta(deltas_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(deltas_path)
    df = df[df["degradation"] == "gaussian_noise"]
    df = df[~df["invalid_circularity"].fillna(False)]

    # Cell-level means
    agg = df.groupby("cell_id")[["delta_perim_abs_err", "delta_circ_abs_err"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(agg["delta_perim_abs_err"], agg["delta_circ_abs_err"],
               color=COLORS["noise"], alpha=0.6, s=18, edgecolors="none")
    # Trend line
    m = np.isfinite(agg["delta_perim_abs_err"]) & np.isfinite(agg["delta_circ_abs_err"])
    if m.sum() > 2:
        z = np.polyfit(agg.loc[m, "delta_perim_abs_err"], agg.loc[m, "delta_circ_abs_err"], 1)
        p = np.poly1d(z)
        xs = np.linspace(agg.loc[m, "delta_perim_abs_err"].min(),
                         agg.loc[m, "delta_perim_abs_err"].max(), 100)
        ax.plot(xs, p(xs), "k--", lw=1.2, alpha=0.7, label="OLS fit")
    ax.axhline(0, color="gray", lw=0.7, ls=":")
    ax.axvline(0, color="gray", lw=0.7, ls=":")
    ax.set_xlabel("Delta perimeter absolute error (cell mean)")
    ax.set_ylabel("Delta circularity absolute error (cell mean)")
    ax.set_title("Fig 2: Perimeter Error Delta vs Circularity Error Delta")
    ax.legend(fontsize=8)
    _save(fig, "fig02_perim_delta_vs_circ_delta")


# ---------------------------------------------------------------------------
# Figure 3: Area error delta vs circularity error delta
# ---------------------------------------------------------------------------

def fig_area_vs_circ_delta(deltas_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(deltas_path)
    df = df[df["degradation"] == "gaussian_noise"]
    df = df[~df["invalid_circularity"].fillna(False)]

    agg = df.groupby("cell_id")[["delta_area_abs_err", "delta_circ_abs_err"]].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(agg["delta_area_abs_err"], agg["delta_circ_abs_err"],
               color=COLORS["clean"], alpha=0.6, s=18, edgecolors="none")
    m = np.isfinite(agg["delta_area_abs_err"]) & np.isfinite(agg["delta_circ_abs_err"])
    if m.sum() > 2:
        z = np.polyfit(agg.loc[m, "delta_area_abs_err"], agg.loc[m, "delta_circ_abs_err"], 1)
        p = np.poly1d(z)
        xs = np.linspace(agg.loc[m, "delta_area_abs_err"].min(),
                         agg.loc[m, "delta_area_abs_err"].max(), 100)
        ax.plot(xs, p(xs), "k--", lw=1.2, alpha=0.7, label="OLS fit")
    ax.axhline(0, color="gray", lw=0.7, ls=":")
    ax.axvline(0, color="gray", lw=0.7, ls=":")
    ax.set_xlabel("Delta area absolute error (cell mean)")
    ax.set_ylabel("Delta circularity absolute error (cell mean)")
    ax.set_title("Fig 3: Area Error Delta vs Circularity Error Delta")
    ax.legend(fontsize=8)
    _save(fig, "fig03_area_delta_vs_circ_delta")


# ---------------------------------------------------------------------------
# Figure 4: Dice-matched perturbation: circularity change
# ---------------------------------------------------------------------------

def fig_dice_matched_circularity(dm_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(dm_path)
    df = df[df["band_populated"] == True]
    df = df[~df["perturbation_type"].str.startswith("ALL_")]

    if len(df) == 0:
        print("  [SKIP] No Dice-matched data for fig 4.")
        return

    bands = sorted(df["dice_band"].unique())
    fig, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 5), sharey=True)
    if len(bands) == 1:
        axes = [axes]

    for ax, band in zip(axes, bands):
        sub = df[df["dice_band"] == band].copy()
        sub = sub.sort_values("perturbation_type")
        colors = [PERT_COLORS.get(p, "#999999") for p in sub["perturbation_type"]]
        bars = ax.bar(sub["perturbation_type"], sub["mean_abs_circularity_change"].values,
                      color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(f"Dice band: {band}")
        ax.set_xlabel("Perturbation type")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel("|Circularity change|" if ax == axes[0] else "")

    fig.suptitle("Fig 4: Dice-Matched Perturbation - Circularity Change", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig04_dice_matched_circularity")


# ---------------------------------------------------------------------------
# Figure 5: Dice-matched perturbation: N/C change
# ---------------------------------------------------------------------------

def fig_dice_matched_nc(dm_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(dm_path)
    df = df[df["band_populated"] == True]
    df = df[~df["perturbation_type"].str.startswith("ALL_")]

    if len(df) == 0:
        print("  [SKIP] No Dice-matched data for fig 5.")
        return

    bands = sorted(df["dice_band"].unique())
    fig, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 5), sharey=True)
    if len(bands) == 1:
        axes = [axes]

    for ax, band in zip(axes, bands):
        sub = df[df["dice_band"] == band].copy()
        sub = sub.sort_values("perturbation_type")
        colors = [PERT_COLORS.get(p, "#999999") for p in sub["perturbation_type"]]
        ax.bar(sub["perturbation_type"], sub["mean_abs_nc_change"].values,
               color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(f"Dice band: {band}")
        ax.set_xlabel("Perturbation type")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylabel("|N/C change|" if ax == axes[0] else "")

    fig.suptitle("Fig 5: Dice-Matched Perturbation - N/C Change", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig05_dice_matched_nc")


# ---------------------------------------------------------------------------
# Figure 6: Perturbation area/perimeter change summary
# ---------------------------------------------------------------------------

def fig_perturbation_area_perimeter(pert_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(pert_path)
    df = df[df["valid"] == True]

    agg = df.groupby("perturbation").agg(
        mean_area_change_rel=("area_change_rel", "mean"),
        mean_perim_change_rel=("perimeter_change_rel", "mean"),
    ).reset_index()

    x = np.arange(len(agg))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_area = [PERT_COLORS.get(p, "#999") for p in agg["perturbation"]]
    b1 = ax.bar(x - width/2, agg["mean_area_change_rel"], width, label="Area (rel)",
                color=COLORS["clean"], alpha=0.8, edgecolor="white")
    b2 = ax.bar(x + width/2, agg["mean_perim_change_rel"], width, label="Perimeter (rel)",
                color=COLORS["noise"], alpha=0.8, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["perturbation"], rotation=30)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_ylabel("Mean relative change")
    ax.set_title("Fig 6: Perturbation Area and Perimeter Change")
    ax.legend()
    _save(fig, "fig06_perturbation_area_perimeter")


# ---------------------------------------------------------------------------
# Figure 7: Raw vs post-processed circularity error
# ---------------------------------------------------------------------------

def fig_raw_vs_pp_circularity(pp_path: Path) -> None:
    import pandas as pd
    if not pp_path.exists():
        print("  [SKIP] postprocessing_test_results.csv not found for fig 7.")
        return

    df = pd.read_csv(pp_path)
    noise_df = df[df["degradation"] == "gaussian_noise"]
    raw = noise_df[noise_df["processing"] == "raw"]["circularity_absolute_error"]
    pp  = noise_df[noise_df["processing"] == "post_processed"]["circularity_absolute_error"]

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot([raw.dropna().values, pp.dropna().values],
                          positions=[0, 1], showmedians=True)
    parts["bodies"][0].set_facecolor(COLORS["raw"])
    parts["bodies"][1].set_facecolor(COLORS["pp"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Raw prediction", "Post-processed"])
    ax.set_ylabel("Circularity absolute error")
    ax.set_title("Fig 7: Raw vs Post-processed Circularity Error (Noise, Test Set)")
    _save(fig, "fig07_raw_vs_pp_circularity")


# ---------------------------------------------------------------------------
# Figure 8: Raw vs post-processed Dice and N/C trade-off
# ---------------------------------------------------------------------------

def fig_raw_vs_pp_tradeoff(pp_path: Path) -> None:
    import pandas as pd
    if not pp_path.exists():
        print("  [SKIP] postprocessing_test_results.csv not found for fig 8.")
        return

    df = pd.read_csv(pp_path)
    noise_df = df[df["degradation"] == "gaussian_noise"]

    metrics = {
        "nucleus_dice":              "Nucleus Dice",
        "nc_absolute_error":         "N/C absolute error",
        "circularity_absolute_error": "Circularity absolute error",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, label) in zip(axes, metrics.items()):
        raw = noise_df[noise_df["processing"] == "raw"][col].dropna().values
        pp  = noise_df[noise_df["processing"] == "post_processed"][col].dropna().values
        parts = ax.violinplot([raw, pp], positions=[0, 1], showmedians=True)
        parts["bodies"][0].set_facecolor(COLORS["raw"])
        parts["bodies"][1].set_facecolor(COLORS["pp"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Raw", "PP"], fontsize=9)
        ax.set_title(label, fontsize=9)

    fig.suptitle("Fig 8: Post-processing Trade-off (Noise, Test Set)", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig08_raw_vs_pp_tradeoff")


# ---------------------------------------------------------------------------
# Figures 9–10: Top-5 improved / worsened cells
# ---------------------------------------------------------------------------

def _top5_cell_panel(paired_df, pp_results_df, title: str, name: str, best: bool = True) -> None:
    """Panel showing top-5 cells by circularity change from post-processing."""
    import pandas as pd

    noise_pairs = paired_df[
        (paired_df["metric"] == "circularity_absolute_error") &
        (paired_df["degradation"] == "gaussian_noise")
    ].copy()

    # Aggregate by cell_id (mean across seeds/models/severities)
    agg = noise_pairs.groupby("cell_id")["difference_pp_minus_raw"].mean().reset_index()
    agg = agg.sort_values("difference_pp_minus_raw", ascending=best)  # ascending=True -> most improvement
    top5 = agg.head(5)

    if len(top5) == 0:
        print(f"  [SKIP] No data for {name}")
        return

    fig, axes = plt.subplots(1, len(top5), figsize=(3 * len(top5), 4))
    if len(top5) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, top5.iterrows()):
        cid = row["cell_id"]
        diff = row["difference_pp_minus_raw"]
        # Get representative row for this cell
        cell_pp = pp_results_df[
            (pp_results_df["cell_id"] == cid) &
            (pp_results_df["processing"] == "post_processed") &
            (pp_results_df["degradation"] == "gaussian_noise")
        ]
        cell_raw = pp_results_df[
            (pp_results_df["cell_id"] == cid) &
            (pp_results_df["processing"] == "raw") &
            (pp_results_df["degradation"] == "gaussian_noise")
        ]

        raw_circ_err = cell_raw["circularity_absolute_error"].mean() if len(cell_raw) > 0 else float("nan")
        pp_circ_err  = cell_pp["circularity_absolute_error"].mean()  if len(cell_pp)  > 0 else float("nan")
        raw_dice     = cell_raw["nucleus_dice"].mean()                if len(cell_raw) > 0 else float("nan")
        pp_dice      = cell_pp["nucleus_dice"].mean()                 if len(cell_pp)  > 0 else float("nan")

        label_text = (
            f"Cell: {cid[:20]}\n"
            f"Circ err: {raw_circ_err:.3f}->{pp_circ_err:.3f}\n"
            f"Dice: {raw_dice:.3f}->{pp_dice:.3f}\n"
            f"PP delta: {diff:+.4f}"
        )
        ax.text(0.5, 0.5, label_text, ha="center", va="center",
                fontsize=7, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", alpha=0.8))
        ax.axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    _save(fig, name)


def fig_top5_improved(paired_path: Path, pp_path: Path) -> None:
    import pandas as pd
    if not paired_path.exists() or not pp_path.exists():
        print("  [SKIP] post-processing results not found for fig 9.")
        return
    paired_df = pd.read_csv(paired_path)
    pp_df = pd.read_csv(pp_path)
    _top5_cell_panel(paired_df, pp_df,
                     "Fig 9: Top-5 Cells Most Improved by Post-processing",
                     "fig09_top5_improved", best=True)


def fig_top5_worsened(paired_path: Path, pp_path: Path) -> None:
    import pandas as pd
    if not paired_path.exists() or not pp_path.exists():
        print("  [SKIP] post-processing results not found for fig 10.")
        return
    paired_df = pd.read_csv(paired_path)
    pp_df = pd.read_csv(pp_path)
    _top5_cell_panel(paired_df, pp_df,
                     "Fig 10: Top-5 Cells Most Worsened by Post-processing",
                     "fig10_top5_worsened", best=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(out_dir: Path = OUT_DIR) -> None:
    """Generate all 10 required figures."""
    global FIG_DIR
    FIG_DIR = out_dir / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    deltas_path = out_dir / "real_prediction_deltas.csv"
    dm_path     = out_dir / "dice_matched_perturbation_analysis.csv"
    pert_path   = out_dir / "controlled_mask_perturbations.csv"
    pp_path     = out_dir / "postprocessing_test_results.csv"
    paired_path = out_dir / "postprocessing_paired_differences.csv"

    print("Generating figures ...")
    if deltas_path.exists():
        fig_area_vs_perimeter_change(deltas_path)
        fig_perim_vs_circ_delta(deltas_path)
        fig_area_vs_circ_delta(deltas_path)
    else:
        print("  [SKIP] Deltas CSV not found (figs 1-3).")

    if dm_path.exists():
        fig_dice_matched_circularity(dm_path)
        fig_dice_matched_nc(dm_path)
    else:
        print("  [SKIP] Dice-matched CSV not found (figs 4-5).")

    if pert_path.exists():
        fig_perturbation_area_perimeter(pert_path)
    else:
        print("  [SKIP] Perturbation CSV not found (fig 6).")

    fig_raw_vs_pp_circularity(pp_path)
    fig_raw_vs_pp_tradeoff(pp_path)
    fig_top5_improved(paired_path, pp_path)
    fig_top5_worsened(paired_path, pp_path)

    print("All figures complete.")


if __name__ == "__main__":
    run()
