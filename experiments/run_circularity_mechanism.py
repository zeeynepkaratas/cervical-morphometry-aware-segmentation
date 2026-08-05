"""Master experiment runner for the circularity mechanism analysis.

Usage:
    python experiments/run_circularity_mechanism.py [--skip-inference]

Phases:
    0. Data leakage audit
    A. Area–perimeter decomposition (from existing CSVs)
    B. Controlled mask perturbation benchmark
    C. Post-processing validation selection + test evaluation (requires inference)
    D. Statistics, plots, report

--skip-inference skips Phase C inference (useful if checkpoints / GPU not available).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "circularity_mechanism"


def run_phase_0():
    print("\n" + "=" * 60)
    print("PHASE 0: Data leakage audit")
    print("=" * 60)
    from src.leakage_audit import run
    run(out_dir=OUT_DIR)


def run_phase_a():
    print("\n" + "=" * 60)
    print("PHASE A: Area–perimeter decomposition")
    print("=" * 60)
    from src.circularity_mechanism import run as run_ap
    ap_df, deltas_df = run_ap(out_dir=OUT_DIR)

    print("\n--- Phase A statistics ---")
    from src.mechanism_statistics import run as run_stats
    run_stats(deltas_path=OUT_DIR / "real_prediction_deltas.csv", out_dir=OUT_DIR)


def run_phase_b():
    print("\n" + "=" * 60)
    print("PHASE B: Controlled mask perturbation benchmark")
    print("=" * 60)
    from src.mask_perturbation import run as run_pert
    run_pert(out_dir=OUT_DIR)

    from src.dice_matched_analysis import run as run_dm
    run_dm(out_dir=OUT_DIR)


def run_phase_c(skip_inference: bool = False):
    print("\n" + "=" * 60)
    print("PHASE C: Post-processing selection + test evaluation")
    print("=" * 60)
    if skip_inference:
        print("  SKIPPED (--skip-inference flag set)")
        # Write a placeholder selection file indicating skip
        import json
        sel = {
            "decision": "SKIPPED",
            "selected_op": None,
            "candidates": [],
            "selection_set": "unet_val",
            "test_not_seen": True,
            "kernel": "3x3_ellipse",
            "iterations": 1,
            "note": "Inference skipped; torch checkpoints or GPU not available.",
        }
        sel_path = OUT_DIR / "postprocessing_selection.json"
        sel_path.write_text(json.dumps(sel, indent=2), encoding="utf-8")
        print(f"  Written placeholder: {sel_path}")
        return

    from src.postprocessing import run as run_pp
    run_pp(out_dir=OUT_DIR)


def run_phase_d():
    print("\n" + "=" * 60)
    print("PHASE D: Additional statistics + statistical intervals")
    print("=" * 60)

    # Compute statistical intervals over post-processing results (if available)
    pp_results_path = OUT_DIR / "postprocessing_test_results.csv"
    perturbations_path = OUT_DIR / "controlled_mask_perturbations.csv"

    if pp_results_path.exists():
        _pp_statistics(pp_results_path)
    else:
        print("  [SKIP] postprocessing_test_results.csv not found.")

    if perturbations_path.exists():
        _perturbation_statistics(perturbations_path)
    else:
        print("  [SKIP] controlled_mask_perturbations.csv not found.")

    # Compile statistical_intervals.csv
    _compile_statistical_intervals()

    # Re-run leakage audit with postprocessing selection
    from src.leakage_audit import run as run_audit
    run_audit(out_dir=OUT_DIR)


def _pp_statistics(results_path: Path):
    """Bootstrap intervals for post-processing test results."""
    import numpy as np
    import pandas as pd
    from src.mechanism_statistics import _cluster_bootstrap_mean, BOOTSTRAP_REPS, BOOTSTRAP_SEED

    df = pd.read_csv(results_path)
    raw_noise = df[(df["processing"] == "raw") & (df["degradation"] == "gaussian_noise")].copy()
    pp_noise  = df[(df["processing"] == "post_processed") & (df["degradation"] == "gaussian_noise")].copy()

    rows = []
    if len(raw_noise) > 0 and len(pp_noise) > 0:
        merged = raw_noise.merge(pp_noise, on=["cell_id", "seed", "model", "degradation", "severity"],
                                  suffixes=("_raw", "_pp"))
        metrics = [
            ("circularity_absolute_error", "pp_circularity_error_change"),
            ("nc_absolute_error",          "pp_nc_error_change"),
            ("nucleus_dice",               "pp_dice_change"),
        ]
        for raw_col, label in metrics:
            pp_col = raw_col + "_pp"
            raw_col2 = raw_col + "_raw"
            if pp_col in merged.columns and raw_col2 in merged.columns:
                delta = (merged[pp_col] - merged[raw_col2]).values
                r = _cluster_bootstrap_mean(delta, merged["cell_id_raw"].values)
                rows.append({"source": "postprocessing_test", "metric": label, **r})

    return rows


def _perturbation_statistics(perturbations_path: Path):
    """Clustered bootstrap for perturbation circularity/N/C changes."""
    import numpy as np
    import pandas as pd
    from src.mechanism_statistics import _cluster_bootstrap_mean, BOOTSTRAP_REPS, BOOTSTRAP_SEED

    df = pd.read_csv(perturbations_path)
    valid_df = df[df["valid"] == True].copy()

    rows = []
    for pert_type in sorted(valid_df["perturbation"].unique()):
        sub = valid_df[valid_df["perturbation"] == pert_type]
        for metric, label in [
            ("circularity_change", f"{pert_type}_circularity_change"),
            ("nc_change",          f"{pert_type}_nc_change"),
            ("area_change_rel",    f"{pert_type}_area_change_rel"),
            ("perimeter_change_rel", f"{pert_type}_perim_change_rel"),
        ]:
            if metric in sub.columns:
                r = _cluster_bootstrap_mean(sub[metric].values, sub["cell_id"].values)
                rows.append({"source": "controlled_perturbation", "metric": label, **r})

    return rows


def _compile_statistical_intervals():
    """Merge all bootstrap intervals into a single CSV."""
    import pandas as pd

    all_rows = []

    # Phase A bootstrap
    boot_path = OUT_DIR / "area_perimeter_bootstrap.csv"
    if boot_path.exists():
        boot_df = pd.read_csv(boot_path)
        boot_df["source"] = "phase_a_mechanism"
        all_rows.append(boot_df)

    # PP statistics
    pp_results_path = OUT_DIR / "postprocessing_test_results.csv"
    if pp_results_path.exists():
        pp_rows = _pp_statistics(pp_results_path)
        if pp_rows:
            all_rows.append(pd.DataFrame(pp_rows))

    # Perturbation statistics
    pert_path = OUT_DIR / "controlled_mask_perturbations.csv"
    if pert_path.exists():
        pert_rows = _perturbation_statistics(pert_path)
        if pert_rows:
            all_rows.append(pd.DataFrame(pert_rows))

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        out_path = OUT_DIR / "statistical_intervals.csv"
        combined.to_csv(out_path, index=False)
        print(f"Written: {out_path}  ({len(combined)} rows)")
    else:
        print("  [SKIP] No statistical data available for intervals.")


def main():
    parser = argparse.ArgumentParser(description="Circularity mechanism analysis runner")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip Phase C inference (post-processing requires GPU + checkpoints)")
    parser.add_argument("--phase", choices=["0", "A", "B", "C", "D", "all"], default="all",
                        help="Run only a specific phase")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    phases = args.phase
    run_all = (phases == "all")

    if run_all or phases == "0":
        run_phase_0()
    if run_all or phases == "A":
        run_phase_a()
    if run_all or phases == "B":
        run_phase_b()
    if run_all or phases == "C":
        run_phase_c(skip_inference=args.skip_inference)
    if run_all or phases == "D":
        run_phase_d()

    print("\n" + "=" * 60)
    print("All phases complete.")
    print(f"Results: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
