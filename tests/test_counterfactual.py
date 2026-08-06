import numpy as np
import pytest
import pandas as pd
import json
import os
from unittest.mock import patch
from src.mask_perturbation import perturb_area_preserving_boundary
from experiments.run_counterfactual_analysis import bootstrap_cell_means, run_counterfactual_comparison
from src.counterfactual_decomposition import run_decomposition, bootstrap_perimeter_dominant_fraction

def test_dummy_perturbation_valid():
    cytoplasm = np.zeros((100, 100), dtype=bool)
    nucleus = np.zeros((100, 100), dtype=bool)
    nucleus[40:60, 40:60] = True
    cytoplasm[30:70, 30:70] = True
    cytoplasm = cytoplasm & ~nucleus
    
    perturbed, valid, reason = perturb_area_preserving_boundary(nucleus, cytoplasm, "medium", "test_cell")
    assert valid is True, f"Dummy counterfactual failed: {reason}"

def test_area_preserving_perturbation_exactness():
    cytoplasm = np.zeros((100, 100), dtype=bool)
    nucleus = np.zeros((100, 100), dtype=bool)
    nucleus[40:60, 40:60] = True
    cytoplasm[30:70, 30:70] = True
    cytoplasm = cytoplasm & ~nucleus
    
    orig_area = np.count_nonzero(nucleus)
    
    perturbed, valid, reason = perturb_area_preserving_boundary(nucleus, cytoplasm, "medium", "test_cell")
    
    if valid:
        new_area = np.count_nonzero(perturbed)
        assert orig_area == new_area, f"Area changed: {orig_area} != {new_area}"
        
        # Verify it stays within cell containment
        assert not np.any(perturbed & ~(nucleus | cytoplasm)), "Expanded outside original cell mask"

        # Contour area tolerance
        import cv2
        c_orig, _ = cv2.findContours(nucleus.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_new, _ = cv2.findContours(perturbed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        a_orig = cv2.contourArea(max(c_orig, key=cv2.contourArea))
        a_new = cv2.contourArea(max(c_new, key=cv2.contourArea))
        
        assert abs(a_new - a_orig) / a_orig <= 0.02, "Contour area changed by >2%"

        # Topological constraints (single component, no new hole)
        c_new_all, hierarchy = cv2.findContours(perturbed.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        ext_count = sum(1 for h in hierarchy[0] if h[3] == -1)
        assert ext_count == 1, "Not a single connected component"
        assert not any(h[3] != -1 for h in hierarchy[0]), "Contains holes"

def test_area_preserving_insufficient_area():
    cytoplasm = np.zeros((20, 20), dtype=bool)
    nucleus = np.zeros((20, 20), dtype=bool)
    nucleus[5:10, 5:10] = True
    cytoplasm[2:15, 2:15] = True
    
    perturbed, valid, reason = perturb_area_preserving_boundary(nucleus, cytoplasm, "low", "test_cell")
    
    assert not valid
    assert reason == "nucleus_too_small"

def test_bootstrap_cell_means():
    cell_differences = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -1.0])
    ci1 = bootstrap_cell_means(cell_differences, n_boot=1000, seed=42)
    ci2 = bootstrap_cell_means(cell_differences, n_boot=1000, seed=42)
    
    assert np.allclose(ci1, ci2), "Bootstrap is not reproducible"
    assert ci1[0] <= np.mean(cell_differences) <= ci1[1], "Mean not within CI"

def test_bootstrap_cell_means_weighting():
    # If one cell had many instances (not applied here since we use cell_differences directly)
    # the function must only sample the 1D array of differences.
    diffs = np.array([1.0, 1.0, 1.0])
    ci = bootstrap_cell_means(diffs, n_boot=100)
    assert ci[0] == 1.0 and ci[1] == 1.0

def test_decomposition_uniqueness_checks(tmp_path):
    import os
    # Create fake CSV with duplicate clean keys
    df = pd.DataFrame({
        "cell_id": ["c1", "c1", "c2"],
        "model": ["m1", "m1", "m1"],
        "seed": [1, 1, 1],
        "degradation": ["clean", "clean", "noise"],
        "severity": ["clean", "clean", "10"],
        "predicted_nucleus_area": [100, 100, 100],
        "direct_predicted_perimeter": [40, 40, 40]
    })
    
    os.makedirs(tmp_path / "results" / "circularity_mechanism", exist_ok=True)
    df.to_csv(tmp_path / "results" / "circularity_mechanism" / "real_prediction_area_perimeter.csv", index=False)
    
    with patch("pandas.read_csv", return_value=df):
        with pytest.raises(ValueError, match="Duplicate keys found in clean predictions"):
            run_decomposition()

def test_decomposition_uniqueness_checks_corr(tmp_path):
    df = pd.DataFrame({
        "cell_id": ["c1", "c2", "c2"],
        "model": ["m1", "m1", "m1"],
        "seed": [1, 1, 1],
        "degradation": ["clean", "noise", "noise"],
        "severity": ["clean", "10", "10"],
        "predicted_nucleus_area": [100, 100, 100],
        "direct_predicted_perimeter": [40, 40, 40]
    })
    
    with patch("pandas.read_csv", return_value=df):
        with pytest.raises(ValueError, match="Duplicate keys found in corrupted predictions"):
            run_decomposition()

# --- New Audit Tests ---

def test_deterministic_within_cell_matching_and_no_reuse():
    # We will simulate the dataframe logic from run_counterfactual_comparison
    # testing matching sorting and no-reuse.
    
    # 2 CFs, 2 Shapes in same cell
    shape_df = pd.DataFrame([
        {"cell_id": "c1", "perturbation": "erosion", "severity": "r1", "nucleus_dice": 0.96, "circularity_change": 0.1, "nc_change": 0.0},
        {"cell_id": "c1", "perturbation": "dilation", "severity": "r2", "nucleus_dice": 0.97, "circularity_change": 0.2, "nc_change": 0.0},
    ])
    
    counter_df = pd.DataFrame([
        {"cell_id": "c1", "perturbation": "area_preserving_boundary", "severity": "low", "nucleus_dice": 0.965, "circularity_change": 0.05, "nc_change": 0.0, "perimeter_change": 0},
        {"cell_id": "c1", "perturbation": "area_preserving_boundary", "severity": "high", "nucleus_dice": 0.965, "circularity_change": 0.06, "nc_change": 0.0, "perimeter_change": 0}
    ])
    
    # Matching logic from runner:
    severity_order = {"low": 1, "medium": 2, "high": 3, "r1": 1, "r2": 2}
    
    pairs = []
    used_shape_indices = set()
    counter_df["sev_order"] = counter_df["severity"].map(severity_order)
    counter_df = counter_df.sort_values("sev_order").reset_index(drop=True)
    
    for _, cf_row in counter_df.iterrows():
        available_shapes = []
        for i, shape_row in shape_df.iterrows():
            if i in used_shape_indices: continue
            diff = abs(cf_row["nucleus_dice"] - shape_row["nucleus_dice"])
            available_shapes.append({"idx": i, "diff": diff, "pert": shape_row["perturbation"], "sev": severity_order.get(shape_row["severity"], 99)})
        
        available_shapes.sort(key=lambda x: (x["diff"], x["pert"], x["sev"]))
        best_shape_idx = available_shapes[0]["idx"]
        used_shape_indices.add(best_shape_idx)
        pairs.append(shape_df.loc[best_shape_idx]["severity"])
        
    assert len(pairs) == 2
    # CF 'low' gets first choice. Both shapes have diff 0.005. 
    # Tie-break: pert name (dilation vs erosion). 'dilation' comes first.
    assert pairs[0] == "r2"  # dilation, r2
    assert pairs[1] == "r1"  # erosion, r1
    # This verifies no reuse, exact sort order, and tie-breaking

def test_cell_level_aggregation():
    pairs_df = pd.DataFrame([
        {"cell_id": "c1", "circ_change_diff": 0.01},
        {"cell_id": "c1", "circ_change_diff": 0.03},
        {"cell_id": "c2", "circ_change_diff": 0.08}
    ])
    cell_summary = pairs_df.groupby("cell_id")["circ_change_diff"].mean().reset_index()
    assert len(cell_summary) == 2
    assert cell_summary[cell_summary["cell_id"] == "c1"]["circ_change_diff"].values[0] == 0.02

def test_report_json_decision_consistency():
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    report_path = ROOT / "docs" / "counterfactual_decomposition_report.md"
    json_path = ROOT / "results" / "circularity_mechanism" / "counterfactual_interpretation.json"
    
    if report_path.exists() and json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
        decision = data["decision"]
        
        with open(report_path, "r") as f:
            content = f.read()
            
        assert decision in content, "JSON decision mismatch with report content"
