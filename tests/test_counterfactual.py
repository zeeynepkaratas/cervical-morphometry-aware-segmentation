import numpy as np
import pytest
import pandas as pd
from src.mask_perturbation import perturb_area_preserving_boundary
from experiments.run_counterfactual_analysis import clustered_bootstrap_ci

def test_area_preserving_perturbation_exactness():
    # Create a dummy image and nucleus
    cytoplasm = np.zeros((100, 100), dtype=bool)
    nucleus = np.zeros((100, 100), dtype=bool)
    
    # 20x20 square nucleus = 400 area
    nucleus[40:60, 40:60] = True
    
    # 40x40 cytoplasm around it
    cytoplasm[30:70, 30:70] = True
    cytoplasm = cytoplasm & ~nucleus
    
    orig_area = np.count_nonzero(nucleus)
    
    perturbed, valid, reason = perturb_area_preserving_boundary(nucleus, cytoplasm, "medium", "test_cell")
    
    if valid:
        new_area = np.count_nonzero(perturbed)
        assert orig_area == new_area, f"Area changed: {orig_area} != {new_area}"
        
        # Verify it stays within cytoplasm
        assert not np.any(perturbed & ~(nucleus | cytoplasm)), "Expanded outside original cell mask"

def test_area_preserving_insufficient_area():
    # 5x5 nucleus = 25 area (< 50 limit)
    cytoplasm = np.zeros((20, 20), dtype=bool)
    nucleus = np.zeros((20, 20), dtype=bool)
    nucleus[5:10, 5:10] = True
    cytoplasm[2:15, 2:15] = True
    
    perturbed, valid, reason = perturb_area_preserving_boundary(nucleus, cytoplasm, "low", "test_cell")
    
    assert not valid
    assert reason == "nucleus_too_small"

def test_clustered_bootstrap_reproducibility():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -1.0])
    cell_ids = np.array(["a", "a", "b", "c", "d", "d"])
    
    ci1 = clustered_bootstrap_ci(data, cell_ids, n_boot=1000, seed=42)
    ci2 = clustered_bootstrap_ci(data, cell_ids, n_boot=1000, seed=42)
    
    assert np.allclose(ci1, ci2)
    assert ci1[0] <= np.mean(data) <= ci1[1]
