import pytest

from src.pilot.target_specific_coverage import (
    cluster_bootstrap_target_gap,
    condition_key,
    more_fragile_target,
    pair_target_changes,
    rows_by_cell_id,
)


def test_condition_key_matches_same_condition_with_numeric_severity():
    row_a = {"seed": "1", "model": "m", "degradation": "gaussian_blur", "severity": "2.0"}
    row_b = {"seed": 1, "model": "m", "degradation": "gaussian_blur", "severity": 2}

    assert condition_key(row_a) == condition_key(row_b)


def test_coverage_change_sign_interpretation():
    assert more_fragile_target(-0.03) == "circularity"
    assert more_fragile_target(0.03) == "nc"
    assert more_fragile_target(0.0) == "equal"


def test_pair_target_changes_matches_only_within_same_condition():
    rows = [
        {
            "seed": 1,
            "model": "m",
            "degradation": "gaussian_blur",
            "severity": 1,
            "target": "nc",
            "clean_coverage": 0.9,
            "empirical_coverage": 0.88,
            "coverage_delta_vs_clean": -0.02,
        },
        {
            "seed": 1,
            "model": "m",
            "degradation": "gaussian_blur",
            "severity": "1.0",
            "target": "circularity",
            "clean_coverage": 0.9,
            "empirical_coverage": 0.84,
            "coverage_delta_vs_clean": -0.06,
        },
        {
            "seed": 1,
            "model": "m",
            "degradation": "gaussian_blur",
            "severity": 2,
            "target": "circularity",
            "clean_coverage": 0.9,
            "empirical_coverage": 0.7,
            "coverage_delta_vs_clean": -0.2,
        },
    ]

    paired = pair_target_changes(rows)

    assert len(paired) == 1
    assert paired[0]["target_gap_difference"] == pytest.approx(-0.04)
    assert paired[0]["more_fragile_target"] == "circularity"


def test_cluster_bootstrap_preserves_cell_id_dependency():
    rows = [
        {
            "cell_id": "a",
            "seed": 1,
            "model": "m",
            "degradation": degradation,
            "severity": severity,
            "target": target,
            "covered": covered,
            "valid": True,
        }
        for degradation, severity, target, covered in [
            ("clean", "clean", "nc", True),
            ("clean", "clean", "circularity", True),
            ("gaussian_blur", 1, "nc", True),
            ("gaussian_blur", 1, "circularity", False),
        ]
    ]
    rows += [
        {
            "cell_id": "b",
            "seed": 1,
            "model": "m",
            "degradation": degradation,
            "severity": severity,
            "target": target,
            "covered": covered,
            "valid": True,
        }
        for degradation, severity, target, covered in [
            ("clean", "clean", "nc", True),
            ("clean", "clean", "circularity", True),
            ("gaussian_blur", 1, "nc", True),
            ("gaussian_blur", 1, "circularity", True),
        ]
    ]

    grouped = rows_by_cell_id(rows)
    result = cluster_bootstrap_target_gap(rows, repeats=20, seed=7)

    assert set(grouped) == {"a", "b"}
    assert len(grouped["a"]) == 4
    assert result["cluster_unit"] == "cell_id"
    assert result["cluster_count"] == 2
