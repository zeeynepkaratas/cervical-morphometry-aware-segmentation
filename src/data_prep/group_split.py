"""Cell-level train/calibration/test splitting utilities."""

import random
from typing import Dict, List

from src.utils.config import RANDOM_SEED, SPLIT_RATIOS


def split_by_original_cell(cell_ids: List[str], seed: int = RANDOM_SEED) -> Dict[str, List[str]]:
    """
    Split original cell IDs into train/calibration/test sets.

    The split must be called before degraded variants are generated. All variants
    derived from the same original cell must remain in the same split.

    Returns:
        {"train": [...], "calibration": [...], "test": [...]}
    """
    unique_ids = sorted(set(cell_ids))
    if len(unique_ids) != len(cell_ids):
        raise ValueError("cell_ids must contain each original cell exactly once.")

    rng = random.Random(seed)
    shuffled = unique_ids[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(round(n_total * SPLIT_RATIOS["train"]))
    n_calibration = int(round(n_total * SPLIT_RATIOS["calibration"]))
    n_train = min(n_train, n_total)
    n_calibration = min(n_calibration, n_total - n_train)

    train = sorted(shuffled[:n_train])
    calibration = sorted(shuffled[n_train : n_train + n_calibration])
    test = sorted(shuffled[n_train + n_calibration :])
    assert_no_leakage(train, calibration, test)
    return {"train": train, "calibration": calibration, "test": test}


def sample_one_variant_per_cell(cell_id_to_variants: Dict[str, List[str]], seed: int) -> Dict[str, str]:
    """
    Select one variant per original cell for conformal calibration/test runs.

    This preserves statistical independence by ensuring one sampled variant per
    original cell for a given seed.
    """
    rng = random.Random(seed)
    selected = {}
    for cell_id, variants in sorted(cell_id_to_variants.items()):
        if not variants:
            raise ValueError(f"No variants available for cell_id={cell_id}")
        selected[cell_id] = rng.choice(sorted(variants))
    return selected


def assert_no_leakage(train_ids: List[str], cal_ids: List[str], test_ids: List[str]) -> None:
    """Assert that train/calibration/test ID sets are disjoint."""
    train = set(train_ids)
    calibration = set(cal_ids)
    test = set(test_ids)
    if train & calibration:
        raise AssertionError("Leakage between train and calibration sets.")
    if train & test:
        raise AssertionError("Leakage between train and test sets.")
    if calibration & test:
        raise AssertionError("Leakage between calibration and test sets.")
