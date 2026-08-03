"""Tests for official metric definitions."""

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.measurements.morphometry import compute_nc_ratio
from src.segmentation.train_unet import build_segmentation_target


def test_compute_nc_ratio_uses_cytoplasm_only() -> None:
    nucleus = np.zeros((4, 4), dtype=bool)
    cytoplasm = np.zeros((4, 4), dtype=bool)
    nucleus[:2, :2] = True
    cytoplasm[2:, :] = True
    assert compute_nc_ratio(nucleus, cytoplasm) == 4 / 8


def test_target_mapping_is_background_cytoplasm_nucleus() -> None:
    nucleus = np.zeros((4, 4), dtype=bool)
    cytoplasm = np.zeros((4, 4), dtype=bool)
    cytoplasm[:2, :] = True
    nucleus[2:, :2] = True
    target = build_segmentation_target(nucleus, cytoplasm, size=(4, 4)).numpy()
    assert set(np.unique(target).tolist()) == {0, 1, 2}
    assert np.all(target[nucleus] == 2)
    assert np.all(target[cytoplasm] == 1)
