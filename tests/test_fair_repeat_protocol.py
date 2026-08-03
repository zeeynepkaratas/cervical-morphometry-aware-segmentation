"""Tests for fair paired repeat protocol helpers."""

import sys
from pathlib import Path

import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments.run_fair_repeat import _checkpoint_config, _load_resume
from src.pilot.fair_repeat import RunSpec, paired_differences, prepare_initial_state, train_order_hash
from src.segmentation.unet_model import UNet


class TinyDataset:
    """Minimal dataset shell for DataLoader order hashing."""

    def __init__(self) -> None:
        self.image_paths = [Path(f"cell_{index}.bmp") for index in range(12)]

    def __len__(self) -> int:
        return len(self.image_paths)


def test_same_seed_models_share_initial_state_hash(tmp_path: Path) -> None:
    _, baseline_hash = prepare_initial_state(20260803, 2, tmp_path)
    _, nc_hash = prepare_initial_state(20260803, 2, tmp_path)
    assert baseline_hash == nc_hash


def test_different_seed_initial_state_hashes_differ(tmp_path: Path) -> None:
    _, first = prepare_initial_state(20260803, 2, tmp_path)
    _, second = prepare_initial_state(20260804, 2, tmp_path)
    assert first != second


def test_two_models_receive_same_train_order_for_same_seed() -> None:
    dataset = TinyDataset()
    baseline_order = train_order_hash(dataset, batch_size=3, seed=20260803)
    nc_order = train_order_hash(dataset, batch_size=3, seed=20260803)
    assert baseline_order == nc_order


def test_paired_table_requires_two_models_per_seed_cell() -> None:
    rows = [
        {"seed": 1, "cell_id": "a", "class": "x", "model": "baseline", "nc_absolute_error": 0.4, "foreground_dice": 0.8, "nucleus_dice": 0.7, "cytoplasm_dice": 0.9, "circularity_absolute_error": 0.2, "nucleus_area_absolute_error": 5, "cytoplasm_area_absolute_error": 7},
        {"seed": 1, "cell_id": "a", "class": "x", "model": "nc_aware_lambda_0p10", "nc_absolute_error": 0.3, "foreground_dice": 0.81, "nucleus_dice": 0.72, "cytoplasm_dice": 0.9, "circularity_absolute_error": 0.25, "nucleus_area_absolute_error": 4, "cytoplasm_area_absolute_error": 10},
        {"seed": 1, "cell_id": "b", "class": "x", "model": "baseline", "nc_absolute_error": 0.1, "foreground_dice": 0.8, "nucleus_dice": 0.7, "cytoplasm_dice": 0.9, "circularity_absolute_error": 0.2, "nucleus_area_absolute_error": 5, "cytoplasm_area_absolute_error": 7},
    ]
    paired = paired_differences(rows)
    assert len(paired) == 1
    assert paired[0]["cell_id"] == "a"


def test_delta_sign_is_nc_aware_minus_baseline() -> None:
    rows = [
        {"seed": 1, "cell_id": "a", "class": "x", "model": "baseline", "nc_absolute_error": 0.4, "foreground_dice": 0.8, "nucleus_dice": 0.7, "cytoplasm_dice": 0.9, "circularity_absolute_error": 0.2, "nucleus_area_absolute_error": 5, "cytoplasm_area_absolute_error": 7},
        {"seed": 1, "cell_id": "a", "class": "x", "model": "nc_aware_lambda_0p10", "nc_absolute_error": 0.3, "foreground_dice": 0.81, "nucleus_dice": 0.72, "cytoplasm_dice": 0.9, "circularity_absolute_error": 0.25, "nucleus_area_absolute_error": 4, "cytoplasm_area_absolute_error": 10},
    ]
    paired = paired_differences(rows)
    assert paired[0]["delta_nc_error"] == pytest.approx(-0.1)
    assert paired[0]["delta_foreground_dice"] == 0.010000000000000009


def test_resume_restores_epoch_and_optimizer_state(tmp_path: Path) -> None:
    model = UNet(base_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    path = tmp_path / "last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 3,
            "best_val_dice": 0.8,
            "epochs_without_improvement": 2,
            "history": [{"epoch": 3}],
        },
        path,
    )
    fresh = UNet(base_channels=2)
    fresh_optimizer = torch.optim.Adam(fresh.parameters(), lr=0.001)
    start_epoch, best, stale, history = _load_resume(path, fresh, fresh_optimizer, torch.device("cpu"))
    assert start_epoch == 4
    assert best == 0.8
    assert stale == 2
    assert history == [{"epoch": 3}]
    assert fresh_optimizer.state_dict()["state"]


def test_primary_checkpoint_rule_is_best_val_dice() -> None:
    config = {"image_size": [128, 128], "training": {"base_channels": 2, "learning_rate": 0.001, "batch_size": 8}}
    assert _checkpoint_config(config)["primary_checkpoint_rule"] == "best_val_dice"
