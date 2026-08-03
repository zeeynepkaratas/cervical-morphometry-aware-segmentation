"""Tests for differentiable N/C-aware loss."""

import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.losses.morphometry_loss import (
    CYTOPLASM_CLASS_INDEX,
    NUCLEUS_CLASS_INDEX,
    combined_segmentation_nc_loss,
    nc_ratio_loss,
    soft_nc_ratio_from_logits,
    true_nc_ratio_from_targets,
)
from src.segmentation.train_unet import segmentation_loss


def _target(batch: int = 1) -> torch.Tensor:
    target = torch.zeros((batch, 4, 4), dtype=torch.long)
    target[:, :2, :] = CYTOPLASM_CLASS_INDEX
    target[:, 2:, :2] = NUCLEUS_CLASS_INDEX
    return target


def _perfect_logits(targets: torch.Tensor) -> torch.Tensor:
    logits = torch.full((targets.size(0), 3, targets.size(1), targets.size(2)), -20.0)
    return logits.scatter_(1, targets.unsqueeze(1), 20.0)


def test_perfect_prediction_nc_loss_is_near_zero() -> None:
    targets = _target()
    loss = nc_ratio_loss(_perfect_logits(targets), targets)
    assert loss.item() < 1e-6


def test_nc_ratio_increases_when_nucleus_probability_increases() -> None:
    logits = torch.zeros((1, 3, 2, 2))
    base = soft_nc_ratio_from_logits(logits)
    logits[:, NUCLEUS_CLASS_INDEX] += 2.0
    assert torch.all(soft_nc_ratio_from_logits(logits) > base)


def test_nc_ratio_decreases_when_cytoplasm_probability_increases() -> None:
    logits = torch.zeros((1, 3, 2, 2))
    base = soft_nc_ratio_from_logits(logits)
    logits[:, CYTOPLASM_CLASS_INDEX] += 2.0
    assert torch.all(soft_nc_ratio_from_logits(logits) < base)


def test_loss_backpropagates() -> None:
    targets = _target(batch=2)
    logits = torch.randn((2, 3, 4, 4), requires_grad=True)
    loss = combined_segmentation_nc_loss(logits, targets, lambda_nc=0.1).total_loss
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_loss_and_gradient_are_finite() -> None:
    targets = torch.zeros((2, 4, 4), dtype=torch.long)
    logits = torch.randn((2, 3, 4, 4), requires_grad=True)
    loss = combined_segmentation_nc_loss(logits, targets, lambda_nc=0.2).total_loss
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_batch_size_greater_than_one() -> None:
    ratios = true_nc_ratio_from_targets(_target(batch=3))
    assert ratios.shape == (3,)


def test_lambda_zero_matches_segmentation_loss() -> None:
    targets = _target(batch=2)
    logits = torch.randn((2, 3, 4, 4))
    combined = combined_segmentation_nc_loss(logits, targets, lambda_nc=0.0).total_loss
    baseline = segmentation_loss(logits, targets)
    assert torch.allclose(combined, baseline, atol=0.0, rtol=0.0)


def test_tiny_cytoplasm_area_does_not_overflow() -> None:
    targets = torch.full((1, 4, 4), NUCLEUS_CLASS_INDEX, dtype=torch.long)
    targets[:, 0, 0] = CYTOPLASM_CLASS_INDEX
    logits = _perfect_logits(targets)
    loss = nc_ratio_loss(logits, targets)
    assert torch.isfinite(loss)


def test_class_indices_match_source_project() -> None:
    assert CYTOPLASM_CLASS_INDEX == 1
    assert NUCLEUS_CLASS_INDEX == 2


def test_true_nc_is_nucleus_over_cytoplasm_only() -> None:
    targets = torch.zeros((1, 3, 3), dtype=torch.long)
    targets[:, :2, :] = CYTOPLASM_CLASS_INDEX
    targets[:, 2, :3] = NUCLEUS_CLASS_INDEX
    ratio = true_nc_ratio_from_targets(targets, eps=0.0)
    assert torch.allclose(ratio, torch.tensor([3 / 6]))
