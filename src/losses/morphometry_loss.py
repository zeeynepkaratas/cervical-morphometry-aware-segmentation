"""Differentiable N/C-aware loss for the Herlev pilot."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.segmentation.train_unet import segmentation_loss


CYTOPLASM_CLASS_INDEX = 1
NUCLEUS_CLASS_INDEX = 2


@dataclass(frozen=True)
class LossBreakdown:
    """Named loss components for logging and diagnostics."""

    total_loss: torch.Tensor
    segmentation_loss: torch.Tensor
    nc_loss: torch.Tensor


def soft_nc_ratio_from_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute soft nucleus/cytoplasm-only area ratios from model logits."""
    if logits.ndim != 4 or logits.size(1) < 3:
        raise ValueError(f"Expected logits with shape N x C x H x W and C>=3, got {tuple(logits.shape)}.")
    probabilities = torch.softmax(logits, dim=1)
    p_cytoplasm = probabilities[:, CYTOPLASM_CLASS_INDEX]
    p_nucleus = probabilities[:, NUCLEUS_CLASS_INDEX]
    nucleus_area = p_nucleus.sum(dim=(1, 2))
    cytoplasm_area = p_cytoplasm.sum(dim=(1, 2))
    return nucleus_area / (cytoplasm_area + eps)


def true_nc_ratio_from_targets(targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute target N/C ratios as nucleus area divided by cytoplasm-only area."""
    if targets.ndim != 3:
        raise ValueError(f"Expected target tensor with shape N x H x W, got {tuple(targets.shape)}.")
    nucleus_area = (targets == NUCLEUS_CLASS_INDEX).sum(dim=(1, 2)).float()
    cytoplasm_area = (targets == CYTOPLASM_CLASS_INDEX).sum(dim=(1, 2)).float()
    return nucleus_area / (cytoplasm_area + eps)


def nc_ratio_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """SmoothL1 loss between log soft predicted and target N/C ratios."""
    predicted = soft_nc_ratio_from_logits(logits, eps=eps)
    reference = true_nc_ratio_from_targets(targets, eps=eps)
    return F.smooth_l1_loss(torch.log(predicted + eps), torch.log(reference + eps))


def combined_segmentation_nc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lambda_nc: float,
    eps: float = 1e-6,
) -> LossBreakdown:
    """Return CrossEntropy+foreground Dice plus lambda-scaled N/C loss."""
    seg = segmentation_loss(logits, targets)
    if float(lambda_nc) == 0.0:
        zero = logits.sum() * 0.0
        return LossBreakdown(total_loss=seg, segmentation_loss=seg, nc_loss=zero)
    nc = nc_ratio_loss(logits, targets, eps=eps)
    return LossBreakdown(total_loss=seg + float(lambda_nc) * nc, segmentation_loss=seg, nc_loss=nc)
