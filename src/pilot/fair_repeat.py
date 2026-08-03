"""Fair paired three-seed training, evaluation, and statistics helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_prep.load_herlev import list_herlev_images, load_image_and_masks
from src.losses.morphometry_loss import combined_segmentation_nc_loss
from src.measurements.morphometry import compute_circularity, compute_nc_ratio, relative_error
from src.pilot.config_io import repo_commit_sha
from src.pilot.metrics import (
    dice_score,
    finite_mean,
    finite_median,
    foreground_dice,
    masks_to_target,
    restore_prediction_to_original_size,
)
from src.segmentation.train_unet import HerlevSegmentationDataset, preprocess_rgb_image
from src.segmentation.unet_model import UNet


FAIR_SEEDS = [20260803, 20260804, 20260805]
FAIR_LAMBDAS = [0.0, 0.10]
MAX_EPOCHS = 25
PATIENCE = 6
MIN_DELTA = 1e-4
BOOTSTRAP_REPEATS = 5000
BOOTSTRAP_SEED = 20260803


@dataclass(frozen=True)
class RunSpec:
    """One fair-repeat model run."""

    seed: int
    lambda_nc: float

    @property
    def model(self) -> str:
        return "baseline" if self.lambda_nc == 0.0 else "nc_aware_lambda_0p10"

    @property
    def run_id(self) -> str:
        return f"{self.model}_seed_{self.seed}"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash a model state dict deterministically."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def prepare_initial_state(seed: int, base_channels: int, output_dir: Path) -> tuple[Path, str]:
    """Create or reuse a seed-specific initial U-Net state."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"initial_unet_seed_{seed}.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        return path, str(payload["state_sha256"])
    seed_everything(seed)
    model = UNet(base_channels=base_channels)
    state = model.state_dict()
    state_hash = state_dict_sha256(state)
    torch.save({"seed": seed, "state_dict": state, "state_sha256": state_hash}, path)
    return path, state_hash


def train_order_hash(dataset: HerlevSegmentationDataset, batch_size: int, seed: int) -> str:
    """Hash the shuffled sample order produced by a DataLoader generator."""
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    order: list[str] = []
    for batch_indices in loader.batch_sampler:
        order.extend(dataset.image_paths[index].stem for index in batch_indices)
    return hashlib.sha256("\n".join(order).encode("utf-8")).hexdigest()


def safe_metrics(pred_target: np.ndarray, gt_nucleus: np.ndarray, gt_cytoplasm: np.ndarray) -> dict:
    """Compute segmentation and official morphometry metrics for one cell."""
    gt_target = masks_to_target(gt_nucleus, gt_cytoplasm)
    pred_nucleus = pred_target == 2
    pred_cytoplasm = pred_target == 1
    gt_nc = compute_nc_ratio(gt_nucleus, gt_cytoplasm)
    gt_circularity = compute_circularity(gt_nucleus)
    row = {
        "nucleus_dice": dice_score(pred_nucleus, gt_nucleus),
        "cytoplasm_dice": dice_score(pred_cytoplasm, gt_cytoplasm),
        "foreground_dice": foreground_dice(pred_target, gt_target),
        "gt_nucleus_area": int(np.count_nonzero(gt_nucleus)),
        "pred_nucleus_area": int(np.count_nonzero(pred_nucleus)),
        "gt_cytoplasm_area": int(np.count_nonzero(gt_cytoplasm)),
        "pred_cytoplasm_area": int(np.count_nonzero(pred_cytoplasm)),
        "gt_nc": gt_nc,
        "gt_circularity": gt_circularity,
    }
    row["nucleus_area_absolute_error"] = abs(row["pred_nucleus_area"] - row["gt_nucleus_area"])
    row["cytoplasm_area_absolute_error"] = abs(row["pred_cytoplasm_area"] - row["gt_cytoplasm_area"])
    try:
        pred_nc = compute_nc_ratio(pred_nucleus, pred_cytoplasm)
        row.update(
            pred_nc=pred_nc,
            nc_absolute_error=abs(pred_nc - gt_nc),
            nc_relative_error=relative_error(pred_nc, gt_nc),
            invalid_nc=False,
        )
    except ValueError:
        row.update(pred_nc=float("nan"), nc_absolute_error=float("inf"), nc_relative_error=float("inf"), invalid_nc=True)
    try:
        pred_circularity = compute_circularity(pred_nucleus)
        row.update(
            pred_circularity=pred_circularity,
            circularity_absolute_error=abs(pred_circularity - gt_circularity),
            invalid_circularity=False,
        )
    except ValueError:
        row.update(pred_circularity=float("nan"), circularity_absolute_error=float("inf"), invalid_circularity=True)
    return row


def evaluate_model(
    model: UNet,
    val_ids: list[str],
    data_dir: Path,
    image_size: tuple[int, int],
    device: torch.device,
    *,
    seed: int,
    model_name: str,
) -> list[dict]:
    """Run clean validation inference and return per-cell rows."""
    image_by_id = {path.stem: path for path in list_herlev_images(data_dir)}
    model.eval()
    rows = []
    with torch.no_grad():
        for cell_id in val_ids:
            image_path = image_by_id[cell_id]
            image, gt_nucleus, gt_cytoplasm = load_image_and_masks(image_path)
            logits = model(preprocess_rgb_image(image, size=image_size).unsqueeze(0).to(device))
            pred_resized = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_target = restore_prediction_to_original_size(pred_resized, image.shape[:2], image_size)
            row = {"seed": seed, "cell_id": cell_id, "class": image_path.parent.name, "model": model_name}
            row.update(safe_metrics(pred_target, gt_nucleus, gt_cytoplasm))
            rows.append(row)
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    """Summarize per-cell metric rows."""
    return {
        "foreground_dice": finite_mean(row["foreground_dice"] for row in rows),
        "nucleus_dice": finite_mean(row["nucleus_dice"] for row in rows),
        "cytoplasm_dice": finite_mean(row["cytoplasm_dice"] for row in rows),
        "mean_nc_absolute_error": finite_mean(row["nc_absolute_error"] for row in rows),
        "median_nc_absolute_error": finite_median(row["nc_absolute_error"] for row in rows),
        "invalid_nc_rate": float(np.mean([bool(row["invalid_nc"]) for row in rows])) if rows else float("nan"),
        "mean_circularity_absolute_error": finite_mean(row["circularity_absolute_error"] for row in rows),
        "invalid_circularity_rate": float(np.mean([bool(row["invalid_circularity"]) for row in rows])) if rows else float("nan"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def paired_differences(per_cell_rows: list[dict]) -> list[dict]:
    """Create paired baseline vs N/C-aware deltas per seed and cell."""
    grouped: dict[tuple[int, str], dict[str, dict]] = defaultdict(dict)
    for row in per_cell_rows:
        grouped[(int(row["seed"]), str(row["cell_id"]))][str(row["model"])] = row
    out = []
    for (seed, cell_id), models in sorted(grouped.items()):
        if set(models) != {"baseline", "nc_aware_lambda_0p10"}:
            continue
        base = models["baseline"]
        nc = models["nc_aware_lambda_0p10"]
        out.append(
            {
                "seed": seed,
                "cell_id": cell_id,
                "class": base["class"],
                "delta_nc_error": nc["nc_absolute_error"] - base["nc_absolute_error"],
                "delta_foreground_dice": nc["foreground_dice"] - base["foreground_dice"],
                "delta_nucleus_dice": nc["nucleus_dice"] - base["nucleus_dice"],
                "delta_cytoplasm_dice": nc["cytoplasm_dice"] - base["cytoplasm_dice"],
                "delta_circularity_error": nc["circularity_absolute_error"] - base["circularity_absolute_error"],
                "delta_nucleus_area_error": nc["nucleus_area_absolute_error"] - base["nucleus_area_absolute_error"],
                "delta_cytoplasm_area_error": nc["cytoplasm_area_absolute_error"] - base["cytoplasm_area_absolute_error"],
                "baseline_nc_error": base["nc_absolute_error"],
                "nc_aware_nc_error": nc["nc_absolute_error"],
                "baseline_foreground_dice": base["foreground_dice"],
                "nc_aware_foreground_dice": nc["foreground_dice"],
            }
        )
    return out


def bootstrap_mean_delta(deltas: Iterable[float], repeats: int = BOOTSTRAP_REPEATS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired bootstrap CI for the mean N/C error delta."""
    values = np.asarray([value for value in deltas if np.isfinite(value)], dtype=float)
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "repeats": repeats, "seed": seed}
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=float)
    for index in range(repeats):
        means[index] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "repeats": repeats,
        "seed": seed,
    }


def paired_permutation_pvalue(deltas: Iterable[float], repeats: int = BOOTSTRAP_REPEATS, seed: int = BOOTSTRAP_SEED) -> float:
    """Two-sided sign-flip permutation p-value for mean paired delta."""
    values = np.asarray([value for value in deltas if np.isfinite(value)], dtype=float)
    if len(values) == 0:
        return float("nan")
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(repeats):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(values), replace=True)
        if abs(float(np.mean(values * signs))) >= observed:
            count += 1
    return float((count + 1) / (repeats + 1))


def paired_summary(rows: list[dict], scope: str) -> dict:
    """Summarize paired deltas for one scope."""
    deltas = np.asarray([row["delta_nc_error"] for row in rows if np.isfinite(row["delta_nc_error"])], dtype=float)
    if len(deltas) == 0:
        return {"scope": scope, "n": 0}
    improved = float(np.mean(deltas < 0))
    worsened = float(np.mean(deltas > 0))
    tied = float(np.mean(deltas == 0))
    bootstrap = bootstrap_mean_delta(deltas)
    return {
        "scope": scope,
        "n": int(len(deltas)),
        "mean_delta_nc_error": float(np.mean(deltas)),
        "median_delta_nc_error": float(np.median(deltas)),
        "delta_q1": float(np.percentile(deltas, 25)),
        "delta_q3": float(np.percentile(deltas, 75)),
        "delta_p90": float(np.percentile(deltas, 90)),
        "win_rate_nc_aware": improved,
        "win_rate_baseline": worsened,
        "tie_rate": tied,
        "bootstrap_mean_delta": bootstrap["mean"],
        "bootstrap_ci_low": bootstrap["ci_low"],
        "bootstrap_ci_high": bootstrap["ci_high"],
        "permutation_pvalue": paired_permutation_pvalue(deltas),
    }


def decide(seed_summaries: list[dict], combined_pair: dict) -> tuple[str, list[str]]:
    """Apply the fixed GO/CONDITIONAL_GO/NO_GO rules."""
    reasons = []
    seed_deltas = [row for row in seed_summaries if row.get("model") == "paired_delta"]
    mean_improved = sum(row.get("mean_nc_absolute_error_delta", 0) < 0 for row in seed_deltas)
    median_improved = sum(row.get("median_nc_absolute_error_delta", 0) < 0 for row in seed_deltas)
    max_dice_drop = max([row.get("foreground_dice_drop", 0) for row in seed_deltas] or [0])
    win_rate = combined_pair.get("win_rate_nc_aware", float("nan"))
    ci_high = combined_pair.get("bootstrap_ci_high", float("nan"))
    checks = {
        "mean_improves_two_seeds": mean_improved >= 2,
        "median_improves_two_seeds": median_improved >= 2,
        "win_rate_at_least_55pct": win_rate >= 0.55,
        "dice_drop_safe": max_dice_drop <= 0.01,
        "bootstrap_supports_improvement": ci_high < 0,
    }
    for key, value in checks.items():
        reasons.append(f"{key}={value}")
    passed = sum(checks.values())
    if passed >= 4:
        return "GO", reasons
    if passed >= 2 or (win_rate >= 0.50 and mean_improved >= 2):
        return "CONDITIONAL_GO", reasons
    return "NO_GO", reasons
