"""Controlled mask perturbation benchmark — Phase B.

Applies five families of deterministic perturbations to ground-truth nucleus
masks and measures how each perturbation changes morphometric metrics.

Perturbation families
---------------------
1. Erosion (radius 1 px, radius 2 px)
2. Dilation (radius 1 px, radius 2 px) — clipped to stay within cytoplasm
3. Boundary jitter (low ±1 px, high ±2 px) — deterministic via cell_id hash
4. Small protrusions (low / high) — deterministic bumps along the contour
5. Small indentations (low / high) — deterministic dents along the contour

Safety guarantees
-----------------
- Nucleus is never empty after perturbation.
- Dilated nucleus is clipped to cytoplasm boundary (union mask).
- Nucleus stays within image bounds.
- Invalid samples are reported (not silently dropped).
- All perturbations are deterministic via ``cell_id + perturbation + severity``.

Outputs
-------
results/circularity_mechanism/controlled_mask_perturbations.csv
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Callable, Generator

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.measurements.morphometry import compute_circularity, compute_nc_ratio

def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(pred, target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0
    return float(2.0 * intersection / union)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLIT_FILE = ROOT / "data" / "splits" / "original_herlev_group_split.json"
OUT_DIR = ROOT / "results" / "circularity_mechanism"
HERLEV_FALLBACK = ROOT.parent / "cervical-morphometry-conformal" / "data" / "raw" / "herlev"

# ---------------------------------------------------------------------------
# Perturbation registry
# ---------------------------------------------------------------------------
PERTURBATIONS: list[tuple[str, str]] = [
    ("erosion",      "r1"),
    ("erosion",      "r2"),
    ("dilation",     "r1"),
    ("dilation",     "r2"),
    ("jitter",       "low"),
    ("jitter",       "high"),
    ("protrusion",   "low"),
    ("protrusion",   "high"),
    ("indentation",  "low"),
    ("indentation",  "high"),
]

# ---------------------------------------------------------------------------
# Deterministic seed from cell_id + perturbation + severity
# ---------------------------------------------------------------------------

def _perturb_seed(cell_id: str, perturbation: str, severity: str) -> int:
    key = f"{cell_id}|{perturbation}|{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little", signed=False)


# ---------------------------------------------------------------------------
# Morphometry helpers
# ---------------------------------------------------------------------------

def _safe_perimeter(nucleus_mask: np.ndarray) -> float:
    """OpenCV contour perimeter — consistent with compute_circularity."""
    mask_u8 = nucleus_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float("nan")
    contour = max(contours, key=cv2.contourArea)
    return float(cv2.arcLength(contour, True))


def _safe_area(nucleus_mask: np.ndarray) -> float:
    """OpenCV contour area — consistent with compute_circularity."""
    mask_u8 = nucleus_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float(np.count_nonzero(nucleus_mask))
    return float(cv2.contourArea(max(contours, key=cv2.contourArea)))


def _morphometry(nucleus_mask: np.ndarray, cytoplasm_mask: np.ndarray) -> dict:
    area = _safe_area(nucleus_mask)
    perim = _safe_perimeter(nucleus_mask)
    try:
        circ = compute_circularity(nucleus_mask)
        invalid_circ = False
    except ValueError:
        circ = float("nan")
        invalid_circ = True
    try:
        nc = compute_nc_ratio(nucleus_mask, cytoplasm_mask)
        invalid_nc = False
    except ValueError:
        nc = float("nan")
        invalid_nc = True
    return {
        "area": area,
        "perimeter": perim,
        "circularity": circ,
        "nc": nc,
        "invalid_circularity": invalid_circ,
        "invalid_nc": invalid_nc,
    }


# ---------------------------------------------------------------------------
# Disk mask — used for erosion/dilation
# ---------------------------------------------------------------------------

def _disk_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (x ** 2 + y ** 2 <= radius ** 2).astype(np.uint8)
    return kernel


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------

def perturb_erosion(
    nucleus: np.ndarray, cytoplasm: np.ndarray, severity: str, cell_id: str
) -> tuple[np.ndarray, bool, str]:
    """Erode nucleus by given radius."""
    radius = 1 if severity == "r1" else 2
    kernel = _disk_kernel(radius)
    eroded = cv2.erode(nucleus.astype(np.uint8), kernel, iterations=1).astype(bool)
    if not eroded.any():
        return nucleus, False, "erosion_emptied_nucleus"
    return eroded, True, ""


def perturb_dilation(
    nucleus: np.ndarray, cytoplasm: np.ndarray, severity: str, cell_id: str
) -> tuple[np.ndarray, bool, str]:
    """Dilate nucleus clipped to cytoplasm ∪ nucleus."""
    radius = 1 if severity == "r1" else 2
    kernel = _disk_kernel(radius)
    dilated = cv2.dilate(nucleus.astype(np.uint8), kernel, iterations=1).astype(bool)
    # Cytoplasm mask here is the cytoplasm-only region (exclusive of nucleus)
    # The cell boundary is nucleus | cytoplasm
    cell_boundary = nucleus | cytoplasm
    dilated = dilated & cell_boundary
    if not dilated.any():
        return nucleus, False, "dilation_emptied_nucleus"
    return dilated, True, ""


def perturb_jitter(
    nucleus: np.ndarray, cytoplasm: np.ndarray, severity: str, cell_id: str
) -> tuple[np.ndarray, bool, str]:
    """Deterministic boundary jitter.

    Displaces each contour point by ±max_px in a random direction.
    Low = ±1 px, High = ±2 px.
    """
    max_px = 1 if severity == "low" else 2
    seed = _perturb_seed(cell_id, "jitter", severity)
    rng = np.random.default_rng(seed)

    mask_u8 = nucleus.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return nucleus, False, "jitter_no_contour"

    h, w = nucleus.shape
    contour = max(contours, key=cv2.contourArea).squeeze(axis=1)  # (N,2) xy

    # Displace each point
    displacements = rng.integers(-max_px, max_px + 1, size=contour.shape)
    jittered = contour + displacements

    # Clip to image bounds
    jittered[:, 0] = np.clip(jittered[:, 0], 0, w - 1)
    jittered[:, 1] = np.clip(jittered[:, 1], 0, h - 1)

    # Draw filled polygon
    new_mask = np.zeros_like(nucleus, dtype=np.uint8)
    cv2.fillPoly(new_mask, [jittered.reshape(-1, 1, 2).astype(np.int32)], 1)
    new_mask = new_mask.astype(bool)

    if not new_mask.any():
        return nucleus, False, "jitter_empty_result"
    return new_mask, True, ""


def _add_contour_bumps(
    nucleus: np.ndarray,
    cytoplasm: np.ndarray,
    max_px: int,
    spacing: int,
    outward: bool,
    cell_id: str,
    severity: str,
    perturb_type: str,
) -> tuple[np.ndarray, bool, str]:
    """Add outward bumps (protrusions) or inward dents (indentations)."""
    seed = _perturb_seed(cell_id, perturb_type, severity)
    rng = np.random.default_rng(seed)

    mask_u8 = nucleus.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return nucleus, False, "bump_no_contour"

    h, w = nucleus.shape
    contour = max(contours, key=cv2.contourArea).squeeze(axis=1)
    N = len(contour)

    if N < 8:
        return nucleus, False, "bump_contour_too_small"

    # Select evenly spaced bump centres with random phase
    phase = rng.integers(0, spacing)
    bump_indices = list(range(phase, N, spacing))

    # For each bump, push contour points ±max_px in normal direction
    new_mask = nucleus.astype(np.uint8).copy()
    for bi in bump_indices:
        # Estimate local outward normal via neighbouring points
        prev_pt = contour[(bi - 1) % N].astype(float)
        next_pt = contour[(bi + 1) % N].astype(float)
        tangent = next_pt - prev_pt
        tang_len = np.linalg.norm(tangent)
        if tang_len < 1e-6:
            continue
        tangent /= tang_len
        normal = np.array([-tangent[1], tangent[0]])  # rotate 90° CCW = outward

        mag = rng.integers(1, max_px + 1)
        direction = normal if outward else -normal
        delta = (direction * mag).astype(int)

        # Draw a small disk at displaced point
        cx = int(contour[bi, 0]) + delta[0]
        cy = int(contour[bi, 1]) + delta[1]
        cx = np.clip(cx, 0, w - 1)
        cy = np.clip(cy, 0, h - 1)
        cv2.circle(new_mask, (cx, cy), max_px, 1 if outward else 0, -1)

    result = new_mask.astype(bool)

    # For protrusions: clip to cell boundary
    if outward:
        cell_boundary = nucleus | cytoplasm
        result = result & cell_boundary

    if not result.any():
        return nucleus, False, "bump_empty_result"

    return result, True, ""


def perturb_protrusion(
    nucleus: np.ndarray, cytoplasm: np.ndarray, severity: str, cell_id: str
) -> tuple[np.ndarray, bool, str]:
    """Add small outward protrusions. Low: 1 px bump every ~20 pts. High: 2 px every ~10 pts."""
    if severity == "low":
        return _add_contour_bumps(nucleus, cytoplasm, 1, 20, True, cell_id, severity, "protrusion")
    else:
        return _add_contour_bumps(nucleus, cytoplasm, 2, 10, True, cell_id, severity, "protrusion")


def perturb_indentation(
    nucleus: np.ndarray, cytoplasm: np.ndarray, severity: str, cell_id: str
) -> tuple[np.ndarray, bool, str]:
    """Add small inward indentations. Low: 1 px dent every ~20 pts. High: 2 px every ~10 pts."""
    if severity == "low":
        return _add_contour_bumps(nucleus, cytoplasm, 1, 20, False, cell_id, severity, "indentation")
    else:
        return _add_contour_bumps(nucleus, cytoplasm, 2, 10, False, cell_id, severity, "indentation")


PERTURB_FN: dict[str, Callable] = {
    "erosion":     perturb_erosion,
    "dilation":    perturb_dilation,
    "jitter":      perturb_jitter,
    "protrusion":  perturb_protrusion,
    "indentation": perturb_indentation,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resolve_herlev_dir() -> Path:
    import os
    env_dir = os.environ.get("HERLEV_DATA_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir).resolve()
    if HERLEV_FALLBACK.exists():
        return HERLEV_FALLBACK.resolve()
    raise FileNotFoundError(
        "Herlev data directory not found. "
        "Set HERLEV_DATA_DIR or ensure the source repo is at the fallback path."
    )


def _load_splits() -> dict[str, list[str]]:
    data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    return data


def _load_herlev_index(herlev_dir: Path) -> dict[str, Path]:
    """Build {cell_id: image_path} index from the Herlev directory."""
    from src.data_prep.load_herlev import list_herlev_images
    images = list_herlev_images(herlev_dir)
    return {p.stem: p for p in images}


# ---------------------------------------------------------------------------
# Per-cell benchmark
# ---------------------------------------------------------------------------

def _run_cell(
    cell_id: str,
    nucleus_gt: np.ndarray,
    cytoplasm_gt: np.ndarray,
) -> list[dict]:
    """Apply all perturbations to a single cell and return metric rows."""
    base = _morphometry(nucleus_gt, cytoplasm_gt)
    rows = []

    for pert_type, severity in PERTURBATIONS:
        fn = PERTURB_FN[pert_type]
        try:
            perturbed, valid, reason = fn(nucleus_gt, cytoplasm_gt, severity, cell_id)
        except Exception as exc:
            valid = False
            reason = str(exc)
            perturbed = nucleus_gt

        if not valid:
            row = {
                "cell_id": cell_id,
                "perturbation": pert_type,
                "severity": severity,
                "valid": False,
                "failure_reason": reason,
            }
            rows.append(row)
            continue

        # Cytoplasm mask for perturbed: original cell minus new nucleus
        cell_mask = nucleus_gt | cytoplasm_gt
        perturbed_cytoplasm = cell_mask & ~perturbed

        m = _morphometry(perturbed, perturbed_cytoplasm)

        row = {
            "cell_id": cell_id,
            "perturbation": pert_type,
            "severity": severity,
            "valid": True,
            "failure_reason": "",
            # Dice between perturbed and GT nucleus
            "nucleus_dice": dice_score(perturbed, nucleus_gt),
            # IoU
            "nucleus_iou": (
                float(np.logical_and(perturbed, nucleus_gt).sum())
                / (float(np.logical_or(perturbed, nucleus_gt).sum()) + 1e-8)
            ),
            # Area
            "gt_area": base["area"],
            "perturbed_area": m["area"],
            "area_change": m["area"] - base["area"],
            "area_change_rel": (m["area"] - base["area"]) / (base["area"] + 1e-6),
            # Perimeter
            "gt_perimeter": base["perimeter"],
            "perturbed_perimeter": m["perimeter"],
            "perimeter_change": m["perimeter"] - base["perimeter"],
            "perimeter_change_rel": (m["perimeter"] - base["perimeter"]) / (base["perimeter"] + 1e-6),
            # Circularity
            "gt_circularity": base["circularity"],
            "perturbed_circularity": m["circularity"],
            "circularity_change": m["circularity"] - base["circularity"],
            # N/C
            "gt_nc": base["nc"],
            "perturbed_nc": m["nc"],
            "nc_change": m["nc"] - base["nc"],
            # Flags
            "invalid_circularity": m["invalid_circularity"],
            "invalid_nc": m["invalid_nc"],
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(splits_to_use: list[str] | None = None, out_dir: Path = OUT_DIR) -> None:
    """Run controlled perturbation benchmark.

    Args:
        splits_to_use: list of split names to include (default: ['unet_val', 'calibration', 'test']).
            Design/validation on unet_val, final evaluation on test.
    """
    if splits_to_use is None:
        splits_to_use = ["unet_val", "calibration", "test"]

    out_dir.mkdir(parents=True, exist_ok=True)
    herlev_dir = _resolve_herlev_dir()
    print(f"Herlev data directory: {herlev_dir}")

    splits = _load_splits()
    image_index = _load_herlev_index(herlev_dir)

    from src.data_prep.load_herlev import load_image_and_masks

    all_rows = []
    n_cells = 0
    n_invalid = 0

    for split_name in splits_to_use:
        cell_ids = splits.get(split_name, [])
        print(f"Processing split '{split_name}': {len(cell_ids)} cells ...")

        for cell_id in cell_ids:
            if cell_id not in image_index:
                print(f"  WARNING: {cell_id} not in Herlev index; skipping.")
                continue

            try:
                _, nucleus_gt, cytoplasm_gt = load_image_and_masks(image_index[cell_id])
            except Exception as exc:
                print(f"  WARNING: could not load {cell_id}: {exc}; skipping.")
                continue

            rows = _run_cell(cell_id, nucleus_gt, cytoplasm_gt)
            for r in rows:
                r["split"] = split_name
            all_rows.extend(rows)
            n_cells += 1
            n_invalid += sum(1 for r in rows if not r["valid"])

    import pandas as pd
    df = pd.DataFrame(all_rows)
    out_path = out_dir / "controlled_mask_perturbations.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDone. {n_cells} cells × {len(PERTURBATIONS)} perturbations.")
    print(f"Invalid perturbation instances: {n_invalid}")
    print(f"Written: {out_path}  ({len(df):,} rows)")


if __name__ == "__main__":
    run()
