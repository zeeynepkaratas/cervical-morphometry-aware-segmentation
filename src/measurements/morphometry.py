"""Morphometric measurement helpers.

Locked project measurements:
    - N/C ratio: area-based, robust
    - Nucleus circularity: boundary-based, sensitive

Do not add a fifth mandatory measurement here. Optional exploratory
measurements remain isolated in ``measure_exploratory``.
"""

import numpy as np

from src.utils.config import RE_EPSILON

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - fallback is used when OpenCV is absent.
    cv2 = None


def compute_nc_ratio(nucleus_mask: np.ndarray, cytoplasm_mask: np.ndarray) -> float:
    """
    Compute the area-based nucleus/cytoplasm ratio.

    Locked convention for this project:
        N/C ratio = nucleus_area / cytoplasm_area

    Here ``cytoplasm_area`` means cytoplasm-only pixels, excluding nucleus pixels.
    This matches the Herlev ``*-d.bmp`` labels after splitting them into
    pixel-exclusive boolean masks (2=nucleus, 3=cytoplasm). It is intentionally
    not nucleus_area / whole_cell_area.
    """
    nucleus_mask = np.asarray(nucleus_mask).astype(bool)
    cytoplasm_mask = np.asarray(cytoplasm_mask).astype(bool)

    if nucleus_mask.shape != cytoplasm_mask.shape:
        raise ValueError(
            "nucleus_mask and cytoplasm_mask must have the same shape: "
            f"{nucleus_mask.shape} != {cytoplasm_mask.shape}"
        )
    if np.logical_and(nucleus_mask, cytoplasm_mask).any():
        raise ValueError("nucleus_mask and cytoplasm_mask must be pixel-exclusive.")

    cytoplasm_area = int(np.count_nonzero(cytoplasm_mask))
    if cytoplasm_area == 0:
        raise ValueError("cytoplasm_mask is empty; N/C ratio is undefined.")

    nucleus_area = int(np.count_nonzero(nucleus_mask))
    return float(nucleus_area / cytoplasm_area)


def compute_circularity(nucleus_mask: np.ndarray) -> float:
    """
    Compute nucleus circularity.

    Circularity = 4*pi*area / perimeter^2.

    Locked official implementation uses OpenCV contours:
        ``cv2.contourArea`` for area and ``cv2.arcLength`` for perimeter.

    If OpenCV is unavailable, a NumPy fallback uses pixel area and a boundary
    transition perimeter approximation only so the function remains executable
    in lightweight environments. Reported/production circularity values should
    be generated with OpenCV.
    """
    nucleus_mask = np.asarray(nucleus_mask).astype(bool)
    if nucleus_mask.ndim != 2:
        raise ValueError(f"nucleus_mask must be 2D, got shape {nucleus_mask.shape}")
    if not nucleus_mask.any():
        raise ValueError("nucleus_mask is empty; circularity is undefined.")

    if cv2 is not None:
        mask_uint8 = nucleus_mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No nucleus contour found; circularity is undefined.")
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
    else:
        area = float(np.count_nonzero(nucleus_mask))
        padded = np.pad(nucleus_mask, pad_width=1, mode="constant", constant_values=False)
        horizontal_edges = np.logical_xor(padded[:, 1:], padded[:, :-1]).sum()
        vertical_edges = np.logical_xor(padded[1:, :], padded[:-1, :]).sum()
        perimeter = float(horizontal_edges + vertical_edges)

    if area <= 0 or perimeter <= 0:
        raise ValueError("Nucleus area/perimeter is zero; circularity is undefined.")

    circularity = float(4.0 * np.pi * area / (perimeter ** 2))
    return min(circularity, 1.0)


def relative_error(predicted: float, reference: float, eps: float = RE_EPSILON) -> float:
    """RE = |predicted - reference| / (|reference| + eps)."""
    return abs(predicted - reference) / (abs(reference) + eps)