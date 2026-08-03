"""
Controlled image degradations for Phase 5.

The four locked corruption families are intentionally fixed here and in
``src.utils.config.DEGRADATION_SEVERITY_LEVELS``. They are robustness-test
levels, not a tuned search space.
"""

import hashlib

import cv2
import numpy as np

from src.utils.config import DEGRADATION_SEVERITY_LEVELS


def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur with a fixed sigma.

    The locked severities sigma=[1,2,3] are common robustness-test levels:
    small, moderate, and strong optical defocus while keeping the cell visible.
    Kernel size is derived as roughly +/-3 sigma and forced to be odd.
    """
    image = np.asarray(image, dtype=np.uint8)
    radius = max(1, int(round(3.0 * float(sigma))))
    kernel_size = 2 * radius + 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=float(sigma), sigmaY=float(sigma))


def apply_gaussian_noise(image: np.ndarray, std: float, seed: int | None = None) -> np.ndarray:
    """
    Add zero-mean Gaussian sensor noise in uint8 intensity units.

    std=[10,20,30] follows the usual image-corruption convention of increasing
    additive noise on a 0..255 scale. When ``seed`` is provided, a local NumPy
    generator is used so the same image/variant is reproducible independent of
    loop order or earlier random draws.
    """
    image_float = np.asarray(image, dtype=np.float32)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=float(std), size=image_float.shape)
    return np.clip(image_float + noise, 0, 255).astype(np.uint8)


def apply_contrast_change(image: np.ndarray, alpha: float) -> np.ndarray:
    """
    Change contrast around the image's per-channel mean.

    alpha=[0.75,1.25,1.50] covers one low-contrast and two high-contrast
    shifts. The mean-pivot formula ``mean + alpha * (image - mean)`` preserves
    average brightness more cleanly than raw intensity scaling, so this
    degradation isolates contrast rather than mixing contrast with darkening or
    brightening.
    """
    image_float = np.asarray(image, dtype=np.float32)
    channel_mean = image_float.mean(axis=(0, 1), keepdims=True)
    contrasted = channel_mean + float(alpha) * (image_float - channel_mean)
    return np.clip(contrasted, 0, 255).astype(np.uint8)


def apply_low_resolution(image: np.ndarray, scale: float) -> np.ndarray:
    """
    Simulate low resolution by downsampling and then restoring original size.

    scale=[0.75,0.50,0.25] gives mild, medium, and severe spatial-detail loss.
    Downsampling uses area interpolation; upsampling uses nearest neighbor to
    preserve the blocky acquisition artifact intentionally.
    """
    image = np.asarray(image, dtype=np.uint8)
    if not 0 < float(scale) <= 1:
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    height, width = image.shape[:2]
    down_width = max(1, int(round(width * float(scale))))
    down_height = max(1, int(round(height * float(scale))))
    down = cv2.resize(image, (down_width, down_height), interpolation=cv2.INTER_AREA)
    return cv2.resize(down, (width, height), interpolation=cv2.INTER_NEAREST)


DEGRADATION_FUNCTIONS = {
    "gaussian_blur": apply_gaussian_blur,
    "gaussian_noise": apply_gaussian_noise,
    "contrast_change": apply_contrast_change,
    "low_resolution": apply_low_resolution,
}


def _stable_seed(base_seed: int, variant_key: str) -> int:
    key = f"{base_seed}|{variant_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], byteorder="little", signed=False)


def generate_all_variants(image: np.ndarray, seed: int | None = None) -> dict:
    """
    Generate every locked degradation x severity variant for one clean image.

    Returns keys such as ``gaussian_blur_1`` and ``gaussian_noise_20``. The
    severity grid is fixed in config, not searched or tuned. If ``seed`` is
    supplied, stochastic variants are reproducible independent of dictionary
    iteration order.
    """
    variants = {}
    for degradation_name, levels in DEGRADATION_SEVERITY_LEVELS.items():
        if degradation_name not in DEGRADATION_FUNCTIONS:
            raise KeyError(f"No degradation function registered for {degradation_name}")
        for level in levels:
            level_label = f"{float(level):g}".replace(".", "p")
            key = f"{degradation_name}_{level_label}"
            if degradation_name == "gaussian_noise" and seed is not None:
                variants[key] = DEGRADATION_FUNCTIONS[degradation_name](image, level, seed=_stable_seed(seed, key))
            else:
                variants[key] = DEGRADATION_FUNCTIONS[degradation_name](image, level)
    return variants
