"""Herlev dataset loading and validation helpers for Phase 1."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - exercised only when OpenCV is absent.
    cv2 = None


EXPECTED_HERLEV_IMAGE_COUNT = 917
IMAGE_EXTENSIONS = {".bmp", ".dib", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_KEYWORDS = {
    "mask",
    "masks",
    "seg",
    "segmentation",
    "label",
    "labels",
    "annotation",
    "annotations",
    "nucleus",
    "nuclei",
    "cytoplasm",
    "cyto",
}
NUCLEUS_KEYWORDS = ("nucleus", "nuclei", "nuc", "nuclear")
CYTOPLASM_KEYWORDS = ("cytoplasm", "cyto", "cell")
OFFICIAL_NUCLEUS_VALUE = 2
OFFICIAL_CYTOPLASM_VALUE = 3


def _normalise_stem(path: Path) -> str:
    stem = path.stem.lower()
    for token in (
        "_nucleus",
        "-nucleus",
        "_nuclei",
        "-nuclei",
        "_nuc",
        "-nuc",
        "_cytoplasm",
        "-cytoplasm",
        "_cyto",
        "-cyto",
        "_mask",
        "-mask",
        "_seg",
        "-seg",
        "_label",
        "-label",
        "-d",
    ):
        stem = stem.replace(token, "")
    return stem


def _is_mask_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.stem.lower()
    return (
        name.endswith("-d")
        or bool(parts & MASK_KEYWORDS)
        or any(keyword in name for keyword in MASK_KEYWORDS)
    )


def _read_image(path: Path, *, color: bool) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if color:
                image = image.convert("RGB")
            return np.asarray(image)
    except Exception as exc:
        raise ValueError(f"Could not read file: {path}") from exc


def _mask_to_bool(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        # Images are read with Pillow, so channel order is RGB, not OpenCV BGR.
        # For mask binarization, any non-zero channel is enough and avoids
        # color-order assumptions in this fallback path.
        mask = mask.max(axis=2)
    return mask > 0


def _find_dataset_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if parent.name.lower() == "herlev":
            return parent
    return path.parent


def _find_mask_file(image_path: Path, keywords: Tuple[str, ...]) -> Optional[Path]:
    raw_dir = _find_dataset_root(image_path)
    target_stem = _normalise_stem(image_path)
    candidates = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path != image_path
        and _is_mask_path(path)
    ]

    matches = [
        path
        for path in candidates
        if _normalise_stem(path) == target_stem
        and any(keyword in path.stem.lower() or keyword in str(path.parent).lower() for keyword in keywords)
    ]
    return sorted(matches)[0] if matches else None


def _find_official_label_mask(image_path: Path) -> Optional[Path]:
    candidate = image_path.with_name(f"{image_path.stem}-d{image_path.suffix.lower()}")
    if candidate.exists():
        return candidate
    candidate = image_path.with_name(f"{image_path.stem}-d{image_path.suffix.upper()}")
    if candidate.exists():
        return candidate
    return None


def _split_official_label_mask(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    nucleus_mask = mask == OFFICIAL_NUCLEUS_VALUE
    cytoplasm_mask = mask == OFFICIAL_CYTOPLASM_VALUE
    return nucleus_mask, cytoplasm_mask


def _load_rgb_with_pillow(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _make_overlay_image(image_path: Path, alpha: float = 0.45) -> Image.Image:
    image_pil = _load_rgb_with_pillow(image_path)
    _, nucleus_mask, cytoplasm_mask = load_image_and_masks(image_path)

    overlay = np.asarray(image_pil).astype(np.float32)
    nucleus_color = np.array([255, 0, 0], dtype=np.float32)
    cytoplasm_color = np.array([0, 80, 255], dtype=np.float32)
    overlay[cytoplasm_mask] = (1 - alpha) * overlay[cytoplasm_mask] + alpha * cytoplasm_color
    overlay[nucleus_mask] = (1 - alpha) * overlay[nucleus_mask] + alpha * nucleus_color
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def _build_mask_index(raw_dir: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and _is_mask_path(path):
            index.setdefault(_normalise_stem(path), []).append(path)
    return index


def list_herlev_images(raw_dir: Path) -> List[Path]:
    """List non-mask image files under the Herlev raw dataset directory."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Herlev directory not found: {raw_dir}")

    images = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and not _is_mask_path(path)
    )

    if len(images) != EXPECTED_HERLEV_IMAGE_COUNT:
        print(
            "WARNING: unexpected Herlev image count "
            f"({len(images)} found, {EXPECTED_HERLEV_IMAGE_COUNT} expected)."
        )

    return images


def load_image_and_masks(image_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one Herlev image and its nucleus/cytoplasm masks."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    label_mask_path = _find_official_label_mask(image_path)
    nucleus_path = _find_mask_file(image_path, NUCLEUS_KEYWORDS) if label_mask_path is None else None
    cytoplasm_path = _find_mask_file(image_path, CYTOPLASM_KEYWORDS) if label_mask_path is None else None
    if label_mask_path is None and (nucleus_path is None or cytoplasm_path is None):
        raise FileNotFoundError(
            "Missing nucleus/cytoplasm mask for "
            f"{image_path.name} (nucleus={nucleus_path}, cytoplasm={cytoplasm_path})"
        )

    image = _read_image(image_path, color=True)
    if label_mask_path is not None:
        nucleus_mask, cytoplasm_mask = _split_official_label_mask(_read_image(label_mask_path, color=False))
    else:
        nucleus_mask = _mask_to_bool(_read_image(nucleus_path, color=False))
        cytoplasm_mask = _mask_to_bool(_read_image(cytoplasm_path, color=False))

    image_shape = image.shape[:2]
    if nucleus_mask.shape != image_shape or cytoplasm_mask.shape != image_shape:
        raise ValueError(
            "Image and mask shapes do not match: "
            f"{image_path.name} image={image_shape}, nucleus={nucleus_mask.shape}, "
            f"cytoplasm={cytoplasm_mask.shape}"
        )
    if not nucleus_mask.any():
        raise ValueError(f"Empty nucleus mask: {nucleus_path}")
    if not cytoplasm_mask.any():
        raise ValueError(f"Empty cytoplasm mask: {cytoplasm_path}")
    if np.logical_and(nucleus_mask, cytoplasm_mask).any():
        raise ValueError(f"Overlapping nucleus/cytoplasm masks: {image_path.name}")

    return image, nucleus_mask, cytoplasm_mask


def validate_herlev_dataset(raw_dir: Path) -> dict:
    """Validate Phase 1 Herlev image count, mask matching, and mask consistency."""
    raw_dir = Path(raw_dir)
    images = list_herlev_images(raw_dir)
    mask_index = _build_mask_index(raw_dir)
    issues = []
    matched_masks = 0
    checked_masks = 0
    mask_values = {"nucleus": set(), "cytoplasm": set()}

    for image_path in images:
        stem = _normalise_stem(image_path)
        official_label_mask = _find_official_label_mask(image_path)
        matching_masks = mask_index.get(stem, [])
        nucleus_candidates = [
            path
            for path in matching_masks
            if any(keyword in path.stem.lower() or keyword in str(path.parent).lower() for keyword in NUCLEUS_KEYWORDS)
        ]
        cytoplasm_candidates = [
            path
            for path in matching_masks
            if any(keyword in path.stem.lower() or keyword in str(path.parent).lower() for keyword in CYTOPLASM_KEYWORDS)
        ]

        if official_label_mask is None and (not nucleus_candidates or not cytoplasm_candidates):
            issues.append(
                {
                    "image": str(image_path),
                    "issue": "missing_mask",
                    "nucleus_candidates": [str(path) for path in nucleus_candidates],
                    "cytoplasm_candidates": [str(path) for path in cytoplasm_candidates],
                }
            )
            continue

        matched_masks += 1
        nucleus_path = sorted(nucleus_candidates)[0] if nucleus_candidates else None
        cytoplasm_path = sorted(cytoplasm_candidates)[0] if cytoplasm_candidates else None

        try:
            image = _read_image(image_path, color=True)
            if official_label_mask is not None:
                label_raw = _read_image(official_label_mask, color=False)
                nucleus_raw = label_raw
                cytoplasm_raw = label_raw
                nucleus_mask, cytoplasm_mask = _split_official_label_mask(label_raw)
            else:
                nucleus_raw = _read_image(nucleus_path, color=False)
                cytoplasm_raw = _read_image(cytoplasm_path, color=False)
                nucleus_mask = _mask_to_bool(nucleus_raw)
                cytoplasm_mask = _mask_to_bool(cytoplasm_raw)
            checked_masks += 1

            mask_values["nucleus"].update(np.unique(nucleus_mask.astype(np.uint8)).astype(int).tolist())
            mask_values["cytoplasm"].update(np.unique(cytoplasm_mask.astype(np.uint8)).astype(int).tolist())

            if nucleus_mask.shape != image.shape[:2] or cytoplasm_mask.shape != image.shape[:2]:
                issues.append({"image": str(image_path), "issue": "shape_mismatch"})
            if not nucleus_mask.any():
                issues.append({"image": str(image_path), "issue": "empty_nucleus_mask"})
            if not cytoplasm_mask.any():
                issues.append({"image": str(image_path), "issue": "empty_cytoplasm_mask"})
            if np.logical_and(nucleus_mask, cytoplasm_mask).any():
                issues.append({"image": str(image_path), "issue": "overlapping_masks"})
        except Exception as exc:
            issues.append({"image": str(image_path), "issue": "mask_validation_error", "detail": str(exc)})

    summary = {
        "raw_dir": str(raw_dir),
        "expected_images": EXPECTED_HERLEV_IMAGE_COUNT,
        "n_images": len(images),
        "n_matched_masks": matched_masks,
        "n_checked_masks": checked_masks,
        "mask_values": {key: sorted(values) for key, values in mask_values.items()},
        "issues": issues,
        "manual_visual_alignment_check": {
            "status": "visual_artifacts_generated_manual_judgement_required",
            "alignment_grid": "results/figures/herlev_alignment_check.png",
            "side_by_side_grid": "results/figures/herlev_side_by_side_check.png",
            "note": "Code verifies shape/value/overlap consistency; final visual alignment is a manual review step.",
        },
        "passed": len(images) == EXPECTED_HERLEV_IMAGE_COUNT and matched_masks == len(images) and not issues,
    }

    project_root = raw_dir.parents[2] if len(raw_dir.parents) >= 3 else Path.cwd()
    results_dir = project_root / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "herlev_validation_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    return summary


def generate_alignment_check_grid(
    raw_dir: Path,
    output_path: Path,
    n_samples: int = 20,
    seed: int = 42,
) -> Path:
    """Create a reproducible contact sheet of image/mask overlays for manual review."""
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    images = list_herlev_images(raw_dir)
    if not images:
        raise ValueError(f"No Herlev images found under: {raw_dir}")

    rng = random.Random(seed)
    selected_images = rng.sample(images, min(n_samples, len(images)))

    n_cols = 4
    n_rows = int(np.ceil(len(selected_images) / n_cols))
    tile_size = 180
    label_height = 34
    padding = 10
    sheet_width = n_cols * tile_size + (n_cols + 1) * padding
    sheet_height = n_rows * (tile_size + label_height) + (n_rows + 1) * padding
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, image_path in enumerate(selected_images):
        overlay_pil = _make_overlay_image(image_path)
        overlay_pil.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)

        row, col = divmod(index, n_cols)
        cell_x = padding + col * (tile_size + padding)
        cell_y = padding + row * (tile_size + label_height + padding)
        image_x = cell_x + (tile_size - overlay_pil.width) // 2
        sheet.paste(overlay_pil, (image_x, cell_y))

        label = image_path.name
        if len(label) > 32:
            label = f"{label[:14]}...{label[-15:]}"
        text_y = cell_y + tile_size + 4
        draw.text((cell_x, text_y), label, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def generate_side_by_side_check(
    raw_dir: Path,
    output_path: Path,
    n_samples: int = 20,
    seed: int = 42,
) -> Path:
    """Create a reproducible raw-vs-overlay contact sheet for manual inspection."""
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    images = list_herlev_images(raw_dir)
    if not images:
        raise ValueError(f"No Herlev images found under: {raw_dir}")

    rng = random.Random(seed)
    selected_images = rng.sample(images, min(n_samples, len(images)))

    panel_size = 180
    label_height = 28
    gap = 12
    padding = 12
    pair_width = panel_size * 2 + gap
    row_height = panel_size + label_height
    sheet_width = pair_width + padding * 2
    sheet_height = len(selected_images) * row_height + (len(selected_images) + 1) * padding
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, image_path in enumerate(selected_images):
        raw_pil = _load_rgb_with_pillow(image_path)
        overlay_pil = _make_overlay_image(image_path)
        raw_pil.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)
        overlay_pil.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)

        row_y = padding + index * (row_height + padding)
        left_x = padding + (panel_size - raw_pil.width) // 2
        right_panel_x = padding + panel_size + gap
        right_x = right_panel_x + (panel_size - overlay_pil.width) // 2
        sheet.paste(raw_pil, (left_x, row_y))
        sheet.paste(overlay_pil, (right_x, row_y))

        label = image_path.name
        if len(label) > 46:
            label = f"{label[:21]}...{label[-22:]}"
        text_width = draw.textlength(label, font=font)
        text_x = padding + max(0, int((pair_width - text_width) / 2))
        draw.text((text_x, row_y + panel_size + 4), label, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path
