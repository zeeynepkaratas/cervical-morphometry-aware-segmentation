"""Data-only SIPaKMeD loader for outcome-blind technical preflight.

This module reads RGB images and official contour files only. It contains no
model, checkpoint, inference, or prediction-metric code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


SIPAKMED_CLASS_DIRS = {
    "im_Dyskeratotic": "Dyskeratotic",
    "im_Koilocytotic": "Koilocytotic",
    "im_Metaplastic": "Metaplastic",
    "im_Parabasal": "Parabasal",
    "im_Superficial-Intermediate": "Superficial-Intermediate",
}


@dataclass(frozen=True)
class SipakmedSample:
    sample_id: str
    parent_source_id: str
    diagnostic_class: str
    image_path: Path
    nucleus_contour_path: Path
    cytoplasm_contour_path: Path
    native_width: int
    native_height: int


@dataclass(frozen=True)
class SipakmedLoadedSample:
    sample_id: str
    parent_source_id: str
    diagnostic_class: str
    image: np.ndarray
    nucleus_mask: np.ndarray
    cell_mask: np.ndarray
    cytoplasm_only_mask: np.ndarray
    native_width: int
    native_height: int


def parse_contour_file(path: Path) -> list[tuple[float, float]]:
    """Parse official SIPaKMeD contour files as one ``x,y`` point per line."""
    points: list[tuple[float, float]] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid contour row in {path} line {line_number}: {raw_line!r}")
        try:
            x_coord = float(parts[0])
            y_coord = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"Non-numeric contour row in {path} line {line_number}: {raw_line!r}") from exc
        points.append((x_coord, y_coord))
    if len(points) < 3:
        raise ValueError(f"Contour must contain at least 3 points: {path}")
    return points


def contour_bounds(points: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    coords = list(points)
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    return min(xs), min(ys), max(xs), max(ys)


def contour_in_bounds(points: Iterable[tuple[float, float]], width: int, height: int) -> bool:
    min_x, min_y, max_x, max_y = contour_bounds(points)
    return min_x >= 0 and min_y >= 0 and max_x < width and max_y < height


def rasterize_contour(points: Iterable[tuple[float, float]], width: int, height: int) -> np.ndarray:
    """Rasterize a closed polygon in native image geometry."""
    canvas = Image.new("1", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(list(points), outline=1, fill=1)
    return np.asarray(canvas, dtype=bool)


def sample_id_from_path(image_path: Path, class_dir_name: str) -> str:
    return f"{SIPAKMED_CLASS_DIRS[class_dir_name]}__{image_path.stem}"


def parent_source_id_from_stem(stem: str, class_dir_name: str) -> str:
    return f"{SIPAKMED_CLASS_DIRS[class_dir_name]}__{stem.split('_', 1)[0]}"


def list_sipakmed_samples(raw_dir: Path) -> list[SipakmedSample]:
    """List isolated-cell samples with deterministic image/contour mapping."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"SIPaKMeD raw directory not found: {raw_dir}")

    samples: list[SipakmedSample] = []
    for class_dir_name in sorted(SIPAKMED_CLASS_DIRS):
        class_dir = raw_dir / class_dir_name
        cropped_dir = class_dir / "CROPPED"
        if not cropped_dir.exists():
            continue
        for image_path in sorted(cropped_dir.glob("*.bmp")):
            nucleus_path = image_path.with_name(f"{image_path.stem}_nuc.dat")
            cytoplasm_path = image_path.with_name(f"{image_path.stem}_cyt.dat")
            with Image.open(image_path) as image:
                width, height = image.size
            samples.append(
                SipakmedSample(
                    sample_id=sample_id_from_path(image_path, class_dir_name),
                    parent_source_id=parent_source_id_from_stem(image_path.stem, class_dir_name),
                    diagnostic_class=SIPAKMED_CLASS_DIRS[class_dir_name],
                    image_path=image_path,
                    nucleus_contour_path=nucleus_path,
                    cytoplasm_contour_path=cytoplasm_path,
                    native_width=width,
                    native_height=height,
                )
            )
    return samples


def load_sipakmed_sample(sample: SipakmedSample) -> SipakmedLoadedSample:
    """Load one SIPaKMeD sample and construct native-geometry GT masks.

    The official cytoplasm contour is treated as the outer cell/cytoplasm
    boundary after direct file inspection shows a single filled contour that
    contains the nucleus. The canonical cytoplasm-only mask is therefore the
    cell mask with nucleus pixels removed.
    """
    with Image.open(sample.image_path) as image_pil:
        image = np.asarray(image_pil.convert("RGB"), dtype=np.uint8)

    nucleus_points = parse_contour_file(sample.nucleus_contour_path)
    cytoplasm_points = parse_contour_file(sample.cytoplasm_contour_path)
    nucleus_mask = rasterize_contour(nucleus_points, sample.native_width, sample.native_height)
    cell_mask = rasterize_contour(cytoplasm_points, sample.native_width, sample.native_height)
    cytoplasm_only_mask = np.logical_and(cell_mask, ~nucleus_mask)
    return SipakmedLoadedSample(
        sample_id=sample.sample_id,
        parent_source_id=sample.parent_source_id,
        diagnostic_class=sample.diagnostic_class,
        image=image,
        nucleus_mask=nucleus_mask,
        cell_mask=cell_mask,
        cytoplasm_only_mask=cytoplasm_only_mask,
        native_width=sample.native_width,
        native_height=sample.native_height,
    )
