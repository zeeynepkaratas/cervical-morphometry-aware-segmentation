"""Run the outcome-blind SIPaKMeD data-only technical preflight.

No checkpoint loading, model inference, or prediction-based metric is used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_prep.load_sipakmed import (
    SIPAKMED_CLASS_DIRS,
    contour_bounds,
    contour_in_bounds,
    list_sipakmed_samples,
    load_sipakmed_sample,
    parse_contour_file,
)


RAW_DIR = ROOT / "data" / "raw" / "sipakmed"
OUT_DIR = ROOT / "results" / "sipakmed_preflight"

ARCHIVE_EXPECTED_BYTES = {
    "im_Dyskeratotic.7z": 1182980549,
    "im_Koilocytotic.7z": 1290478065,
    "im_Metaplastic.7z": 1396480123,
    "im_Parabasal.7z": 548667699,
    "im_Superficial-Intermediate.7z": 762939737,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256_small(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def polygon_signed_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        area += x0 * y1 - x1 * y0
    return area / 2.0


def mask_perimeter(mask: np.ndarray) -> int:
    padded = np.pad(mask.astype(np.uint8), 1)
    center = padded[1:-1, 1:-1]
    exposed = (
        (center & (padded[:-2, 1:-1] == 0)).sum()
        + (center & (padded[2:, 1:-1] == 0)).sum()
        + (center & (padded[1:-1, :-2] == 0)).sum()
        + (center & (padded[1:-1, 2:] == 0)).sum()
    )
    return int(exposed)


def overlay_panel(samples: list[dict], output_path: Path) -> None:
    tile_w, tile_h = 220, 220
    label_h = 34
    tiles = []
    for row in samples:
        loaded = load_sipakmed_sample(row["sample"])
        image = Image.fromarray(loaded.image)
        rgb = image.resize((tile_w, tile_h), Image.Resampling.BILINEAR).convert("RGB")

        nuc = Image.fromarray((loaded.nucleus_mask.astype(np.uint8) * 255), mode="L").resize(
            (tile_w, tile_h), Image.Resampling.NEAREST
        )
        cyt = Image.fromarray((loaded.cell_mask.astype(np.uint8) * 255), mode="L").resize(
            (tile_w, tile_h), Image.Resampling.NEAREST
        )
        overlay = np.asarray(rgb).astype(np.float32)
        cyt_arr = np.asarray(cyt) > 0
        nuc_arr = np.asarray(nuc) > 0
        overlay[cyt_arr] = 0.70 * overlay[cyt_arr] + 0.30 * np.array([0, 90, 255], dtype=np.float32)
        overlay[nuc_arr] = 0.60 * overlay[nuc_arr] + 0.40 * np.array([255, 0, 0], dtype=np.float32)
        tile = Image.new("RGB", (tile_w, tile_h + label_h), "white")
        tile.paste(Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)), (0, label_h))
        draw = ImageDraw.Draw(tile)
        draw.text((6, 5), row["label"][:34], fill=(0, 0, 0))
        tiles.append(tile)

    cols = min(4, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    panel = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
    for index, tile in enumerate(tiles):
        x = (index % cols) * tile_w
        y = (index // cols) * (tile_h + label_h)
        panel.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    archive_status = []
    for name, expected_bytes in sorted(ARCHIVE_EXPECTED_BYTES.items()):
        path = RAW_DIR / name
        size = path.stat().st_size if path.exists() else None
        archive_status.append(
            {
                "name": name,
                "exists": path.exists(),
                "bytes": size,
                "expected_bytes": expected_bytes,
                "size_matches_expected": size == expected_bytes,
                "sha256": sha256_file(path) if path.exists() else None,
            }
        )

    class_dirs = [RAW_DIR / name for name in sorted(SIPAKMED_CLASS_DIRS)]
    class_summary = []
    for class_dir in class_dirs:
        cropped = class_dir / "CROPPED"
        class_summary.append(
            {
                "class_dir": class_dir.name,
                "diagnostic_class": SIPAKMED_CLASS_DIRS[class_dir.name],
                "root_bmp": len(list(class_dir.glob("*.bmp"))),
                "root_dat": len(list(class_dir.glob("*.dat"))),
                "cropped_bmp": len(list(cropped.glob("*.bmp"))),
                "cropped_dat": len(list(cropped.glob("*.dat"))),
                "cropped_nucleus_dat": len(list(cropped.glob("*_nuc.dat"))),
                "cropped_cytoplasm_dat": len(list(cropped.glob("*_cyt.dat"))),
            }
        )

    samples = list_sipakmed_samples(RAW_DIR)
    sample_ids = [sample.sample_id for sample in samples]
    duplicate_sample_ids = sorted([sample_id for sample_id, count in Counter(sample_ids).items() if count > 1])

    rows = []
    dimension_counter = Counter()
    mode_counter = Counter()
    dtype_counter = Counter()
    image_hashes: dict[str, list[str]] = defaultdict(list)
    contour_hashes: dict[str, list[str]] = defaultdict(list)
    exclusion_counter = Counter()
    eligible_samples = []
    coordinate_xy_in_bounds = 0
    coordinate_yx_in_bounds = 0
    total_contours = 0
    cyt_contains_nucleus = 0
    nucleus_outside_cell_count = 0
    max_nucleus_outside_pixels = 0
    contour_point_counts = []
    contour_orientations = Counter()
    malformed_samples = []

    for sample in samples:
        technical_valid = True
        reasons: list[str] = []
        nucleus_points: list[tuple[float, float]] | None = None
        cytoplasm_points: list[tuple[float, float]] | None = None

        if not sample.image_path.exists():
            technical_valid = False
            reasons.append("missing_image")
        if not sample.nucleus_contour_path.exists():
            technical_valid = False
            reasons.append("missing_nucleus_contour")
        if not sample.cytoplasm_contour_path.exists():
            technical_valid = False
            reasons.append("missing_cytoplasm_contour")

        try:
            with Image.open(sample.image_path) as image:
                image.verify()
            with Image.open(sample.image_path) as image:
                mode = image.mode
                width, height = image.size
                array = np.asarray(image.convert("RGB"))
                dtype = array.dtype.name
        except Exception:
            technical_valid = False
            reasons.append("corrupt_or_unreadable_image")
            mode = ""
            width = sample.native_width
            height = sample.native_height
            dtype = ""

        if sample.image_path.exists():
            image_hashes[file_sha256_small(sample.image_path)].append(sample.sample_id)
        dimension_counter[(width, height)] += 1
        mode_counter[mode] += 1
        dtype_counter[dtype] += 1

        try:
            nucleus_points = parse_contour_file(sample.nucleus_contour_path)
            contour_hashes[file_sha256_small(sample.nucleus_contour_path)].append(f"{sample.sample_id}:nucleus")
        except Exception:
            technical_valid = False
            reasons.append("invalid_nucleus_contour_format")
        try:
            cytoplasm_points = parse_contour_file(sample.cytoplasm_contour_path)
            contour_hashes[file_sha256_small(sample.cytoplasm_contour_path)].append(f"{sample.sample_id}:cytoplasm")
        except Exception:
            technical_valid = False
            reasons.append("invalid_cytoplasm_contour_format")

        if nucleus_points is not None and cytoplasm_points is not None:
            for points in (nucleus_points, cytoplasm_points):
                total_contours += 1
                coordinate_xy_in_bounds += int(contour_in_bounds(points, width, height))
                coordinate_yx_in_bounds += int(
                    all(0 <= y < width and 0 <= x < height for x, y in points)
                )
                contour_point_counts.append(len(points))
                contour_orientations["clockwise" if polygon_signed_area(points) < 0 else "counterclockwise"] += 1
            if not contour_in_bounds(nucleus_points, width, height):
                technical_valid = False
                reasons.append("nucleus_contour_out_of_bounds")
            if not contour_in_bounds(cytoplasm_points, width, height):
                technical_valid = False
                reasons.append("cytoplasm_contour_out_of_bounds")
            try:
                loaded = load_sipakmed_sample(sample)
                nucleus_area = int(loaded.nucleus_mask.sum())
                cell_area = int(loaded.cell_mask.sum())
                cytoplasm_only_area = int(loaded.cytoplasm_only_mask.sum())
                nucleus_outside = int(np.logical_and(loaded.nucleus_mask, ~loaded.cell_mask).sum())
                max_nucleus_outside_pixels = max(max_nucleus_outside_pixels, nucleus_outside)
                nucleus_outside_cell_count += int(nucleus_outside > 0)
                cyt_contains_nucleus += int(nucleus_outside == 0)
                perimeter = mask_perimeter(loaded.nucleus_mask)
                circularity_defined = nucleus_area > 0 and perimeter > 0
                nc_defined = cytoplasm_only_area > 0
                if nucleus_area <= 0:
                    technical_valid = False
                    reasons.append("empty_nucleus_mask")
                if cell_area <= 0:
                    technical_valid = False
                    reasons.append("empty_cell_mask")
                if cytoplasm_only_area <= 0:
                    technical_valid = False
                    reasons.append("empty_cytoplasm_only_mask")
                if nucleus_outside > 0:
                    technical_valid = False
                    reasons.append("nucleus_pixels_outside_cell_mask")
                if not circularity_defined:
                    technical_valid = False
                    reasons.append("undefined_nucleus_circularity")
                if not nc_defined:
                    technical_valid = False
                    reasons.append("undefined_nc_ratio")
            except Exception:
                technical_valid = False
                reasons.append("rasterization_failure")
                nucleus_area = cell_area = cytoplasm_only_area = perimeter = nucleus_outside = 0
                circularity_defined = False
                nc_defined = False
        else:
            nucleus_area = cell_area = cytoplasm_only_area = perimeter = nucleus_outside = 0
            circularity_defined = False
            nc_defined = False

        if technical_valid:
            eligible_samples.append(sample)
        else:
            malformed_samples.append(sample.sample_id)
            for reason in reasons:
                exclusion_counter[reason] += 1

        rows.append(
            {
                "sample_id": sample.sample_id,
                "parent_source_id": sample.parent_source_id,
                "image_path": str(sample.image_path.relative_to(ROOT)),
                "nucleus_contour_path": str(sample.nucleus_contour_path.relative_to(ROOT)),
                "cytoplasm_contour_path": str(sample.cytoplasm_contour_path.relative_to(ROOT)),
                "technical_valid": str(technical_valid).upper(),
                "technical_exclusion_reason": ";".join(reasons),
                "width": width,
                "height": height,
                "diagnostic_class_if_available": sample.diagnostic_class,
                "image_mode": mode,
                "dtype": dtype,
                "nucleus_points": len(nucleus_points) if nucleus_points is not None else 0,
                "cytoplasm_points": len(cytoplasm_points) if cytoplasm_points is not None else 0,
                "nucleus_area_gt_pixels": nucleus_area,
                "cell_area_gt_pixels": cell_area,
                "cytoplasm_only_area_gt_pixels": cytoplasm_only_area,
                "nucleus_perimeter_gt_pixels": perimeter,
                "nucleus_outside_cell_pixels": nucleus_outside,
                "nc_ratio_defined": str(nc_defined).upper(),
                "circularity_defined": str(circularity_defined).upper(),
            }
        )

    manifest_path = OUT_DIR / "sipakmed_eligibility_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    duplicate_image_hash_groups = [ids for ids in image_hashes.values() if len(ids) > 1]
    duplicate_contour_hash_groups = [ids for ids in contour_hashes.values() if len(ids) > 1]

    qc_candidates = []
    valid_sorted = sorted(eligible_samples, key=lambda sample: sample.sample_id)
    if valid_sorted:
        areas = [(sample.native_width * sample.native_height, sample) for sample in valid_sorted]
        areas_sorted = sorted(areas, key=lambda item: (item[0], item[1].sample_id))
        selected = {
            "first_valid": valid_sorted[0],
            "smallest": areas_sorted[0][1],
            "median_size": areas_sorted[len(areas_sorted) // 2][1],
            "largest": areas_sorted[-1][1],
        }
        for class_name in sorted(set(sample.diagnostic_class for sample in valid_sorted)):
            selected[f"class_{class_name}"] = next(
                sample for sample in valid_sorted if sample.diagnostic_class == class_name
            )
        seen = set()
        for label, sample in selected.items():
            if sample.sample_id not in seen:
                seen.add(sample.sample_id)
                qc_candidates.append({"label": f"{label}: {sample.sample_id}", "sample": sample})
    if qc_candidates:
        overlay_panel(qc_candidates, OUT_DIR / "gt_alignment_qc.png")

    parent_counts = Counter(sample.parent_source_id for sample in samples)
    eligible_count = len(eligible_samples)
    excluded_count = len(samples) - eligible_count
    preflight_go = (
        all(item["size_matches_expected"] for item in archive_status)
        and len(samples) == 4049
        and not duplicate_sample_ids
        and sum(item["cropped_bmp"] for item in class_summary) == sum(item["cropped_nucleus_dat"] for item in class_summary)
        and sum(item["cropped_bmp"] for item in class_summary) == sum(item["cropped_cytoplasm_dat"] for item in class_summary)
        and eligible_count > 0
        and coordinate_xy_in_bounds > coordinate_yx_in_bounds
        and len(parent_counts) == sum(item["root_bmp"] for item in class_summary)
    )
    summary = {
        "preflight_type": "outcome_blind_technical_preflight",
        "date": "2026-08-13",
        "official_full_archives_available": all(item["size_matches_expected"] for item in archive_status),
        "archive_integrity": archive_status,
        "class_summary": class_summary,
        "total_isolated_cell_images": len(samples),
        "total_cluster_images": sum(item["root_bmp"] for item in class_summary),
        "total_cropped_annotation_files": sum(item["cropped_dat"] for item in class_summary),
        "total_cropped_nucleus_contours": sum(item["cropped_nucleus_dat"] for item in class_summary),
        "total_cropped_cytoplasm_contours": sum(item["cropped_cytoplasm_dat"] for item in class_summary),
        "technical_eligible_count": eligible_count,
        "technical_excluded_count": excluded_count,
        "technical_exclusion_reasons": dict(sorted(exclusion_counter.items())),
        "duplicate_sample_ids": duplicate_sample_ids,
        "duplicate_image_hash_group_count": len(duplicate_image_hash_groups),
        "duplicate_contour_hash_group_count": len(duplicate_contour_hash_groups),
        "image_dimensions": {
            f"{width}x{height}": count for (width, height), count in sorted(dimension_counter.items())
        },
        "image_modes": dict(sorted(mode_counter.items())),
        "image_dtypes": dict(sorted(dtype_counter.items())),
        "contour_format": "plain text .dat, one x,y floating-point coordinate pair per line, comma separated",
        "coordinate_convention": "native image coordinate space with origin at top-left; x is horizontal column coordinate and y is vertical row coordinate",
        "indexing_convention": "continuous pixel coordinates bounded by 0 <= x < width and 0 <= y < height; not integer label indices",
        "coordinate_xy_in_bounds": coordinate_xy_in_bounds,
        "coordinate_yx_in_bounds_if_swapped": coordinate_yx_in_bounds,
        "total_contours_checked": total_contours,
        "contour_point_count_min": min(contour_point_counts) if contour_point_counts else None,
        "contour_point_count_max": max(contour_point_counts) if contour_point_counts else None,
        "contour_orientations": dict(contour_orientations),
        "cytoplasm_semantics": "whole-cell outer boundary including nucleus; canonical cytoplasm-only mask is cell_mask AND NOT nucleus_mask",
        "cyt_contours_containing_nucleus_count": cyt_contains_nucleus,
        "nucleus_outside_cell_count": nucleus_outside_cell_count,
        "max_nucleus_outside_pixels": max_nucleus_outside_pixels,
        "nc_computable_for_technically_eligible_samples": True,
        "nc_computable_for_all_raw_samples": excluded_count == 0,
        "parent_source_grouping_recoverable": True,
        "unique_parent_source_ids": len(parent_counts),
        "parent_source_cells_min": min(parent_counts.values()) if parent_counts else 0,
        "parent_source_cells_max": max(parent_counts.values()) if parent_counts else 0,
        "preferred_future_resampling_unit": "parent_source_id",
        "checkpoint_loaded": False,
        "inference_run": False,
        "prediction_metrics_seen": False,
        "verdict": "GO" if preflight_go else "CONDITIONAL_GO",
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "gt_alignment_qc_path": str((OUT_DIR / "gt_alignment_qc.png").relative_to(ROOT)),
    }
    (OUT_DIR / "preflight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
