import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_prep.load_sipakmed import (
    contour_in_bounds,
    list_sipakmed_samples,
    load_sipakmed_sample,
    parse_contour_file,
    rasterize_contour,
)


def test_parse_contour_file_reads_xy_float_pairs(tmp_path: Path) -> None:
    path = tmp_path / "001_01_nuc.dat"
    path.write_text("1.25,2.75\n3,4\n5.5,6.5\n", encoding="utf-8")
    assert parse_contour_file(path) == [(1.25, 2.75), (3.0, 4.0), (5.5, 6.5)]


def test_rasterize_contour_uses_xy_order() -> None:
    mask = rasterize_contour([(1, 2), (3, 2), (3, 4), (1, 4)], width=6, height=7)
    assert mask.shape == (7, 6)
    assert mask[2:5, 1:4].any()
    assert not mask[:2, :].any()


def test_contour_bounds_are_native_xy() -> None:
    assert contour_in_bounds([(0.0, 0.0), (4.5, 2.5), (1.0, 3.0)], width=5, height=4)
    assert not contour_in_bounds([(0.0, 0.0), (5.0, 2.5), (1.0, 3.0)], width=5, height=4)


def test_list_and_load_sipakmed_sample_constructs_cytoplasm_only(tmp_path: Path) -> None:
    cropped = tmp_path / "im_Parabasal" / "CROPPED"
    cropped.mkdir(parents=True)
    image_path = cropped / "001_02.bmp"
    Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8)).save(image_path)
    (cropped / "001_02_cyt.dat").write_text("1,1\n10,1\n10,8\n1,8\n", encoding="utf-8")
    (cropped / "001_02_nuc.dat").write_text("4,3\n6,3\n6,5\n4,5\n", encoding="utf-8")

    samples = list_sipakmed_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].sample_id == "Parabasal__001_02"
    assert samples[0].parent_source_id == "Parabasal__001"

    loaded = load_sipakmed_sample(samples[0])
    assert loaded.image.shape == (10, 12, 3)
    assert loaded.nucleus_mask.any()
    assert loaded.cell_mask.any()
    assert not np.logical_and(loaded.nucleus_mask, loaded.cytoplasm_only_mask).any()
    assert np.logical_and(loaded.nucleus_mask, loaded.cell_mask).sum() == loaded.nucleus_mask.sum()
