"""Pilot train/validation split generation from the original train cells only."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.data_prep.load_herlev import list_herlev_images
from src.utils.config import FORBIDDEN_SPLITS, PILOT_SPLIT_SEED


def _source_train_ids(split: dict) -> list[str]:
    for forbidden in FORBIDDEN_SPLITS:
        if forbidden in split and not isinstance(split[forbidden], list):
            raise ValueError(f"Unexpected split metadata for {forbidden}.")
    train_ids = split.get("train") or split.get("train_full")
    if train_ids is None:
        raise KeyError("Original split must contain 'train' or 'train_full'.")
    return sorted(set(train_ids))


def _class_by_cell_id(data_dir: Path) -> dict[str, str]:
    return {path.stem: path.parent.name for path in list_herlev_images(data_dir)}


def create_pilot_train_validation_split(
    original_split_path: Path,
    data_dir: Path,
    output_path: Path,
    seed: int = PILOT_SPLIT_SEED,
    validation_fraction: float = 0.20,
) -> dict:
    """Create a deterministic approximately stratified split from train cells only."""
    original = json.loads(Path(original_split_path).read_text(encoding="utf-8"))
    train_full = _source_train_ids(original)
    class_by_id = _class_by_cell_id(data_dir)
    missing = sorted(cell_id for cell_id in train_full if cell_id not in class_by_id)
    if missing:
        raise ValueError(f"Split IDs missing from Herlev data directory: {missing[:5]} ...")

    rng = random.Random(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for cell_id in train_full:
        by_class[class_by_id[cell_id]].append(cell_id)

    pilot_train: list[str] = []
    pilot_validation: list[str] = []
    for cls, ids in sorted(by_class.items()):
        shuffled = sorted(ids)
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * validation_fraction)) if len(shuffled) > 1 else 0
        pilot_validation.extend(shuffled[:n_val])
        pilot_train.extend(shuffled[n_val:])

    pilot_train = sorted(pilot_train)
    pilot_validation = sorted(pilot_validation)
    if set(pilot_train) & set(pilot_validation):
        raise AssertionError("Pilot train/validation leakage detected.")
    if set(pilot_train) | set(pilot_validation) != set(train_full):
        raise AssertionError("Pilot split does not exactly cover original train cells.")

    result = {
        "seed": seed,
        "source_split": str(original_split_path),
        "source_pool": "train_full" if "train_full" in original else "train",
        "forbidden_source_splits_not_used": sorted(FORBIDDEN_SPLITS & set(original.keys())),
        "validation_fraction": validation_fraction,
        "n_train": len(pilot_train),
        "n_validation": len(pilot_validation),
        "class_distribution": {
            "train": dict(sorted(Counter(class_by_id[cell_id] for cell_id in pilot_train).items())),
            "validation": dict(sorted(Counter(class_by_id[cell_id] for cell_id in pilot_validation).items())),
        },
        "train_ids": pilot_train,
        "validation_ids": pilot_validation,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
