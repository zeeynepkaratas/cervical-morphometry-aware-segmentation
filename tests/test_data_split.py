"""Tests for pilot split guardrails."""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def test_original_split_metadata_is_present() -> None:
    split_path = ROOT_DIR / "data" / "splits" / "original_herlev_group_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assert "calibration" in split
    assert "test" in split
    assert "train_full" in split or "train" in split


def test_pilot_split_does_not_overlap_forbidden_splits_if_generated() -> None:
    split_path = ROOT_DIR / "data" / "splits" / "pilot_train_validation_split.json"
    if not split_path.exists():
        return
    original = json.loads((ROOT_DIR / "data" / "splits" / "original_herlev_group_split.json").read_text())
    pilot = json.loads(split_path.read_text())
    pilot_ids = set(pilot["train_ids"]) | set(pilot["validation_ids"])
    assert not (pilot_ids & set(original["calibration"]))
    assert not (pilot_ids & set(original["test"]))
    assert not (set(pilot["train_ids"]) & set(pilot["validation_ids"]))
