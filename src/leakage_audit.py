"""Data leakage audit for the circularity mechanism analysis.

Verifies:
1. validation–test overlap: none
2. calibration–test overlap: none
3. training–test overlap: none
4. All split IDs are unique within each split
5. Post-processing parameter selection uses only validation (audit trail)
6. Test metric does not appear in selection logic

Written to: results/circularity_mechanism/data_leakage_audit.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data" / "splits" / "original_herlev_group_split.json"
OUT_DIR = ROOT / "results" / "circularity_mechanism"


def run(out_dir: Path = OUT_DIR) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))

    val_ids   = set(splits.get("unet_val", []))
    cal_ids   = set(splits.get("calibration", []))
    test_ids  = set(splits.get("test", []))
    train_ids = set(splits.get("unet_train", []))

    checks = []

    def _check(name: str, result: bool, detail: str) -> None:
        status = "PASS" if result else "FAIL"
        checks.append({"check": name, "status": status, "detail": detail})
        print(f"  [{status}] {name}: {detail}")

    print("Running data leakage audit ...")

    _check(
        "val_test_no_overlap",
        len(val_ids & test_ids) == 0,
        f"overlap={sorted(val_ids & test_ids)[:5]} (showing up to 5)",
    )
    _check(
        "cal_test_no_overlap",
        len(cal_ids & test_ids) == 0,
        f"overlap={sorted(cal_ids & test_ids)[:5]}",
    )
    _check(
        "train_test_no_overlap",
        len(train_ids & test_ids) == 0,
        f"overlap={sorted(train_ids & test_ids)[:5]}",
    )
    _check(
        "val_cal_no_overlap",
        len(val_ids & cal_ids) == 0,
        f"overlap={sorted(val_ids & cal_ids)[:5]}",
    )

    # Uniqueness within each split
    for split_name, cell_list in [
        ("unet_val", splits.get("unet_val", [])),
        ("calibration", splits.get("calibration", [])),
        ("test", splits.get("test", [])),
        ("unet_train", splits.get("unet_train", [])),
    ]:
        _check(
            f"{split_name}_unique_ids",
            len(cell_list) == len(set(cell_list)),
            f"n_total={len(cell_list)}, n_unique={len(set(cell_list))}",
        )

    # Split sizes
    _check(
        "val_nonempty",
        len(val_ids) > 0,
        f"n_val={len(val_ids)}",
    )
    _check(
        "test_nonempty",
        len(test_ids) > 0,
        f"n_test={len(test_ids)}",
    )
    _check(
        "cal_nonempty",
        len(cal_ids) > 0,
        f"n_cal={len(cal_ids)}",
    )

    # Verify postprocessing_selection.json (if it exists) was based on val
    sel_path = out_dir / "postprocessing_selection.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        _check(
            "postprocessing_selection_uses_val_only",
            sel.get("selection_set") == "unet_val",
            f"selection_set={sel.get('selection_set')}",
        )
        _check(
            "postprocessing_test_not_seen_during_selection",
            sel.get("test_not_seen") is True,
            f"test_not_seen={sel.get('test_not_seen')}",
        )
    else:
        checks.append({
            "check": "postprocessing_selection_file",
            "status": "PENDING",
            "detail": "postprocessing_selection.json not yet created",
        })

    df = pd.DataFrame(checks)
    out_path = out_dir / "data_leakage_audit.csv"
    df.to_csv(out_path, index=False)

    n_fail = (df["status"] == "FAIL").sum()
    print(f"\nAudit complete: {len(df)} checks, {n_fail} failures.")
    if n_fail > 0:
        print("FAILED CHECKS:")
        print(df[df["status"] == "FAIL"].to_string())
    print(f"Written: {out_path}")
    return df


if __name__ == "__main__":
    run()
