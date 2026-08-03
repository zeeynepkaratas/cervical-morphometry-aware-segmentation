"""Generate the locked pilot train/validation split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pilot.config_io import load_config, resolve_herlev_data_dir
from src.pilot.splits import create_pilot_train_validation_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    config = load_config(ROOT_DIR / args.config)
    data_dir = resolve_herlev_data_dir(config, args.data_dir)
    split_cfg = config["splits"]
    result = create_pilot_train_validation_split(
        original_split_path=ROOT_DIR / split_cfg["original_split"],
        data_dir=data_dir,
        output_path=ROOT_DIR / split_cfg["pilot_split"],
        seed=int(config["seed"]),
        validation_fraction=float(split_cfg["validation_fraction"]),
    )
    print(f"Wrote pilot split with train={result['n_train']} validation={result['n_validation']}")


if __name__ == "__main__":
    main()
