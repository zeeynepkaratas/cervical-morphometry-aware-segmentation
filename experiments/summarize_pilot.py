"""Summarize pilot metrics and make the locked decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pilot.config_io import load_config
from src.pilot.summary import summarize_per_cell


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    config = load_config(ROOT_DIR / args.config)
    out_dir = ROOT_DIR / config["outputs"]["pilot_dir"]
    summary = summarize_per_cell(out_dir / "per_cell_predictions.csv", out_dir)
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
