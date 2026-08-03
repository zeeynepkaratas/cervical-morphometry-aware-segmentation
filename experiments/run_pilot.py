"""One-command pilot runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> None:
    completed = subprocess.run([sys.executable, *args], cwd=ROOT_DIR)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    shared = ["--config", args.config]
    if args.data_dir:
        shared += ["--data-dir", args.data_dir]
    _run(["experiments/make_pilot_split.py", *shared])
    _run(["experiments/train_pilot_models.py", *shared])
    _run(["experiments/evaluate_pilot_models.py", *shared])
    _run(["experiments/summarize_pilot.py", "--config", args.config])


if __name__ == "__main__":
    main()
