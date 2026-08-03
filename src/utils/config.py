"""Shared constants for the Herlev-only N/C-aware segmentation pilot."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATA_SPLITS = DATA_DIR / "splits"
DATA_RAW_HERLEV = ROOT_DIR / "data" / "raw" / "herlev"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_PILOT = RESULTS_DIR / "pilot"
RESULTS_TABLES = RESULTS_PILOT
RESULTS_FIGURES = RESULTS_PILOT / "figures"

MEASUREMENTS = ["nc_ratio", "circularity"]
MEASUREMENT_DOMAINS = {
    "nc_ratio": (0.0, None),
    "circularity": (0.0, 1.0),
}

DEGRADATIONS = ["gaussian_blur", "gaussian_noise"]
DEGRADATION_SEVERITY_LEVELS = {
    "gaussian_blur": [1.0, 2.0, 3.0],
    "gaussian_noise": [10.0, 20.0, 30.0],
}

RE_EPSILON = 1e-6
RANDOM_SEED = 42
PILOT_SPLIT_SEED = 20260803
SPLIT_RATIOS = {"train": 0.80, "validation": 0.20}

FORBIDDEN_SPLITS = {"calibration", "test"}
FORBIDDEN_DATASET_TOKENS = {"cx22", "cqr", "mondrian", "conformal"}
