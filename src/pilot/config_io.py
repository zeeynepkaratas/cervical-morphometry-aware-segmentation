"""Configuration and path helpers for pilot commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from src.utils.config import FORBIDDEN_DATASET_TOKENS, ROOT_DIR


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) if yaml is not None else _parse_simple_yaml(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return data


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple two-level YAML shape used by configs/pilot.yaml."""
    import ast

    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not raw_line.startswith(" "):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                root[key] = _parse_scalar(value)
                current = None
            else:
                current = {}
                root[key] = current
        else:
            if current is None:
                raise ValueError(f"Unsupported YAML indentation: {raw_line}")
            key, value = line.strip().split(":", 1)
            current[key.strip()] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    """Parse a scalar/list from the pilot YAML fallback."""
    import ast

    if value in {"null", "None", ""}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("["):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [item.strip().strip("'\"") for item in value.strip("[]").split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def repo_commit_sha() -> str:
    """Return the current git commit SHA, or UNKNOWN outside git."""
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True).strip()
    except Exception:
        return "UNKNOWN"


def resolve_herlev_data_dir(config: dict[str, Any], data_dir_arg: str | None = None) -> Path:
    """Resolve Herlev data from CLI, env, config, or a read-only source checkout."""
    candidates = []
    if data_dir_arg:
        candidates.append(Path(data_dir_arg))
    if os.environ.get("HERLEV_DATA_DIR"):
        candidates.append(Path(os.environ["HERLEV_DATA_DIR"]))
    configured = config.get("data", {}).get("herlev_data_dir")
    if configured:
        candidates.append(Path(configured))
    candidates.append(ROOT_DIR.parent / "cervical-morphometry-conformal" / "data" / "raw" / "herlev")
    candidates.append(ROOT_DIR / "data" / "raw" / "herlev")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Herlev data directory not found. Set HERLEV_DATA_DIR or pass --data-dir. "
        f"Checked: {[str(path) for path in candidates]}"
    )


def guard_forbidden_paths(*paths: Path) -> None:
    """Reject paths that indicate Cx22 or conformal resources in pilot commands."""
    for path in paths:
        lowered = str(path).lower()
        matched = sorted(token for token in FORBIDDEN_DATASET_TOKENS if token in lowered)
        if matched:
            raise ValueError(f"Forbidden pilot path token(s) {matched} in path: {path}")
