"""Portable paths for the frozen paper experiment harness."""
import os
from pathlib import Path


def resolve_data_dir(value=None):
    selected = value or os.environ.get("CTR_DATA")
    if not selected:
        raise ValueError("Pass --data-dir or set CTR_DATA to a prepared dataset directory")
    path = Path(selected).expanduser().resolve()
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"Prepared metadata not found: {path / 'meta.json'}")
    return path
