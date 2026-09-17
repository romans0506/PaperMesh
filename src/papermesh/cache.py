"""Tiny JSON file cache with a time-to-live, shared by the API clients."""

import json
import time
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_TTL_SECONDS = 24 * 60 * 60


def read_json(path: Path, ttl_seconds: int | None = DEFAULT_TTL_SECONDS) -> Any | None:
    """Return the cached value, or None if missing, expired, or unreadable. ttl_seconds=None never expires."""
    if not path.exists():
        return None
    if ttl_seconds is not None and time.time() - path.stat().st_mtime > ttl_seconds:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
