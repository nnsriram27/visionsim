from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def cache_key(blend_path: Path, solver_cfg: dict) -> str:
    """Stable cache key from the blend identity and solver-relevant config.

    Args:
        blend_path: Path to the source blend file.
        solver_cfg: Solver-relevant config values that affect the result.

    Returns:
        A short hex digest used as the cache subdirectory name.
    """
    blend_path = Path(blend_path)
    try:
        mtime = blend_path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    payload = json.dumps({"p": str(blend_path), "m": mtime, "c": solver_cfg}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def write_temperatures(cache_root: Path, key: str, per_object: dict[str, np.ndarray], meta: dict) -> Path:
    """Write per-object temperature histories to ``<cache_root>/<key>/temperatures.npz``.

    Args:
        cache_root: Root directory for thermal caches.
        key: Cache key from :func:`cache_key`.
        per_object: Mapping of object name to a ``(timesteps, vertices)`` array.
        meta: JSON-serializable metadata stored alongside the arrays.

    Returns:
        The path to the written ``.npz`` archive.
    """
    out_dir = Path(cache_root) / key
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "temperatures.npz"
    save_data: dict[str, Any] = {"__meta__": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)}
    save_data.update(per_object)
    np.savez_compressed(out, **save_data)
    return out


def read_temperatures(cache_root: Path, key: str) -> dict[str, np.ndarray] | None:
    """Read per-object temperature histories, or return ``None`` on a cache miss.

    Args:
        cache_root: Root directory for thermal caches.
        key: Cache key from :func:`cache_key`.

    Returns:
        Mapping of object name to its history array, or ``None`` if absent.
    """
    path = Path(cache_root) / key / "temperatures.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in data.files if k != "__meta__"}
