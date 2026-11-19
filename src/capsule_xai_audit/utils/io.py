"""File I/O helpers: atomic writes, hashing, JSON with numpy support."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so that readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, default=_json_default, allow_nan=True)
    atomic_write_bytes(path, (text + "\n").encode())


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(obj, sort_keys=True, default=_json_default) + "\n")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    atomic_write_bytes(path, df.to_csv(index=False).encode())


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: Iterable[str]) -> str:
    """Order-sensitive hash of a sequence of strings (e.g. the image ids of an index)."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest()
