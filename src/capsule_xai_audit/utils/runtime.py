"""Device selection, mixed precision and run provenance."""

from __future__ import annotations

import contextlib
import platform
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

_TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "timm",
    "captum",
    "numpy",
    "scikit-learn",
    "scikit-image",
    "pandas",
)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@contextlib.contextmanager
def autocast(device: torch.device, enabled: bool) -> Iterator[None]:
    """fp16 autocast on CUDA (T4 has no bf16 tensor cores); a no-op elsewhere."""
    if enabled and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def git_revision(cwd: Path | None = None) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    if sha is None:
        # A code archive without .git (the Kaggle upload) carries its revision in REVISION.
        for parent in (cwd or Path.cwd()).resolve().parents:
            stamp = parent / "REVISION"
            if stamp.is_file():
                return {"sha": None, "dirty": None, "revision_file": stamp.read_text().strip()}
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def provenance() -> dict[str, Any]:
    """Everything needed to tell two runs apart later. Deliberately excludes host names."""
    versions: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(terse=True),
        "packages": versions,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "git": git_revision(Path(__file__).resolve().parent),
    }
