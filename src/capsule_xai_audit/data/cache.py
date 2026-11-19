"""Resized uint8 image cache backed by a NumPy memmap.

Decoding 47k PNGs every epoch is the bottleneck on Kaggle, so every image is decoded once,
centre-cropped to a square, resized to ``size x size`` and stored in one ``(N, H, W, 3)``
uint8 file (~7 GB for Kvasir-Capsule at 224). Row ``i`` of the cache is row ``i`` of the
index it was built from; the index hash in the sidecar JSON guards against mismatches.

Building is resumable: progress is committed to the sidecar after every chunk.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from capsule_xai_audit.utils.io import read_json, sha256_strings, write_json
from capsule_xai_audit.utils.logging import get_logger

log = get_logger(__name__)


class CacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class CacheMeta:
    n: int
    size: int
    index_sha256: str
    filled: int

    @property
    def complete(self) -> bool:
        return self.filled == self.n


def cache_paths(prefix: Path) -> tuple[Path, Path, Path]:
    """(pixels, original sizes, sidecar json) for a cache prefix such as ``cache/kvasir_224``."""
    prefix = Path(prefix)
    return prefix.with_suffix(".u8"), prefix.with_suffix(".sizes.npy"), prefix.with_suffix(".json")


def square_crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return left, top, left + side, top + side


def load_resized(path: str | Path, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Decode, centre-crop to square and resize (antialiased bicubic)."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        square = rgb.crop(square_crop_box(width, height))
        resized = square.resize((size, size), Image.Resampling.BICUBIC, reducing_gap=3.0)
        return np.asarray(resized, dtype=np.uint8), (width, height)


def transform_boxes(
    boxes: Sequence[Sequence[float]], width: int, height: int, size: int
) -> list[tuple[float, float, float, float]]:
    """Map ``[x0, y0, x1, y1]`` boxes from original pixels into cache pixels."""
    left, top, right, _ = square_crop_box(width, height)
    scale = size / (right - left)
    out = []
    for x0, y0, x1, y1 in boxes:
        nx0, ny0 = (x0 - left) * scale, (y0 - top) * scale
        nx1, ny1 = (x1 - left) * scale, (y1 - top) * scale
        nx0, ny0 = max(0.0, nx0), max(0.0, ny0)
        nx1, ny1 = min(float(size), nx1), min(float(size), ny1)
        if nx1 > nx0 and ny1 > ny0:
            out.append((nx0, ny0, nx1, ny1))
    return out


def _load_job(args: tuple[str, int]) -> tuple[np.ndarray, tuple[int, int]]:
    return load_resized(*args)


def build_cache(
    image_paths: Sequence[str],
    image_ids: Sequence[str],
    prefix: Path,
    size: int = 224,
    num_workers: int = 4,
    chunk: int = 2048,
) -> CacheMeta:
    pixels_path, sizes_path, meta_path = cache_paths(prefix)
    pixels_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(image_paths)
    index_sha = sha256_strings(image_ids)

    filled = 0
    if meta_path.exists():
        meta = CacheMeta(**read_json(meta_path))
        if meta.index_sha256 == index_sha and meta.size == size and meta.n == n:
            filled = meta.filled
            if meta.complete:
                log.info("cache %s is complete", pixels_path)
                return meta
            log.info("resuming cache %s at %d/%d", pixels_path, filled, n)
        else:
            log.warning("cache %s was built from a different index; rebuilding", pixels_path)
    mode = "r+" if filled and pixels_path.exists() else "w+"
    pixels = np.lib.format.open_memmap(pixels_path, mode=mode, dtype=np.uint8, shape=(n, size, size, 3))
    sizes = np.load(sizes_path) if filled and sizes_path.exists() else np.zeros((n, 2), dtype=np.int32)

    # "spawn": forking a process that already runs torch threads can deadlock.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, num_workers), mp_context=context) as pool:
        for start in range(filled, n, chunk):
            stop = min(n, start + chunk)
            jobs = [(image_paths[i], size) for i in range(start, stop)]
            for offset, (arr, wh) in enumerate(pool.map(_load_job, jobs, chunksize=32)):
                pixels[start + offset] = arr
                sizes[start + offset] = wh
            pixels.flush()
            np.save(sizes_path, sizes)
            write_json(meta_path, asdict(CacheMeta(n=n, size=size, index_sha256=index_sha, filled=stop)))
            log.info("cache %s: %d/%d", pixels_path.name, stop, n)
    return CacheMeta(n=n, size=size, index_sha256=index_sha, filled=n)


def open_cache(prefix: Path, image_ids: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read-only memmap of pixels and the (N, 2) original (width, height) array."""
    pixels_path, sizes_path, meta_path = cache_paths(prefix)
    if not meta_path.exists():
        raise CacheError(f"no cache at {prefix}; run `cxa cache` first")
    meta = CacheMeta(**read_json(meta_path))
    if not meta.complete:
        raise CacheError(f"cache {prefix} is incomplete ({meta.filled}/{meta.n}); rerun `cxa cache`")
    if image_ids is not None and sha256_strings(image_ids) != meta.index_sha256:
        raise CacheError(f"cache {prefix} was built from a different index")
    pixels = np.load(pixels_path, mmap_mode="r")
    return pixels, np.load(sizes_path)
