"""Near-duplicate matching between two image caches (CPU only).

Used to find the Capsule Vision 2024 frames that came from Kvasir-Capsule. Two stages:

1. Perceptual hash (64-bit DCT pHash) of every image; candidate pairs are all pairs with
   Hamming distance <= ``max_hamming``. The comparison is brute force in blocks: 54k x 47k
   XOR+popcount takes seconds and, unlike bucketed schemes, misses nothing.
2. Verification by normalised cross-correlation (NCC) of 64x64 grey thumbnails. A query
   is a duplicate when its best candidate reaches ``min_ncc``.

pHash is computed from the cached 224x224 images, so both datasets pass through the same
resize before hashing regardless of their original resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

from capsule_xai_audit.utils.logging import get_logger

log = get_logger(__name__)

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _grey(img: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(img).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.float64)


def phash(img: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.uint64:
    """DCT perceptual hash, equivalent to ``imagehash.phash`` with default settings."""
    pixels = _grey(img, hash_size * highfreq_factor)
    low = dctn(pixels, type=2, norm="ortho")[:hash_size, :hash_size]
    bits = (low > np.median(low)).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def phash_all(pixels: np.ndarray, positions: np.ndarray | None = None) -> np.ndarray:
    rows = np.arange(len(pixels)) if positions is None else positions
    return np.fromiter((phash(pixels[i]) for i in rows), dtype=np.uint64, count=len(rows))


def popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):  # NumPy >= 2.0
        return np.bitwise_count(x).astype(np.uint16)
    return _POPCOUNT8[x.view(np.uint8).reshape(*x.shape, 8)].sum(axis=-1, dtype=np.uint16)


def hamming_candidates(
    query: np.ndarray, reference: np.ndarray, max_hamming: int, block: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All (query_idx, ref_idx, distance) with Hamming distance <= ``max_hamming``."""
    qi, ri, dist = [], [], []
    for start in range(0, len(query), block):
        q = query[start : start + block]
        d = popcount64(np.bitwise_xor(q[:, None], reference[None, :]))
        rows, cols = np.nonzero(d <= max_hamming)
        qi.append(rows + start)
        ri.append(cols)
        dist.append(d[rows, cols])
    if not qi:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.uint16)
    return np.concatenate(qi), np.concatenate(ri), np.concatenate(dist)


def ncc_thumbnail(img: np.ndarray, size: int = 64) -> np.ndarray:
    """Zero-mean, unit-norm grey thumbnail; the dot product of two is their NCC."""
    g = _grey(img, size).ravel()
    g -= g.mean()
    norm = np.linalg.norm(g)
    return (g / norm if norm > 0 else g).astype(np.float32)


@dataclass(frozen=True)
class DedupResult:
    """One row per query image."""

    table: pd.DataFrame  # query_pos, best_ref_pos, hamming, ncc, is_duplicate

    @property
    def duplicate_positions(self) -> np.ndarray:
        return self.table.loc[self.table["is_duplicate"], "query_pos"].to_numpy()


def match_duplicates(
    query_pixels: np.ndarray,
    reference_pixels: np.ndarray,
    query_hashes: np.ndarray | None = None,
    reference_hashes: np.ndarray | None = None,
    max_hamming: int = 10,
    min_ncc: float = 0.95,
    thumb_size: int = 64,
) -> DedupResult:
    qh = phash_all(query_pixels) if query_hashes is None else query_hashes
    rh = phash_all(reference_pixels) if reference_hashes is None else reference_hashes
    q_idx, r_idx, dist = hamming_candidates(qh, rh, max_hamming)
    log.info("dedup: %d candidate pairs for %d query images", len(q_idx), len(qh))

    thumbs: dict[tuple[str, int], np.ndarray] = {}

    def thumb(kind: str, pixels: np.ndarray, i: int) -> np.ndarray:
        key = (kind, i)
        if key not in thumbs:
            thumbs[key] = ncc_thumbnail(pixels[i], thumb_size)
        return thumbs[key]

    ncc = np.array(
        [
            float(thumb("q", query_pixels, q) @ thumb("r", reference_pixels, r))
            for q, r in zip(q_idx, r_idx, strict=True)
        ],
        dtype=np.float64,
    )
    table = pd.DataFrame(
        {
            "query_pos": np.arange(len(qh)),
            "best_ref_pos": -1,
            "hamming": -1,
            "ncc": np.nan,
            "is_duplicate": False,
        }
    )
    if len(q_idx):
        pairs = pd.DataFrame({"q": q_idx, "r": r_idx, "d": dist.astype(int), "ncc": ncc})
        best = pairs.sort_values(["q", "ncc"], ascending=[True, False]).drop_duplicates("q")
        table.loc[best["q"].to_numpy(), "best_ref_pos"] = best["r"].to_numpy()
        table.loc[best["q"].to_numpy(), "hamming"] = best["d"].to_numpy()
        table.loc[best["q"].to_numpy(), "ncc"] = best["ncc"].to_numpy()
        table["is_duplicate"] = table["ncc"].fillna(-1.0) >= min_ncc
    log.info(
        "dedup: %d of %d query images matched (NCC >= %.2f)", table["is_duplicate"].sum(), len(qh), min_ncc
    )
    return DedupResult(table=table)
