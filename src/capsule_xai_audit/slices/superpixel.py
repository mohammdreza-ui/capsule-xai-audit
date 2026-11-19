"""Slice A: patch vs superpixel explanation units on one frozen DINOv2 linear probe.

The model is ``logits = W · normalize(mean_p f_p - mu) + b`` over the final-normed patch
tokens ``f_p``, with ``mu`` the mean training feature (CL2N probe). Because the head is
linear in the mean token, the class margin ``z_c - mean_k z_k`` splits exactly into
per-patch terms relative to ``mu``

    c_p = (w_c - mean_k w_k) · (f_p - mu) / (N · ||mean_p f_p - mu||),
    sum_p c_p = margin - (b_c - mean_k b_k).

A superpixel's score is the area-weighted mean of the patch terms it overlaps, which is
the same as applying the head to the superpixel's area-weighted mean feature. Both units
explain the same model, so any difference in localisation comes from the unit alone.
"""

from __future__ import annotations

import numpy as np
from skimage.segmentation import slic


def slic_segments(image: np.ndarray, n_segments: int, compactness: float) -> np.ndarray:
    """SLIC label map (H, W) with contiguous labels starting at 0."""
    labels = slic(image, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1)
    _, contiguous = np.unique(labels, return_inverse=True)
    return contiguous.reshape(labels.shape).astype(np.int32)


def patch_unit_map(grid: tuple[int, int], height: int, width: int) -> np.ndarray:
    """(H, W) map of the patch index each pixel belongs to (nearest-neighbour upsampling)."""
    gh, gw = grid
    rows = np.minimum((np.arange(height) * gh) // height, gh - 1)
    cols = np.minimum((np.arange(width) * gw) // width, gw - 1)
    return (rows[:, None] * gw + cols[None, :]).astype(np.int32)


def overlap_matrix(segments: np.ndarray, units: np.ndarray, n_segments: int, n_units: int) -> np.ndarray:
    """(S, N) pixel counts of segment ``s`` overlapping unit ``n``."""
    flat = segments.ravel().astype(np.int64) * n_units + units.ravel()
    return np.bincount(flat, minlength=n_segments * n_units).reshape(n_segments, n_units).astype(np.float64)


def area_weighted(overlap: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean of per-unit ``values`` (N,) or (N, D) over each segment, weighted by overlap area."""
    areas = overlap.sum(1)
    pooled = overlap @ values
    return np.asarray(pooled / (areas[:, None] if pooled.ndim == 2 else areas))


def patch_contributions(
    patch_features: np.ndarray, weight: np.ndarray, target: int, center: np.ndarray
) -> np.ndarray:
    """Exact per-patch share ``c_p`` of the class margin (see module docstring)."""
    n = len(patch_features)
    shifted = patch_features - center
    norm = max(float(np.linalg.norm(shifted.mean(0))), 1e-12)
    direction = weight[target] - weight.mean(0)
    return np.asarray(shifted @ direction / (n * norm))


def region_adjacency(segments: np.ndarray) -> np.ndarray:
    """Undirected edges (E, 2), ``a < b``, between 4-connected neighbouring segments."""
    pairs = np.concatenate(
        [
            np.stack([segments[:, :-1].ravel(), segments[:, 1:].ravel()], 1),
            np.stack([segments[:-1, :].ravel(), segments[1:, :].ravel()], 1),
        ]
    )
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(np.sort(pairs, axis=1), axis=0).astype(np.int32)


def segment_centroids(segments: np.ndarray, n_segments: int) -> np.ndarray:
    ys, xs = np.indices(segments.shape)
    count = np.bincount(segments.ravel(), minlength=n_segments)
    cx = np.bincount(segments.ravel(), weights=xs.ravel(), minlength=n_segments) / count
    cy = np.bincount(segments.ravel(), weights=ys.ravel(), minlength=n_segments) / count
    return np.stack([cx, cy], 1)
