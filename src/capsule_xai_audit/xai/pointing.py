"""Localisation metrics against bounding boxes.

Boxes are ``(x0, y0, x1, y1)`` in the attribution map's pixel coordinates; pixel ``(x, y)``
covers ``[x, x + 1) x [y, y + 1)`` and is located at its centre for hit tests.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Box = tuple[float, float, float, float]


def _inside(x: float, y: float, boxes: Sequence[Box], tolerance: float) -> bool:
    return any(
        x0 - tolerance <= x <= x1 + tolerance and y0 - tolerance <= y <= y1 + tolerance
        for x0, y0, x1, y1 in boxes
    )


def pointing_hit(attr: np.ndarray, boxes: Sequence[Box], tolerance: float = 15.0) -> bool:
    """Pointing game (Zhang et al., 2018): is the maximum within ``tolerance`` px of a box?"""
    if not boxes:
        raise ValueError("pointing game needs at least one box")
    y, x = np.unravel_index(int(np.argmax(attr)), attr.shape)
    return _inside(x + 0.5, y + 0.5, boxes, tolerance)


def unit_pointing_hit(
    units: np.ndarray, scores: np.ndarray, boxes: Sequence[Box], tolerance: float = 15.0
) -> bool:
    """Pointing game for region units: is the centroid of the top-scoring unit inside a box?

    ``units`` is an (H, W) label map with values in ``[0, len(scores))``. Used for both the
    patch grid and superpixels, so the two are scored by the same rule.
    """
    if not boxes:
        raise ValueError("pointing game needs at least one box")
    top = int(np.argmax(scores))
    ys, xs = np.nonzero(units == top)
    return _inside(float(xs.mean()) + 0.5, float(ys.mean()) + 0.5, boxes, tolerance)


def box_mask(boxes: Sequence[Box], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    yc = np.arange(height) + 0.5
    xc = np.arange(width) + 0.5
    for x0, y0, x1, y1 in boxes:
        mask |= ((yc >= y0) & (yc <= y1))[:, None] & ((xc >= x0) & (xc <= x1))[None, :]
    return mask


def energy_in_boxes(attr: np.ndarray, boxes: Sequence[Box]) -> float:
    """Share of positive attribution mass inside the union of boxes (energy pointing game)."""
    positive = np.clip(attr, 0, None)
    total = positive.sum()
    if total <= 0:
        return 0.0
    return float(positive[box_mask(boxes, *attr.shape)].sum() / total)
