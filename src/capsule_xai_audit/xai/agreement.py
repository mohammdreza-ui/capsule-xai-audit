"""Do different evaluation metrics rank attribution methods the same way? (H4)"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import kendalltau

from capsule_xai_audit.eval.bootstrap import cluster_members, resampled_rows

# metric -> True if higher is better
METRIC_DIRECTION: dict[str, bool] = {
    "pointing": True,
    "energy": True,
    "insertion_auc": True,
    "deletion_auc": False,
}


def method_scores(per_image: Mapping[str, np.ndarray], rows: np.ndarray | None = None) -> dict[str, float]:
    """Mean per method, oriented so that higher is better is handled by the caller."""
    return {m: float(np.mean(v if rows is None else v[rows])) for m, v in per_image.items()}


def ranking(scores: Mapping[str, float], higher_is_better: bool) -> list[str]:
    return sorted(scores, key=lambda m: -scores[m] if higher_is_better else scores[m])


def _tau(
    a: Mapping[str, float], b: Mapping[str, float], dir_a: bool, dir_b: bool, methods: Sequence[str]
) -> float:
    va = np.array([a[m] if dir_a else -a[m] for m in methods])
    vb = np.array([b[m] if dir_b else -b[m] for m in methods])
    return float(kendalltau(va, vb).statistic)


def kendall_agreement(
    per_image: Mapping[str, Mapping[str, np.ndarray]],
    groups: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Kendall's tau between method rankings of every pair of metrics, with bootstrap CIs.

    ``per_image[metric][method]`` holds one score per image; all arrays index the same
    images, which are resampled jointly (by video when ``groups`` is given).
    """
    metrics = [m for m in per_image if m in METRIC_DIRECTION]
    methods = sorted(next(iter(per_image.values())))
    n = len(next(iter(per_image[metrics[0]].values())))
    clusters = cluster_members(np.arange(n) if groups is None else groups)
    rng = np.random.default_rng(seed)

    point_scores = {m: method_scores(per_image[m]) for m in metrics}
    out: dict[str, Any] = {
        "methods": methods,
        "rankings": {m: ranking(point_scores[m], METRIC_DIRECTION[m]) for m in metrics},
        "pairs": {},
    }
    pairs = list(combinations(metrics, 2))
    reps = {p: np.empty(n_boot) for p in pairs}
    for b in range(n_boot):
        rows = resampled_rows(clusters, rng)
        boot = {m: method_scores(per_image[m], rows) for m in metrics}
        for a, c in pairs:
            reps[(a, c)][b] = _tau(boot[a], boot[c], METRIC_DIRECTION[a], METRIC_DIRECTION[c], methods)
    for a, c in pairs:
        point = _tau(point_scores[a], point_scores[c], METRIC_DIRECTION[a], METRIC_DIRECTION[c], methods)
        finite = reps[(a, c)][np.isfinite(reps[(a, c)])]
        low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2]) if len(finite) else (np.nan, np.nan)
        out["pairs"][f"{a}~{c}"] = {"tau": point, "low": float(low), "high": float(high)}
    return out
