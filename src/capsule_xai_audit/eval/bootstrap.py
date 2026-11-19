"""Cluster bootstrap confidence intervals.

Consecutive capsule frames of one video are strongly correlated, so resampling frames
would understate uncertainty. Intervals here resample whole videos (clusters) with
replacement; with ``groups=None`` every frame is its own cluster.

Several models evaluated on the same frames (e.g. the seeds of one protocol) are
bootstrapped jointly: one resample of videos is applied to every model and the metric is
averaged over models, so both seed variance and test-set variance enter the interval.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    replicates: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {"point": self.point, "low": self.low, "high": self.high}

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def _interval(point: float, reps: np.ndarray, alpha: float) -> Interval:
    low, high = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return Interval(float(point), float(low), float(high), reps)


def cluster_members(groups: np.ndarray) -> list[np.ndarray]:
    """Row indices of each cluster."""
    _, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, boundaries)


def resampled_rows(clusters: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    pick = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[i] for i in pick])


def bootstrap_metrics(
    metrics: Mapping[str, MetricFn],
    probs: Sequence[np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Interval]:
    """CI of ``mean_m metric(probs[m], labels)`` for every metric, under one cluster bootstrap."""
    if not probs:
        raise ValueError("need at least one probability matrix")
    n = len(labels)
    if any(len(p) != n for p in probs):
        raise ValueError("all probability matrices must cover the same frames")
    clusters = cluster_members(np.arange(n) if groups is None else np.asarray(groups))
    rng = np.random.default_rng(seed)
    reps = {name: np.empty(n_boot) for name in metrics}
    for b in range(n_boot):
        rows = resampled_rows(clusters, rng)
        y = labels[rows]
        sub = [p[rows] for p in probs]
        for name, fn in metrics.items():
            reps[name][b] = np.mean([fn(p, y) for p in sub])
    return {
        name: _interval(float(np.mean([fn(p, labels) for p in probs])), reps[name], alpha)
        for name, fn in metrics.items()
    }


def difference(a: Interval, b: Interval, alpha: float = 0.05) -> Interval:
    """CI of ``a - b`` for two *independently* bootstrapped quantities (different test sets)."""
    n = min(len(a.replicates), len(b.replicates))
    return _interval(a.point - b.point, a.replicates[:n] - b.replicates[:n], alpha)


def paired_bootstrap(
    stat: Callable[[np.ndarray], float],
    n_items: int,
    groups: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Interval:
    """Cluster bootstrap of ``stat(rows)``, where ``rows`` index items shared by both arms.

    Used for paired comparisons on the same frames (ensemble vs single model, patch vs
    superpixel pointing), where resampling both arms together removes item variance.
    """
    clusters = cluster_members(np.arange(n_items) if groups is None else np.asarray(groups))
    rng = np.random.default_rng(seed)
    reps = np.array([stat(resampled_rows(clusters, rng)) for _ in range(n_boot)])
    return _interval(stat(np.arange(n_items)), reps, alpha)
