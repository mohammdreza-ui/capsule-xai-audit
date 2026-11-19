"""Classification and calibration metrics on probability matrices (NumPy, CPU).

Conventions: ``probs`` is (N, K) with rows summing to one, ``labels`` is (N,) int.
Macro averages run over the classes present in ``labels`` (a class absent from a test
fold cannot have a meaningful F1).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import log_softmax, softmax

EPS = 1e-12


def probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return np.asarray(softmax(np.asarray(logits, dtype=np.float64) / temperature, axis=1))


def log_probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return np.asarray(log_softmax(np.asarray(logits, dtype=np.float64) / temperature, axis=1))


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(1) == labels).mean())


def _confusion_counts(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = probs.shape[1]
    pred = probs.argmax(1)
    tp = np.bincount(labels[pred == labels], minlength=k)
    support = np.bincount(labels, minlength=k)
    predicted = np.bincount(pred, minlength=k)
    return tp, support, predicted


def macro_f1(probs: np.ndarray, labels: np.ndarray) -> float:
    """Macro F1 over the classes present in ``labels``.

    Counting-based rather than ``sklearn.metrics.f1_score``: identical result, but fast
    enough to call thousands of times inside a bootstrap.
    """
    tp, support, predicted = _confusion_counts(probs, labels)
    present = support > 0
    denom = (support + predicted)[present]
    f1 = np.where(denom > 0, 2 * tp[present] / np.maximum(denom, 1), 0.0)
    return float(f1.mean())


def macro_f1_over(classes: Sequence[int]) -> Callable[[np.ndarray, np.ndarray], float]:
    """Macro F1 averaged over a fixed class set, e.g. the classes every compared test set has.

    All frames still count (a frame of another class predicted as one of ``classes`` is a
    false positive); only the average is restricted.
    """
    keep = np.asarray(classes, dtype=int)

    def fn(probs: np.ndarray, labels: np.ndarray) -> float:
        tp, support, predicted = _confusion_counts(probs, labels)
        denom = (support + predicted)[keep]
        return float(np.where(denom > 0, 2 * tp[keep] / np.maximum(denom, 1), 0.0).mean())

    return fn


def balanced_accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean recall over the classes present in ``labels``."""
    tp, support, _ = _confusion_counts(probs, labels)
    present = support > 0
    return float((tp[present] / support[present]).mean())


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], EPS, 1.0)).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean over samples of sum_k (p_k - y_k)^2, in [0, 2]."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


@dataclass(frozen=True)
class ReliabilityBins:
    edges: np.ndarray  # (B + 1,)
    confidence: np.ndarray  # (B,) mean confidence per bin (nan if empty)
    accuracy: np.ndarray  # (B,) accuracy per bin (nan if empty)
    count: np.ndarray  # (B,)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> ReliabilityBins:
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins (lo, hi]; confidence is always > 0, the first bin includes 0.
    bin_idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    count = np.bincount(bin_idx, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.bincount(bin_idx, weights=conf, minlength=n_bins) / count
        mean_acc = np.bincount(bin_idx, weights=correct, minlength=n_bins) / count
    return ReliabilityBins(edges, mean_conf, mean_acc, count)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-label expected calibration error with equal-width bins."""
    bins = reliability_bins(probs, labels, n_bins)
    nonempty = bins.count > 0
    weights = bins.count[nonempty] / bins.count.sum()
    return float((weights * np.abs(bins.accuracy[nonempty] - bins.confidence[nonempty])).sum())


def classwise_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Mean over present classes of the one-vs-rest ECE of p_k."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    errors = []
    for k in np.unique(labels):
        p = probs[:, k]
        y = (labels == k).astype(np.float64)
        idx = np.clip(np.searchsorted(edges, p, side="left") - 1, 0, n_bins - 1)
        count = np.bincount(idx, minlength=n_bins)
        nonempty = count > 0
        mean_p = np.bincount(idx, weights=p, minlength=n_bins)[nonempty] / count[nonempty]
        mean_y = np.bincount(idx, weights=y, minlength=n_bins)[nonempty] / count[nonempty]
        errors.append(float((count[nonempty] / len(p) * np.abs(mean_p - mean_y)).sum()))
    return float(np.mean(errors))


METRICS = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "balanced_accuracy": balanced_accuracy,
    "nll": nll,
    "brier": brier,
    "ece": ece,
    "classwise_ece": classwise_ece,
}
LOWER_IS_BETTER = frozenset({"nll", "brier", "ece", "classwise_ece"})


def compute_metrics(
    probs: np.ndarray, labels: np.ndarray, names: tuple[str, ...] | None = None
) -> dict[str, float]:
    selected = names if names is not None else tuple(METRICS)
    return {name: METRICS[name](probs, labels) for name in selected}
