"""Temperature scaling (Guo et al., 2017)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

LOG_T_BOUNDS = (-4.0, 4.0)  # T in [0.018, 54.6]


def _nll_at(logits: np.ndarray, labels: np.ndarray, log_t: float) -> float:
    logp = log_softmax(logits / np.exp(log_t), axis=1)
    return float(-logp[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature minimising validation NLL.

    The NLL is convex in 1/T, hence unimodal in log T, so a bounded scalar search finds
    the global optimum without the step-size tuning that LBFGS on T needs.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    result = minimize_scalar(
        lambda lt: _nll_at(logits, labels, lt), bounds=LOG_T_BOUNDS, method="bounded", options={"xatol": 1e-5}
    )
    if not result.success:
        raise RuntimeError(f"temperature fit failed: {result.message}")
    return float(np.exp(result.x))
