"""Deep ensembles (Lakshminarayanan et al., 2017) from independently seeded models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import logsumexp

from capsule_xai_audit.eval.metrics import log_probs_from_logits


def ensemble_log_probs(member_logits: Sequence[np.ndarray], temperature: float = 1.0) -> np.ndarray:
    """log of the mean member probability, computed stably in log space.

    The result can be treated as logits of the ensemble: softmax of it returns the mean
    probability, and dividing it by a temperature is "pool-then-calibrate" scaling.
    """
    if not member_logits:
        raise ValueError("an ensemble needs at least one member")
    stacked = np.stack([log_probs_from_logits(z, temperature) for z in member_logits])
    return np.asarray(logsumexp(stacked, axis=0) - np.log(len(member_logits)))


def ensemble_probs(member_logits: Sequence[np.ndarray], temperature: float = 1.0) -> np.ndarray:
    return np.exp(ensemble_log_probs(member_logits, temperature))
