"""Model-parameter randomisation test (Adebayo et al., 2018), cascading from the output layer.

After each randomisation step the attributions of the randomised model are compared
with the original ones by Spearman rank correlation of the maps (downsampled to reduce
pixel-level noise). A method that explains the *model* should lose correlation as more
layers are randomised; a method whose maps stay correlated is mostly reflecting the
input. The ``random`` control is model-independent by construction and stays at 1.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from capsule_xai_audit.models.backbones import Classifier

AttributeFn = Callable[[Classifier], torch.Tensor]


def randomize_groups(model: Classifier, prefixes: Sequence[str], seed: int) -> None:
    """Re-initialise, in place, every module whose name is or starts with one of ``prefixes``."""
    torch.manual_seed(seed)
    touched = 0
    for name, module in model.net.named_modules():
        if any(name == p or name.startswith(p + ".") for p in prefixes) and hasattr(
            module, "reset_parameters"
        ):
            module.reset_parameters()
            touched += 1
    if not touched:
        raise ValueError(f"no module matched {prefixes}")


def map_similarity(a: torch.Tensor, b: torch.Tensor, size: int = 56) -> np.ndarray:
    """Per-image Spearman correlation of two (B, H, W) map batches."""
    a_small = F.adaptive_avg_pool2d(a[:, None].float(), size)[:, 0].flatten(1).cpu().numpy()
    b_small = F.adaptive_avg_pool2d(b[:, None].float(), size)[:, 0].flatten(1).cpu().numpy()
    out = np.empty(len(a_small))
    for i, (u, v) in enumerate(zip(a_small, b_small, strict=True)):
        if np.ptp(u) == 0 or np.ptp(v) == 0:
            out[i] = 1.0 if np.array_equal(u, v) else 0.0
        else:
            out[i] = spearmanr(u, v).statistic
    return out


def cascading_randomization(
    model: Classifier, attribute_fn: AttributeFn, seed: int = 0
) -> list[tuple[str, np.ndarray]]:
    """``[(step label, per-image Spearman)]`` for each cumulative randomisation step."""
    original = attribute_fn(model)
    randomized = copy.deepcopy(model)
    steps = []
    for i, group in enumerate(model.spec.layer_groups):
        randomize_groups(randomized, group, seed + i)
        steps.append(("+".join(group), map_similarity(original, attribute_fn(randomized))))
    return steps
