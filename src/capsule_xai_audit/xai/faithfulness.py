"""Deletion / insertion faithfulness curves (Petsiuk et al., 2018, RISE).

Pixels are ranked by attribution (most important first). *Deletion* replaces the top
fraction of pixels with a baseline and tracks the target-class probability: a faithful
map makes it drop fast (low AUC is better). *Insertion* starts from the baseline and
reveals pixels in the same order: a faithful map makes it rise fast (high AUC is better).
The baseline is a heavily blurred copy of the image, which stays closer to the data
distribution than a black image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur

from capsule_xai_audit.utils.runtime import autocast


@dataclass(frozen=True)
class Curves:
    deletion: np.ndarray  # (B, steps + 1) target probability
    insertion: np.ndarray  # (B, steps + 1)

    @property
    def deletion_auc(self) -> np.ndarray:
        return auc(self.deletion)

    @property
    def insertion_auc(self) -> np.ndarray:
        return auc(self.insertion)


def auc(curves: np.ndarray) -> np.ndarray:
    """Trapezoidal area under curves sampled at evenly spaced fractions of [0, 1]."""
    steps = curves.shape[1] - 1
    return np.asarray((curves[:, 1:] + curves[:, :-1]).sum(1) / (2 * steps))


def blur_baseline(x: torch.Tensor, relative_sigma: float = 0.045) -> torch.Tensor:
    """Gaussian blur with sigma proportional to image size (about 10 px at 224 px)."""
    side = min(x.shape[-2:])
    sigma = max(1.0, relative_sigma * side)
    kernel = min(int(2 * round(3 * sigma) + 1), side - 1 if side % 2 == 0 else side)
    return gaussian_blur(x, kernel_size=[kernel, kernel], sigma=[sigma, sigma])


def pixel_ranks(maps: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Rank of every pixel (0 = most important) with random tie-breaking, shape (B, H, W)."""
    b, h, w = maps.shape
    flat = maps.reshape(b, -1).float()
    jitter = (
        torch.rand(flat.shape, generator=generator).to(flat.device)
        * 1e-6
        * (flat.abs().amax(1, keepdim=True) + 1e-12)
    )
    order = torch.argsort(flat + jitter, dim=1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(h * w, device=maps.device).expand(b, -1))
    return ranks.reshape(b, h, w)


@torch.no_grad()
def deletion_insertion(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    maps: torch.Tensor,
    steps: int = 50,
    batch_size: int = 64,
    amp: bool = True,
    seed: int = 0,
    baseline: Literal["blur", "zero"] = "blur",
) -> Curves:
    model.eval()
    device = x.device
    b, _, h, w = x.shape
    reference = blur_baseline(x) if baseline == "blur" else torch.zeros_like(x)
    ranks = pixel_ranks(maps, torch.Generator().manual_seed(seed))
    counts = torch.round(torch.linspace(0, h * w, steps + 1, device=device)).long()
    deletion = np.empty((b, steps + 1))
    insertion = np.empty((b, steps + 1))
    for i in range(b):
        revealed = (ranks[i][None] < counts[:, None, None]).unsqueeze(1).to(x.dtype)  # (S+1, 1, H, W)
        for kind, out in (("deletion", deletion), ("insertion", insertion)):
            if kind == "deletion":
                batch = x[i] * (1 - revealed) + reference[i] * revealed
            else:
                batch = reference[i] * (1 - revealed) + x[i] * revealed
            probs = []
            for start in range(0, len(batch), batch_size):
                with autocast(device, amp):
                    logits = model(batch[start : start + batch_size])
                probs.append(torch.softmax(logits.float(), 1)[:, target[i]].cpu())
            out[i] = torch.cat(probs).numpy()
    return Curves(deletion, insertion)
