"""Frozen-feature extraction and linear probes (Q3, and the Slice A model)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from capsule_xai_audit.data.datasets import to_unit_float
from capsule_xai_audit.models.backbones import Classifier
from capsule_xai_audit.utils.runtime import autocast


@torch.no_grad()
def extract_features(
    model: Classifier, loader: DataLoader[tuple[torch.Tensor, int, int]], device: torch.device, amp: bool
) -> tuple[np.ndarray, np.ndarray]:
    """(features float32 (N, D), positions) for every sample of ``loader``."""
    model.eval()
    feats, positions = [], []
    for images, _, pos in loader:
        with autocast(device, amp):
            f = model.features(to_unit_float(images.to(device, non_blocking=True)))
        feats.append(f.float().cpu().numpy())
        positions.append(pos.numpy())
    return np.concatenate(feats), np.concatenate(positions)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def sample_shots(
    labels: np.ndarray, groups: np.ndarray, candidates: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """``k`` frames per class from ``candidates``, spread over as many videos as possible.

    Videos are visited in random order, one random frame per visit, so a 5-shot support
    set uses 5 different videos whenever the class appears in 5 videos. Classes with
    fewer than ``k`` frames contribute all of them.
    """
    chosen = []
    for cls in np.unique(labels[candidates]):
        members = candidates[labels[candidates] == cls]
        by_video: dict[str, list[int]] = {}
        for pos in rng.permutation(members):
            by_video.setdefault(str(groups[pos]), []).append(int(pos))
        queues = [by_video[v] for v in rng.permutation(sorted(by_video))]
        picked: list[int] = []
        while len(picked) < k and any(queues):
            for q in queues:
                if q and len(picked) < k:
                    picked.append(q.pop())
        chosen.extend(picked)
    return np.array(sorted(chosen), dtype=np.int64)


def center_l2(x: np.ndarray, center: np.ndarray) -> np.ndarray:
    """CL2N: subtract ``center``, then L2-normalise.

    Frozen ImageNet/DINOv2 features of endoscopy frames share a large common component, so
    after plain L2 normalisation they are nearly parallel and a regularised probe leans on
    its intercepts (class priors) instead of the features. Centring on the mean of the
    *unlabelled* training pool removes that component without using any labels.
    """
    return l2_normalize(x - center)


@dataclass(frozen=True)
class LinearProbe:
    """Multinomial logistic regression on centred, L2-normalised features.

    ``weight``/``bias`` are expanded to all ``n_classes`` columns; a class absent from
    the support set gets a large negative bias (probability ~0).
    """

    weight: np.ndarray  # (K, D)
    bias: np.ndarray  # (K,)
    center: np.ndarray  # (D,)

    def logits(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(center_l2(features, self.center) @ self.weight.T + self.bias)


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    C: float,
    center: np.ndarray,
    seed: int = 0,
) -> LinearProbe:
    """``center`` should be the mean feature of the unlabelled training pool."""
    center = np.asarray(center, dtype=np.float64)
    x = center_l2(features.astype(np.float64), center)
    present = np.unique(labels)
    weight = np.zeros((n_classes, x.shape[1]))
    bias = np.full(n_classes, -30.0)
    if len(present) == 1:
        bias[present[0]] = 0.0
        return LinearProbe(weight, bias, center)
    clf = LogisticRegression(C=C, max_iter=5000, random_state=seed)
    clf.fit(x, labels)
    coef, intercept = clf.coef_, clf.intercept_
    if len(present) == 2:  # sklearn returns one row for binary problems
        coef = np.vstack([-coef[0] / 2, coef[0] / 2])
        intercept = np.array([-intercept[0] / 2, intercept[0] / 2])
    weight[clf.classes_] = coef
    bias[clf.classes_] = intercept
    return LinearProbe(weight, bias, center)
