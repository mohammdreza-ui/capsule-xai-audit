"""Saved predictions: everything downstream of a trained model runs on these, on CPU."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from capsule_xai_audit.data.datasets import to_unit_float
from capsule_xai_audit.eval.metrics import probs_from_logits
from capsule_xai_audit.utils.runtime import autocast

BatchTransform = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Predictions:
    logits: np.ndarray  # (N, K) float32
    labels: np.ndarray  # (N,) int64, -1 if unknown
    positions: np.ndarray  # (N,) row in the index the cache was built from
    image_ids: np.ndarray  # (N,) str
    groups: np.ndarray  # (N,) str; video id, or image id when there is no video
    class_names: tuple[str, ...]

    def probs(self, temperature: float = 1.0) -> np.ndarray:
        return probs_from_logits(self.logits, temperature)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}.tmp.npz")
        np.savez_compressed(
            tmp,
            logits=self.logits.astype(np.float32),
            labels=self.labels,
            positions=self.positions,
            image_ids=self.image_ids.astype(str),
            groups=self.groups.astype(str),
            class_names=np.array(self.class_names),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Predictions:
        with np.load(path, allow_pickle=False) as z:
            return cls(
                logits=z["logits"].astype(np.float64),
                labels=z["labels"],
                positions=z["positions"],
                image_ids=z["image_ids"],
                groups=z["groups"],
                class_names=tuple(z["class_names"].tolist()),
            )


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int, int]],
    device: torch.device,
    amp: bool = True,
    transform: BatchTransform | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(logits, labels, positions) for every sample of ``loader``, in loader order."""
    model.eval()
    logits, labels, positions = [], [], []
    for images, y, pos in loader:
        x = to_unit_float(images.to(device, non_blocking=True))
        if transform is not None:
            x = transform(x)
        with autocast(device, amp):
            out = model(x)
        logits.append(out.float().cpu().numpy())
        labels.append(y.numpy())
        positions.append(pos.numpy())
    return np.concatenate(logits), np.concatenate(labels), np.concatenate(positions)
