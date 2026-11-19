"""Pre-registered Kvasir-Capsule <-> Capsule Vision 2024 class mapping (configs/class_mapping.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


class ClassMappingError(ValueError):
    pass


@dataclass(frozen=True)
class ClassMapping:
    kvasir_labels: tuple[str, ...]
    capsule_vision_labels: tuple[str, ...]
    shared: dict[str, str]  # kvasir label -> capsule vision label

    def __post_init__(self) -> None:
        if len(set(self.kvasir_labels)) != len(self.kvasir_labels):
            raise ClassMappingError("duplicate Kvasir labels")
        bad_k = set(self.shared) - set(self.kvasir_labels)
        bad_c = set(self.shared.values()) - set(self.capsule_vision_labels)
        if bad_k or bad_c:
            raise ClassMappingError(f"shared map uses unknown labels: kvasir {bad_k}, capsule vision {bad_c}")
        if len(set(self.shared.values())) != len(self.shared):
            raise ClassMappingError("shared map is not one-to-one")

    @property
    def shared_kvasir_indices(self) -> np.ndarray:
        """Kvasir class indices (model output columns) that have a Capsule Vision counterpart."""
        return np.array([self.kvasir_labels.index(k) for k in self.kvasir_labels if k in self.shared])

    @property
    def shared_kvasir_labels(self) -> tuple[str, ...]:
        return tuple(self.kvasir_labels[i] for i in self.shared_kvasir_indices)

    def capsule_vision_to_kvasir_index(self) -> dict[str, int]:
        inverse = {cv: k for k, cv in self.shared.items()}
        return {cv: self.kvasir_labels.index(k) for cv, k in inverse.items()}


def load_class_mapping(path: Path) -> ClassMapping:
    data = yaml.safe_load(Path(path).read_text())
    try:
        return ClassMapping(
            kvasir_labels=tuple(data["kvasir_labels"]),
            capsule_vision_labels=tuple(data["capsule_vision_labels"]),
            shared=dict(data["shared"]),
        )
    except KeyError as err:
        raise ClassMappingError(f"{path}: missing key {err}") from err


def restrict_to_classes(
    logits: np.ndarray, labels: np.ndarray, keep: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only frames whose label is in ``keep`` and only the ``keep`` logit columns.

    Softmax over the kept columns renormalises the probabilities over the shared classes.
    Labels are re-indexed to positions in ``keep``. Returns (logits, labels, row mask).
    """
    keep = np.asarray(keep)
    remap = -np.ones(logits.shape[1], dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    mask = np.isin(labels, keep)
    return logits[mask][:, keep], remap[labels[mask]], mask
