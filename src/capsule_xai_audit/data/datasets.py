"""PyTorch datasets over the uint8 cache, and GPU-side batch augmentation.

Workers return uint8 CHW tensors; conversion to float, augmentation and normalisation
run on the GPU (normalisation lives in the model wrapper), which keeps the T4's CPU side
from becoming the bottleneck.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from capsule_xai_audit.data.cache import cache_paths
from capsule_xai_audit.utils.seed import worker_init_fn


class CachedImages(Dataset[tuple[torch.Tensor, int, int]]):
    """Rows ``positions`` of a cache; yields ``(uint8 CHW image, label, position)``.

    The memmap is opened lazily so each DataLoader worker gets its own handle instead of a
    pickled copy.
    """

    def __init__(self, prefix: Path, positions: np.ndarray, labels: np.ndarray | None = None) -> None:
        self.pixels_path = cache_paths(prefix)[0]
        self.positions = np.asarray(positions, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        if self.labels is not None and len(self.labels) != len(self.positions):
            raise ValueError("labels and positions differ in length")
        self._pixels: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.positions)

    def _array(self) -> np.ndarray:
        if self._pixels is None:
            self._pixels = np.load(self.pixels_path, mmap_mode="r")
        return self._pixels

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        pos = int(self.positions[i])
        img = torch.from_numpy(np.array(self._array()[pos])).permute(2, 0, 1)  # copy out of the memmap
        label = -1 if self.labels is None else int(self.labels[i])
        return img, label, pos

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_pixels"] = None
        return state


def make_loader(
    dataset: Dataset[tuple[torch.Tensor, int, int]],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator | None = None,
    drop_last: bool = False,
) -> DataLoader[tuple[torch.Tensor, int, int]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
    )


def to_unit_float(batch: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float32 [0, 1]."""
    return batch.float().div_(255.0)


class GeometricAugment(nn.Module):
    """Random flips and 90° rotations, per sample.

    Capsule frames have no canonical orientation, so the dihedral group is a label-
    preserving augmentation. Optionally followed by a per-sample colour transform.
    """

    def __init__(self, color: Callable[[torch.Tensor, torch.Generator | None], torch.Tensor] | None = None):
        super().__init__()
        self.color = color

    def forward(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        n = x.shape[0]
        device = x.device
        flip = torch.rand(n, generator=generator, device=device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        k = torch.randint(0, 4, (n,), generator=generator, device=device)
        rotated = torch.stack(
            [torch.rot90(xi, int(ki), dims=(-2, -1)) for xi, ki in zip(x, k.tolist(), strict=True)]
        )
        if self.color is not None:
            rotated = self.color(rotated, generator)
        return rotated
