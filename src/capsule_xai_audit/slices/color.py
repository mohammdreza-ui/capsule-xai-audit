"""CIELAB colour operations in PyTorch (Slice B).

All functions take and return batches of sRGB images in [0, 1], shape (B, 3, H, W), and
run on whatever device the batch is on. The sRGB <-> XYZ <-> Lab conversion uses the D65
white point and the same matrices as ``skimage.color`` (the tests check agreement).

A *controlled* shift moves every pixel by ``delta_e`` in the a*b* plane along a fixed hue
direction, i.e. a colour cast of exactly ``delta_e`` ΔE76 units before gamut clipping.
Clipping back into sRGB shrinks the shift for saturated pixels, so the realised ΔE is
measured and reported next to the nominal one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_RGB_TO_XYZ = torch.tensor(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ],
    dtype=torch.float64,
)
_XYZ_TO_RGB = torch.linalg.inv(_RGB_TO_XYZ)
_WHITE_D65 = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64)
_DELTA = 6.0 / 29.0


def _apply_matrix(m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,bjhw->bihw", m.to(dtype=x.dtype, device=x.device), x)


def _srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92, ((c.clamp(min=0) + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.0031308, 12.92 * c, 1.055 * c.clamp(min=0) ** (1 / 2.4) - 0.055)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    xyz = _apply_matrix(_RGB_TO_XYZ, _srgb_to_linear(rgb))
    xyz = xyz / _WHITE_D65.to(dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    f = torch.where(xyz > _DELTA**3, xyz.clamp(min=1e-12) ** (1 / 3), xyz / (3 * _DELTA**2) + 4 / 29)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    return torch.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dim=1)


def lab_to_rgb(lab: torch.Tensor, clip: bool = True) -> torch.Tensor:
    fy = (lab[:, 0] + 16) / 116
    fx = fy + lab[:, 1] / 500
    fz = fy - lab[:, 2] / 200
    f = torch.stack([fx, fy, fz], dim=1)
    xyz = torch.where(f > _DELTA, f**3, 3 * _DELTA**2 * (f - 4 / 29))
    xyz = xyz * _WHITE_D65.to(dtype=lab.dtype, device=lab.device).view(1, 3, 1, 1)
    rgb = _linear_to_srgb(_apply_matrix(_XYZ_TO_RGB, xyz))
    return rgb.clamp(0, 1) if clip else rgb


def delta_e76(lab_a: torch.Tensor, lab_b: torch.Tensor) -> torch.Tensor:
    """Per-pixel Euclidean distance in Lab, shape (B, H, W)."""
    return torch.linalg.vector_norm(lab_a - lab_b, dim=1)


def shift_ab(rgb: torch.Tensor, delta_e: torch.Tensor | float, hue_rad: torch.Tensor | float) -> torch.Tensor:
    """Shift a*b* by ``delta_e`` along ``hue_rad``; scalars or per-sample (B,) tensors."""
    lab = rgb_to_lab(rgb)
    de = torch.as_tensor(delta_e, dtype=lab.dtype, device=lab.device).reshape(-1, 1, 1)
    hue = torch.as_tensor(hue_rad, dtype=lab.dtype, device=lab.device).reshape(-1, 1, 1)
    shifted = lab.clone()
    shifted[:, 1] = lab[:, 1] + de * torch.cos(hue)
    shifted[:, 2] = lab[:, 2] + de * torch.sin(hue)
    return lab_to_rgb(shifted).to(rgb.dtype)


@dataclass(frozen=True)
class ColorShift:
    """Fixed shift used at evaluation time."""

    delta_e: float
    hue_deg: float

    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.delta_e == 0:
            return rgb
        return shift_ab(rgb, self.delta_e, math.radians(self.hue_deg))


class RandomLabShift:
    """Training augmentation: with probability ``p`` a cast of ΔE ~ U(0, max) at a random hue."""

    def __init__(self, max_delta_e: float, p: float = 0.8):
        if max_delta_e <= 0:
            raise ValueError("max_delta_e must be positive")
        self.max_delta_e = max_delta_e
        self.p = p

    def __call__(self, rgb: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        n, device = rgb.shape[0], rgb.device
        apply = torch.rand(n, generator=generator, device=device) < self.p
        de = torch.rand(n, generator=generator, device=device) * self.max_delta_e * apply
        hue = torch.rand(n, generator=generator, device=device) * (2 * math.pi)
        return shift_ab(rgb, de, hue)


@dataclass(frozen=True)
class LabStats:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class LabStatsAccumulator:
    """Streaming per-channel Lab mean and std over all pixels of many batches."""

    def __init__(self) -> None:
        self.n = 0
        self.total = torch.zeros(3, dtype=torch.float64)
        self.total_sq = torch.zeros(3, dtype=torch.float64)

    def update(self, rgb: torch.Tensor) -> None:
        lab = rgb_to_lab(rgb.double()).transpose(0, 1).reshape(3, -1).cpu()
        self.n += lab.shape[1]
        self.total += lab.sum(1)
        self.total_sq += (lab**2).sum(1)

    def result(self) -> LabStats:
        mean = self.total / self.n
        std = (self.total_sq / self.n - mean**2).clamp(min=1e-12).sqrt()
        return LabStats(tuple(mean.tolist()), tuple(std.tolist()))


class ReinhardTransfer:
    """Match the dataset-level Lab statistics of ``source`` images to ``target`` (Reinhard et al., 2001)."""

    def __init__(self, source: LabStats, target: LabStats):
        self.source, self.target = source, target

    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        lab = rgb_to_lab(rgb)

        def vec(values: tuple[float, float, float]) -> torch.Tensor:
            return torch.tensor(values, dtype=lab.dtype, device=lab.device).view(1, 3, 1, 1)

        mapped = (lab - vec(self.source.mean)) / vec(self.source.std) * vec(self.target.std) + vec(
            self.target.mean
        )
        return lab_to_rgb(mapped).to(rgb.dtype)
