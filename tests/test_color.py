from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from skimage import color as skcolor

from capsule_xai_audit.slices.color import (
    ColorShift,
    LabStatsAccumulator,
    RandomLabShift,
    ReinhardTransfer,
    delta_e76,
    lab_to_rgb,
    rgb_to_lab,
)


def _batch(seed: int = 0, n: int = 2, size: int = 16) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, size, size, generator=g, dtype=torch.float64)


def test_lab_matches_skimage() -> None:
    rgb = _batch()
    ours = rgb_to_lab(rgb).permute(0, 2, 3, 1).numpy()
    ref = np.stack([skcolor.rgb2lab(img) for img in rgb.permute(0, 2, 3, 1).numpy()])
    np.testing.assert_allclose(ours, ref, atol=1e-3)


def test_lab_roundtrip() -> None:
    rgb = _batch(1)
    np.testing.assert_allclose(lab_to_rgb(rgb_to_lab(rgb)).numpy(), rgb.numpy(), atol=1e-6)


def test_controlled_shift_has_nominal_delta_e_in_gamut() -> None:
    # Mid-grey mucosa-like colours stay in gamut for a 10 ΔE cast.
    rgb = 0.4 + 0.2 * _batch(2)
    for hue in (0.0, 90.0, 225.0):
        shifted = ColorShift(10.0, hue)(rgb)
        realised = delta_e76(rgb_to_lab(rgb), rgb_to_lab(shifted))
        assert realised.mean().item() == pytest.approx(10.0, abs=0.05)
        d_ab = rgb_to_lab(shifted) - rgb_to_lab(rgb)
        angle = math.degrees(math.atan2(d_ab[:, 2].mean().item(), d_ab[:, 1].mean().item())) % 360
        assert angle == pytest.approx(hue, abs=0.5)


def test_zero_shift_is_identity_and_clipping_keeps_range() -> None:
    rgb = _batch(3)
    assert torch.equal(ColorShift(0.0, 45.0)(rgb), rgb)
    shifted = ColorShift(40.0, 30.0)(rgb)
    assert shifted.min() >= 0
    assert shifted.max() <= 1


def test_random_shift_respects_bound() -> None:
    rgb = 0.4 + 0.2 * _batch(4, n=64, size=4)
    out = RandomLabShift(max_delta_e=8.0, p=1.0)(rgb, torch.Generator().manual_seed(0))
    per_image = delta_e76(rgb_to_lab(rgb), rgb_to_lab(out)).mean(dim=(1, 2))
    assert per_image.max() <= 8.0 + 1e-6
    assert per_image.std() > 0.5


def test_reinhard_transfer_matches_target_statistics() -> None:
    src = 0.3 + 0.3 * _batch(5, n=8)
    tgt = 0.5 + 0.2 * _batch(6, n=8)
    acc_s, acc_t = LabStatsAccumulator(), LabStatsAccumulator()
    acc_s.update(src)
    acc_t.update(tgt)
    mapped = ReinhardTransfer(acc_s.result(), acc_t.result())(src)
    acc_m = LabStatsAccumulator()
    acc_m.update(mapped)
    np.testing.assert_allclose(acc_m.result().mean, acc_t.result().mean, atol=0.5)
    np.testing.assert_allclose(acc_m.result().std, acc_t.result().std, atol=0.5)
