"""Seeding for reproducible runs."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA).

    ``deterministic=True`` additionally forces deterministic cuDNN kernels. It is off by
    default: seed-to-seed variance is what the experiments measure, and bit-exact reruns
    are not worth the slowdown on a T4.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds legacy global RNG used by libraries
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker a distinct but reproducible NumPy/random seed.

    Inside a worker ``torch.initial_seed()`` is already ``base_seed + worker_id``.
    """
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002
    random.seed(worker_seed)


def derive_seed(*parts: int | str) -> int:
    """Stable 31-bit seed from a tuple of parts (independent of ``PYTHONHASHSEED``)."""
    digest = hashlib.sha256("/".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF
