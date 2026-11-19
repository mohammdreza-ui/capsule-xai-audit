"""Finding completed training runs and their saved predictions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from capsule_xai_audit.eval.predict import Predictions
from capsule_xai_audit.utils.io import read_json

_SEED_DIR = re.compile(r"seed(\d+)$")


@dataclass(frozen=True)
class RunGroup:
    """All seeds of one (protocol, backbone, tag, fold) combination."""

    protocol: str
    backbone: str
    tag: str
    test_fold: int

    def directory(self, runs_root: Path) -> Path:
        return runs_root / "train" / self.protocol / self.backbone / self.tag / f"fold{self.test_fold}"

    def label(self) -> str:
        return f"{self.protocol}/{self.backbone}/{self.tag}/fold{self.test_fold}"


def completed_runs(group_dir: Path) -> list[tuple[int, Path]]:
    runs = []
    for d in sorted(group_dir.glob("seed*")):
        m = _SEED_DIR.search(d.name)
        meta = d / "run.json"
        if m and meta.exists() and read_json(meta).get("complete"):
            runs.append((int(m.group(1)), d))
    return sorted(runs)


def load_split_predictions(group: RunGroup, runs_root: Path, split: str) -> dict[int, Predictions]:
    """``{seed: predictions}``; all seeds must cover the same frames in the same order."""
    out: dict[int, Predictions] = {}
    for seed, run_dir in completed_runs(group.directory(runs_root)):
        path = run_dir / f"preds_{split}.npz"
        if path.exists():
            out[seed] = Predictions.load(path)
    if not out:
        raise FileNotFoundError(f"no completed runs with preds_{split}.npz in {group.directory(runs_root)}")
    first = next(iter(out.values()))
    for seed, preds in out.items():
        if not np.array_equal(preds.image_ids, first.image_ids):
            raise ValueError(f"{group.label()} seed {seed}: {split} frames differ from the other seeds")
    return out
