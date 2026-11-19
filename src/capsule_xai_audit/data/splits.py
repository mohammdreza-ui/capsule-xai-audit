"""Train/test split protocols for Q1 and the inner validation split.

* ``official``: Kvasir-Capsule's two-fold split exactly as published. It is mostly, but
  not fully, video-level: 7 of 43 videos have frames in both folds, because the split
  assigns each (video, class) pair to one fold.
* ``video``: a strictly patient-level variant of the official split. Every video that
  appears in both official folds moves, with all its frames, to the fold that holds most
  of them; all other videos keep their official fold.
* ``random``: the naive frame-level protocol. The test set has exactly as many frames,
  with the same class proportions, as the official test fold, but frames are drawn from
  all videos, so neighbouring frames of one video land on both sides.

The inner validation split (early stopping, temperature fitting) is frame-level for
``random``. For ``official`` and ``video`` it is built from whole videos wherever the data
allow it (see ``grouped_inner_split``): several Kvasir-Capsule classes occur in only one or
two videos per fold, and a plain grouped split (version 1) left Blood out of the fit set
and most classes out of validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from capsule_xai_audit.config import SplitProtocol

# Bumped whenever the split algorithm changes; stored in run.json so stale runs are retrained.
SPLIT_VERSION = 2


class LeakageError(AssertionError):
    pass


@dataclass(frozen=True)
class Split:
    """Row positions into the index."""

    fit: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        parts = [set(self.fit.tolist()), set(self.val.tolist()), set(self.test.tolist())]
        if parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]:
            raise LeakageError("a frame appears in more than one of fit/val/test")


def assert_disjoint_groups(groups: pd.Series, a: np.ndarray, b: np.ndarray, what: str) -> None:
    shared = set(groups.iloc[a]) & set(groups.iloc[b])
    if shared:
        raise LeakageError(f"{len(shared)} videos are shared between {what}: {sorted(shared)[:5]}")


def _stratify_or_none(labels: np.ndarray) -> np.ndarray | None:
    """Stratify only when every class has at least two members."""
    _, counts = np.unique(labels, return_counts=True)
    return labels if counts.min() >= 2 else None


def video_folds(index: pd.DataFrame) -> np.ndarray:
    """Per-row fold of the ``video`` protocol: each video in the fold holding most of its frames.

    Ties go to fold 0. Videos already inside one official fold keep it.
    """
    counts = index.groupby(["video_id", "official_fold"]).size().unstack(fill_value=0)
    majority = counts.idxmax(axis=1)  # first maximum, i.e. the lower fold on ties
    return index["video_id"].map(majority).to_numpy().astype(int)


def outer_split(
    index: pd.DataFrame, protocol: SplitProtocol, test_fold: int, split_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(len(index))
    official_test = positions[index["official_fold"].to_numpy() == test_fold]
    official_train = positions[index["official_fold"].to_numpy() != test_fold]
    if protocol == "official":
        return official_train, official_test
    if protocol == "video":
        folds = video_folds(index)
        train, test = positions[folds != test_fold], positions[folds == test_fold]
        assert_disjoint_groups(index["video_id"], train, test, "train and test folds")
        return train, test

    labels = index["label_idx"].to_numpy()
    # Draw the same number of test frames per class as the official test fold has.
    rng = np.random.default_rng(split_seed)
    test_parts = []
    for cls, n_test in zip(*np.unique(labels[official_test], return_counts=True), strict=True):
        members = positions[labels == cls]
        test_parts.append(rng.choice(members, size=n_test, replace=False))
    test = np.sort(np.concatenate(test_parts))
    train = np.setdiff1d(positions, test, assume_unique=True)
    return train, test


def frame_numbers(index: pd.DataFrame) -> np.ndarray:
    """Frame number of every row, from Kvasir-Capsule image ids ``<video_id>_<frame>``."""
    return index["image_id"].str.rsplit("_", n=1).str[1].astype(int).to_numpy()


def grouped_inner_split(
    labels: np.ndarray, videos: np.ndarray, frames: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Local fit/validation indices, video-disjoint wherever the class structure allows it.

    1. Whole videos move to validation, greedily: first those that add classes validation
       does not have yet, then any, until about ``val_fraction`` of the frames. A video is
       never moved if that would take the last fit video of one of its classes, or put more
       than ``2 * val_fraction`` of a class's frames into validation (a class with two
       videos of very different length would otherwise lose most of its training frames).
    2. Every class that still has no validation frame (it lives in a single video, or all
       its videos had to stay in fit) gets a temporal block: the last ``val_fraction`` of its
       frames, by frame number, in the fit video that holds most of it. Those frames are
       neighbours of fit frames, so that class's validation score is optimistic; the block
       only keeps model selection and temperature fitting from ignoring the class.

    Every class with at least two frames ends up in both fit and validation.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    target = val_fraction * n
    order = rng.permutation(np.unique(videos))
    rows_of = {v: np.flatnonzero(videos == v) for v in order}
    counts_of = {
        v: dict(zip(*np.unique(labels[r], return_counts=True), strict=True)) for v, r in rows_of.items()
    }
    classes_of = {v: set(c) for v, c in counts_of.items()}
    fit_videos = {c: {v for v in order if c in classes_of[v]} for c in np.unique(labels)}
    total = dict(zip(*np.unique(labels, return_counts=True), strict=True))
    in_val_count = dict.fromkeys(total, 0)

    val_videos: list[str] = []
    covered: set[int] = set()
    size = 0
    while True:
        movable = [
            v
            for v in order
            if v not in val_videos
            and size + len(rows_of[v]) <= 1.5 * target
            and all(
                len(fit_videos[c]) > 1 and in_val_count[c] + k <= 2 * val_fraction * total[c]
                for c, k in counts_of[v].items()
            )
        ]
        if not movable:
            break
        best = max(movable, key=lambda v: len(classes_of[v] - covered))  # first maximum: seeded order
        if not classes_of[best] - covered and size >= target:
            break
        val_videos.append(best)
        size += len(rows_of[best])
        covered |= classes_of[best]
        for c, k in counts_of[best].items():
            fit_videos[c].discard(best)
            in_val_count[c] += k

    in_val = np.zeros(n, dtype=bool)
    for v in val_videos:
        in_val[rows_of[v]] = True
    for c in np.unique(labels):
        members = np.flatnonzero((labels == c) & ~in_val)
        if (labels[in_val] == c).any() or len(members) < 2:
            continue
        counts = pd.Series(videos[members]).value_counts()
        host = members[videos[members] == counts.index[0]]
        take = min(max(1, round(val_fraction * len(members))), len(host) - 1) if len(host) > 1 else 0
        if take == 0:  # a single frame in its video: take one frame from another video
            host = members[videos[members] != counts.index[0]]
            take = 1
        tail = host[np.argsort(frames[host], kind="stable")][-take:]
        in_val[tail] = True
    return np.flatnonzero(~in_val), np.flatnonzero(in_val)


def inner_split(
    index: pd.DataFrame, train: np.ndarray, protocol: SplitProtocol, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    labels = index["label_idx"].to_numpy()[train]
    if protocol in ("official", "video"):
        fit_local, val_local = grouped_inner_split(
            labels,
            index["video_id"].to_numpy()[train],
            frame_numbers(index)[train],
            val_fraction,
            seed,
        )
        return np.sort(train[fit_local]), np.sort(train[val_local])
    fit, val = train_test_split(
        train, test_size=val_fraction, random_state=seed, stratify=_stratify_or_none(labels)
    )
    return np.sort(fit), np.sort(val)


def class_video_counts(index: pd.DataFrame, split: Split) -> dict[str, dict[str, list[int]]]:
    """Per part and class: [frames, videos]. Written to run.json so a split can be audited."""
    out: dict[str, dict[str, list[int]]] = {}
    for part in ("fit", "val", "test"):
        rows = index.iloc[getattr(split, part)]
        grouped = rows.groupby("label")["video_id"].agg(["size", "nunique"])
        out[part] = {str(k): [int(a), int(b)] for k, (a, b) in grouped.iterrows()}
    return out


def make_split(
    index: pd.DataFrame,
    protocol: SplitProtocol,
    test_fold: int,
    split_seed: int,
    val_fraction: float,
) -> Split:
    """Outer split plus inner validation split.

    The inner split is seeded by ``split_seed`` so that every training seed of one
    protocol shares fit/val/test sets; seeds then differ only in initialisation of the
    head and in data order, which is the variance a deep ensemble averages over.
    """
    train, test = outer_split(index, protocol, test_fold, split_seed)
    fit, val = inner_split(index, train, protocol, val_fraction, split_seed)
    return Split(fit=fit, val=val, test=test)
