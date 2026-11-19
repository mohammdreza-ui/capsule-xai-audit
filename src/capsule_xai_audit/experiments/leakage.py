"""Q1: how much does the split protocol inflate macro-F1?

Protocols (see ``data.splits``): ``random`` (frame-level), ``official`` (as published,
7 of 43 videos in both folds) and ``video`` (strictly patient-level). Every pair of
protocols with finished runs is compared; ``random - official`` is the pre-registered
contrast (H1), ``official - video`` isolates the leakage left inside the official split.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from capsule_xai_audit.eval.bootstrap import Interval, bootstrap_metrics, difference
from capsule_xai_audit.eval.metrics import METRICS, macro_f1_over
from capsule_xai_audit.experiments.runs import RunGroup, load_split_predictions
from capsule_xai_audit.utils.io import write_json
from capsule_xai_audit.utils.logging import get_logger

log = get_logger(__name__)

COMMON = "macro_f1_common_classes"
Q1_METRICS = ("macro_f1", COMMON, "balanced_accuracy", "accuracy")
# Ordered from most to least leaky: a difference is always "leakier minus stricter".
PROTOCOLS = ("random", "official", "video")


def mean_sd(values: list[float]) -> list[float]:
    return [float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0]


def leakage_analysis(
    runs_root: Path, backbone: str, test_fold: int, tag: str = "base", n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    loaded = {}
    for protocol in PROTOCOLS:
        try:
            loaded[protocol] = load_split_predictions(
                RunGroup(protocol, backbone, tag, test_fold), runs_root, "test"
            )
        except FileNotFoundError:
            log.info("Q1: no finished %s runs for %s; skipping that protocol", protocol, backbone)
    if len(loaded) < 2:
        raise FileNotFoundError(f"Q1 needs finished runs of at least two protocols, found {list(loaded)}")
    # Macro-F1 averages over the classes a test set contains; the video protocol's test fold
    # lacks some classes, so differences are also reported over the classes all protocols share.
    common = sorted(
        set.intersection(*(set(np.unique(next(iter(p.values())).labels).tolist()) for p in loaded.values()))
    )
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        m: METRICS[m] for m in Q1_METRICS if m != COMMON
    }
    metrics[COMMON] = macro_f1_over(common)

    intervals: dict[str, dict[str, Interval]] = {}
    per_seed: dict[str, dict[str, dict[int, float]]] = {}
    for protocol, preds in loaded.items():
        first = next(iter(preds.values()))
        probs = [p.probs() for p in preds.values()]
        intervals[protocol] = bootstrap_metrics(metrics, probs, first.labels, first.groups, n_boot, seed)
        per_seed[protocol] = {
            m: {s: fn(p.probs(), p.labels) for s, p in preds.items()} for m, fn in metrics.items()
        }

    present = [p for p in PROTOCOLS if p in intervals]
    result: dict[str, Any] = {
        "question": "Q1",
        "backbone": backbone,
        "test_fold": test_fold,
        "tag": tag,
        "protocols": present,
        "n_seeds": {p: len(next(iter(per_seed[p].values()))) for p in present},
        "bootstrap": "video-cluster bootstrap, seeds averaged within each replicate",
        "n_boot": n_boot,
        "common_classes": [first.class_names[c] for c in common],
        "metrics": {},
    }
    for m in Q1_METRICS:
        entry: dict[str, Any] = {p: intervals[p][m].as_dict() for p in present}
        entry["differences"] = {}
        for leaky, strict in combinations(present, 2):
            diff = difference(intervals[leaky][m], intervals[strict][m])
            entry["differences"][f"{leaky}_minus_{strict}"] = {
                **diff.as_dict(),
                "ci_excludes_zero": diff.excludes_zero,
            }
        entry["per_seed"] = {p: per_seed[p][m] for p in present}
        entry["seed_mean_sd"] = {p: mean_sd(list(per_seed[p][m].values())) for p in present}
        result["metrics"][m] = entry
    out = runs_root / "analysis" / f"q1_leakage_{backbone}_{tag}_fold{test_fold}.json"
    write_json(out, result)
    log.info(
        "Q1 %s fold%d macro-F1 differences: %s",
        backbone,
        test_fold,
        result["metrics"]["macro_f1"]["differences"],
    )
    return result
