"""Q2: calibration of single models, temperature scaling and deep ensembles under shift.

Domains
    id_all     Kvasir-Capsule official test fold, all 11 classes.
    id_shared  Same frames restricted to the 8 classes shared with Capsule Vision,
               probabilities renormalised over them.
    external   Capsule Vision 2024 minus Kvasir-Capsule frames, same 8 classes.
``id_shared`` vs ``external`` is the like-for-like comparison for H2b.

Conditions
    single, single_ts      each seed alone, without / with its own temperature
    ensemble, ensemble_ts  mean probability of all seeds; temperature fitted on the pooled
                           ensemble ("pool-then-calibrate")
Temperatures are fitted on the in-distribution validation split, all classes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from capsule_xai_audit.calibration.ensemble import ensemble_log_probs
from capsule_xai_audit.calibration.temperature import fit_temperature
from capsule_xai_audit.data.class_mapping import ClassMapping, restrict_to_classes
from capsule_xai_audit.eval.bootstrap import Interval, bootstrap_metrics, difference, paired_bootstrap
from capsule_xai_audit.eval.metrics import METRICS, probs_from_logits, reliability_bins
from capsule_xai_audit.eval.predict import Predictions
from capsule_xai_audit.experiments.runs import RunGroup, load_split_predictions
from capsule_xai_audit.utils.io import write_json
from capsule_xai_audit.utils.logging import get_logger

log = get_logger(__name__)

Q2_METRICS = ("ece", "nll", "brier", "classwise_ece", "macro_f1", "accuracy")
CONDITIONS = ("single", "single_ts", "ensemble", "ensemble_ts")


def _domain_logits(preds: Predictions, keep: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if keep is None:
        return preds.logits, preds.labels, preds.groups
    logits, labels, mask = restrict_to_classes(preds.logits, preds.labels, keep)
    return logits, labels, preds.groups[mask]


def condition_probs(
    member_logits: list[np.ndarray], temperatures: list[float], ensemble_temperature: float
) -> dict[str, list[np.ndarray]]:
    """Probability matrices per condition (a list of one per seed, or one for ensembles)."""
    return {
        "single": [probs_from_logits(z) for z in member_logits],
        "single_ts": [probs_from_logits(z, t) for z, t in zip(member_logits, temperatures, strict=True)],
        "ensemble": [np.exp(ensemble_log_probs(member_logits))],
        "ensemble_ts": [probs_from_logits(ensemble_log_probs(member_logits), ensemble_temperature)],
    }


def calibration_analysis(
    runs_root: Path,
    mapping: ClassMapping,
    backbone: str,
    test_fold: int,
    tag: str = "base",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    group = RunGroup("official", backbone, tag, test_fold)
    val = load_split_predictions(group, runs_root, "val")
    test = load_split_predictions(group, runs_root, "test")
    try:
        external: dict[int, Predictions] | None = load_split_predictions(group, runs_root, "external")
    except FileNotFoundError:
        external = None
        log.warning("no external predictions for %s; reporting in-distribution only", group.label())
    seeds = sorted(set(val) & set(test) & (set(external) if external else set(val)))
    if len(seeds) < 2:
        raise ValueError(f"{group.label()}: an ensemble needs at least 2 seeds, found {seeds}")

    temperatures = [fit_temperature(val[s].logits, val[s].labels) for s in seeds]
    val_labels = val[seeds[0]].labels
    t_ens = fit_temperature(ensemble_log_probs([val[s].logits for s in seeds]), val_labels)

    keep = mapping.shared_kvasir_indices
    domains: dict[str, tuple[dict[int, Predictions], np.ndarray | None]] = {
        "id_all": (test, None),
        "id_shared": (test, keep),
    }
    if external is not None:
        domains["external"] = (external, keep)

    metrics = {m: METRICS[m] for m in Q2_METRICS}
    result: dict[str, Any] = {
        "question": "Q2",
        "group": group.label(),
        "seeds": seeds,
        "temperatures": dict(zip(map(str, seeds), temperatures, strict=True)),
        "ensemble_temperature": t_ens,
        "shared_classes": list(mapping.shared_kvasir_labels),
        "n_boot": n_boot,
        "domains": {},
    }
    intervals: dict[str, dict[str, dict[str, Interval]]] = {}
    for name, (preds, cols) in domains.items():
        parts = [_domain_logits(preds[s], cols) for s in seeds]
        member_logits = [p[0] for p in parts]
        labels, groups = parts[0][1], parts[0][2]
        by_condition = condition_probs(member_logits, temperatures, t_ens)
        intervals[name] = {
            cond: bootstrap_metrics(metrics, probs, labels, groups, n_boot, seed)
            for cond, probs in by_condition.items()
        }
        paired = _ensemble_vs_single(by_condition, labels, groups, n_boot, seed)
        result["domains"][name] = {
            "n_frames": len(labels),
            "conditions": {
                cond: {m: iv.as_dict() for m, iv in ivs.items()} for cond, ivs in intervals[name].items()
            },
            "ensemble_minus_single": {m: iv.as_dict() for m, iv in paired.items()},
            "reliability": {
                cond: _reliability_dict(by_condition[cond][0], labels) for cond in ("single", "ensemble_ts")
            },
        }

    if "external" in intervals:
        # H2b: after temperature scaling, external ECE stays above in-distribution ECE.
        result["h2b_external_minus_id_shared_ece"] = {
            cond: difference(
                intervals["external"][cond]["ece"], intervals["id_shared"][cond]["ece"]
            ).as_dict()
            for cond in ("single_ts", "ensemble_ts")
        }
    write_json(runs_root / "analysis" / f"q2_calibration_{backbone}_{tag}_fold{test_fold}.json", result)
    return result


def _ensemble_vs_single(
    by_condition: dict[str, list[np.ndarray]],
    labels: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Interval]:
    """Paired (same frames) CI of ensemble minus mean single model, for H2a."""
    singles, ens = by_condition["single"], by_condition["ensemble"][0]

    def stat_for(fn: Callable[[np.ndarray, np.ndarray], float]) -> Callable[[np.ndarray], float]:
        def stat(rows: np.ndarray) -> float:
            y = labels[rows]
            return fn(ens[rows], y) - float(np.mean([fn(p[rows], y) for p in singles]))

        return stat

    return {
        m: paired_bootstrap(stat_for(METRICS[m]), len(labels), groups, n_boot, seed)
        for m in ("ece", "nll", "brier")
    }


def _reliability_dict(probs: np.ndarray, labels: np.ndarray) -> dict[str, list[float]]:
    bins = reliability_bins(probs, labels)
    return {
        "edges": bins.edges.tolist(),
        "confidence": np.nan_to_num(bins.confidence, nan=-1).tolist(),
        "accuracy": np.nan_to_num(bins.accuracy, nan=-1).tolist(),
        "count": bins.count.tolist(),
    }
