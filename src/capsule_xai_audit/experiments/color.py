"""Slice B: how much of the domain gap is colour, and does Lab augmentation help?

For every trained model matched by ``color.runs``:
1. Evaluate a fixed, class-stratified subsample of its test fold under controlled CIELAB
   casts (each ΔE in ``delta_e`` x each hue in ``hue_angles``); metrics are averaged over
   hues. The realised ΔE76 / ΔE2000 after gamut clipping is measured on the same images.
2. Fit OLS slopes of ECE (and other metrics) against nominal ΔE; report the Spearman
   correlation as a monotonicity check (HB1).
3. Re-colour the external set to the Kvasir-Capsule Lab statistics (Reinhard transfer)
   and report which share of the in-distribution -> external gap that recovers.
Models are then grouped by ``tag`` (e.g. ``base`` vs ``lab-aug``) to compare slopes (HB2).
This is a simulation: no psychophysical validation, no display calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from skimage.color import deltaE_ciede2000, rgb2lab

from capsule_xai_audit.config import ColorExperiment, PathsConfig, to_dict
from capsule_xai_audit.data.class_mapping import load_class_mapping, restrict_to_classes
from capsule_xai_audit.data.datasets import CachedImages, make_loader, to_unit_float
from capsule_xai_audit.eval.metrics import compute_metrics, probs_from_logits
from capsule_xai_audit.eval.predict import BatchTransform, predict_logits
from capsule_xai_audit.experiments.train_run import load_external, load_run_model
from capsule_xai_audit.slices.color import (
    ColorShift,
    LabStats,
    LabStatsAccumulator,
    ReinhardTransfer,
    delta_e76,
    rgb_to_lab,
)
from capsule_xai_audit.utils.io import read_json, write_json
from capsule_xai_audit.utils.logging import get_logger
from capsule_xai_audit.utils.runtime import provenance, resolve_device
from capsule_xai_audit.utils.seed import seed_everything

log = get_logger(__name__)

SHIFT_METRICS = ("ece", "nll", "brier", "accuracy", "macro_f1")
GAP_METRICS = ("macro_f1", "ece", "nll", "accuracy")


def match_runs(runs_root: Path, patterns: tuple[str, ...]) -> list[Path]:
    """Completed run directories (relative glob patterns under ``runs_root``)."""
    found = []
    for pattern in patterns:
        for d in sorted(runs_root.glob(pattern)):
            meta = d / "run.json"
            if meta.exists() and read_json(meta).get("complete") and (d / "best.pt").exists():
                found.append(d)
    if not found:
        raise FileNotFoundError(f"no completed runs match {patterns} under {runs_root}")
    return sorted(set(found))


def stratified_subsample(labels: np.ndarray, positions: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Up to ``n`` positions keeping class proportions (every present class kept at least once)."""
    if len(positions) <= n:
        return np.sort(positions)
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(labels[positions]):
        members = positions[labels[positions] == cls]
        k = max(1, round(n * len(members) / len(positions)))
        chosen.append(rng.choice(members, size=min(k, len(members)), replace=False))
    return np.sort(np.concatenate(chosen))


def realised_delta_e(images: torch.Tensor, shift: ColorShift, n_2000: int = 16) -> dict[str, float]:
    """Mean realised ΔE76 (all images) and ΔE2000 (first ``n_2000`` images) after clipping."""
    shifted = shift(images.double())
    de76 = delta_e76(rgb_to_lab(images.double()), rgb_to_lab(shifted)).mean().item()
    a = images[:n_2000].permute(0, 2, 3, 1).double().numpy()
    b = shifted[:n_2000].permute(0, 2, 3, 1).numpy()
    de2000 = float(
        np.mean([deltaE_ciede2000(rgb2lab(x), rgb2lab(y)).mean() for x, y in zip(a, b, strict=True)])
    )
    return {"delta_e76": float(de76), "delta_e2000": de2000}


def lab_stats(
    cache: Path, positions: np.ndarray, batch_size: int, num_workers: int, device: torch.device
) -> LabStats:
    acc = LabStatsAccumulator()
    for images, _, _ in make_loader(CachedImages(cache, positions), batch_size, False, num_workers):
        acc.update(to_unit_float(images.to(device)))
    return acc.result()


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def _evaluate(
    model: torch.nn.Module,
    cache: Path,
    positions: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool,
    transform: BatchTransform | None = None,
) -> np.ndarray:
    loader = make_loader(CachedImages(cache, positions, labels), batch_size, False, num_workers)
    logits, _, _ = predict_logits(model, loader, device, amp, transform)
    return logits


def run_color(exp: ColorExperiment) -> dict[str, Any]:
    cfg, paths = exp.color, exp.paths
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    mapping = load_class_mapping(paths.class_mapping)
    keep = mapping.shared_kvasir_indices
    index = pd.read_csv(paths.kvasir_index)
    labels = index["label_idx"].to_numpy()
    external = load_external(paths)
    run_dirs = match_runs(paths.runs_root, cfg.runs)

    per_run: dict[str, Any] = {}
    realised_cache: dict[tuple[float, float], dict[str, float]] = {}
    stats_cache: dict[str, LabStats] = {}
    for run_dir in run_dirs:
        rel = str(run_dir.relative_to(paths.runs_root))
        meta = read_json(run_dir / "run.json")
        split = np.load(run_dir / "split.npz")
        test = stratified_subsample(labels, split["test"], cfg.max_images, cfg.seed)
        model = load_run_model(run_dir, device)
        args = (cfg.batch_size, cfg.num_workers, device, cfg.amp)

        curve = []
        for de in cfg.delta_e:
            hue_metrics = []
            for hue in cfg.hue_angles if de > 0 else (0.0,):
                shift = ColorShift(de, hue)
                logits = _evaluate(model, paths.kvasir_cache, test, labels[test], *args, transform=shift)
                hue_metrics.append(compute_metrics(probs_from_logits(logits), labels[test], SHIFT_METRICS))
                if (de, hue) not in realised_cache:
                    sample = to_unit_float(
                        next(iter(make_loader(CachedImages(paths.kvasir_cache, test[:64]), 64, False, 0)))[0]
                    )
                    realised_cache[(de, hue)] = realised_delta_e(sample, shift)
            mean = {m: float(np.mean([h[m] for h in hue_metrics])) for m in SHIFT_METRICS}
            realised = [realised_cache[(de, h)] for h in (cfg.hue_angles if de > 0 else (0.0,))]
            curve.append(
                {
                    "delta_e": de,
                    **mean,
                    "per_hue": hue_metrics,
                    "realised_delta_e76": float(np.mean([r["delta_e76"] for r in realised])),
                    "realised_delta_e2000": float(np.mean([r["delta_e2000"] for r in realised])),
                }
            )
        x = np.array([c["delta_e"] for c in curve])
        entry: dict[str, Any] = {
            "tag": meta["config"]["train"]["tag"],
            "seed": meta["config"]["train"]["seed"],
            "n_images": len(test),
            "curve": curve,
            "slopes": {m: ols_slope(x, np.array([c[m] for c in curve])) for m in SHIFT_METRICS},
            "spearman_ece_vs_delta_e": float(spearmanr(x, [c["ece"] for c in curve]).statistic),
        }

        if external is not None and paths.external_cache is not None:
            entry["reinhard"] = _reinhard_gap(
                model, paths, split["fit"], test, labels, external, keep, stats_cache, cfg.seed, args
            )
        per_run[rel] = entry
        log.info("Slice B %s: ECE slope %.5f per ΔE", rel, entry["slopes"]["ece"])

    result = {
        "question": "Slice B",
        "config": to_dict(exp),
        "provenance": provenance(),
        "runs": per_run,
        "by_tag": _compare_tags(per_run, cfg.seed),
        "note": "Simulation only: no psychophysical validation and no display calibration.",
    }
    write_json(paths.runs_root / "analysis" / "sliceB_color.json", result)
    return result


def _reinhard_gap(
    model: torch.nn.Module,
    paths: PathsConfig,
    fit_positions: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    external: pd.DataFrame,
    keep: np.ndarray,
    stats_cache: dict[str, LabStats],
    seed: int,
    args: tuple[int, int, torch.device, bool],
) -> dict[str, Any]:
    """Share of the ID -> external gap recovered by matching external Lab statistics to Kvasir."""
    if paths.external_cache is None:
        raise ValueError("paths.external_cache is not set")
    batch_size, num_workers, device, _ = args
    ext_pos = external["cache_pos"].to_numpy()
    ext_labels = external["label_idx"].to_numpy()
    key = f"{len(fit_positions)}:{int(fit_positions.sum())}"
    if key not in stats_cache:
        rng = np.random.default_rng(seed)
        src = np.sort(rng.choice(fit_positions, size=min(2000, len(fit_positions)), replace=False))
        stats_cache[key] = lab_stats(paths.kvasir_cache, src, batch_size, num_workers, device)
    if "external" not in stats_cache:
        rng = np.random.default_rng(seed)
        ext_sample = np.sort(rng.choice(ext_pos, size=min(2000, len(ext_pos)), replace=False))
        stats_cache["external"] = lab_stats(paths.external_cache, ext_sample, batch_size, num_workers, device)
    transfer = ReinhardTransfer(source=stats_cache["external"], target=stats_cache[key])

    def shared_metrics(logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
        z, yy, _ = restrict_to_classes(logits, y, keep)
        return compute_metrics(probs_from_logits(z), yy, GAP_METRICS)

    id_m = shared_metrics(_evaluate(model, paths.kvasir_cache, test, labels[test], *args), labels[test])
    ext_m = shared_metrics(_evaluate(model, paths.external_cache, ext_pos, ext_labels, *args), ext_labels)
    tr_m = shared_metrics(
        _evaluate(model, paths.external_cache, ext_pos, ext_labels, *args, transform=transfer), ext_labels
    )
    recovered = {}
    for m in GAP_METRICS:
        gap = id_m[m] - ext_m[m]
        recovered[m] = float((tr_m[m] - ext_m[m]) / gap) if abs(gap) > 1e-9 else float("nan")
    return {
        "id_shared": id_m,
        "external": ext_m,
        "external_reinhard": tr_m,
        "gap_recovered_fraction": recovered,
        "kvasir_lab_stats": stats_cache[key].__dict__,
        "external_lab_stats": stats_cache["external"].__dict__,
    }


def _compare_tags(per_run: dict[str, Any], seed: int, n_boot: int = 5000) -> dict[str, Any]:
    """Mean ECE slope per tag across seeds, and the difference between tags with a seed bootstrap."""
    slopes: dict[str, list[float]] = {}
    for entry in per_run.values():
        slopes.setdefault(entry["tag"], []).append(entry["slopes"]["ece"])
    out: dict[str, Any] = {
        tag: {
            "n_seeds": len(v),
            "ece_slope_mean": float(np.mean(v)),
            "ece_slope_sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        }
        for tag, v in slopes.items()
    }
    if "base" in slopes and len(slopes) > 1:
        rng = np.random.default_rng(seed)
        base = np.array(slopes["base"])
        for tag, values in slopes.items():
            if tag == "base":
                continue
            other = np.array(values)
            reps = [
                rng.choice(other, len(other)).mean() - rng.choice(base, len(base)).mean()
                for _ in range(n_boot)
            ]
            low, high = np.quantile(reps, [0.025, 0.975])
            out[f"{tag}_minus_base_ece_slope"] = {
                "point": float(other.mean() - base.mean()),
                "low": float(low),
                "high": float(high),
                "note": "bootstrap over seeds; with fewer than 10 seeds report as an observation",
            }
    return out
