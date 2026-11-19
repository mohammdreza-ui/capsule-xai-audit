"""Q3: which frozen backbone stays best calibrated with the fewest labels?"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from capsule_xai_audit.config import FewShotConfig, FewShotExperiment, PathsConfig, to_dict
from capsule_xai_audit.data.cache import open_cache
from capsule_xai_audit.data.datasets import CachedImages, make_loader
from capsule_xai_audit.eval.metrics import compute_metrics, probs_from_logits
from capsule_xai_audit.experiments.train_run import load_kvasir
from capsule_xai_audit.fewshot.probe import extract_features, fit_probe, sample_shots
from capsule_xai_audit.models.backbones import build_classifier
from capsule_xai_audit.utils.io import sha256_strings, write_json
from capsule_xai_audit.utils.logging import get_logger
from capsule_xai_audit.utils.runtime import provenance, resolve_device
from capsule_xai_audit.utils.seed import derive_seed, seed_everything

log = get_logger(__name__)

PROBE_METRICS = ("ece", "nll", "brier", "macro_f1", "balanced_accuracy", "accuracy")


def feature_path(paths: PathsConfig, backbone: str) -> Path:
    return paths.runs_root / "features" / f"{backbone}.npy"


def cached_features(
    paths: PathsConfig,
    index: pd.DataFrame,
    backbone: str,
    *,
    pretrained: bool,
    batch_size: int,
    num_workers: int,
    amp: bool,
    device: torch.device,
) -> np.ndarray:
    """Pooled frozen features for every index row, computed once and cached as float16."""
    path = feature_path(paths, backbone)
    tag_path = path.with_suffix(".sha256")
    index_sha = sha256_strings(index["image_id"])
    if path.exists() and tag_path.exists() and tag_path.read_text().strip() == index_sha:
        return np.load(path).astype(np.float32)
    img_size = open_cache(paths.kvasir_cache)[0].shape[1]
    model = build_classifier(backbone, num_classes=0, pretrained=pretrained, img_size=img_size).to(device)
    loader = make_loader(
        CachedImages(paths.kvasir_cache, np.arange(len(index))), batch_size, False, num_workers
    )
    feats, positions = extract_features(model, loader, device, amp)
    if not (positions == np.arange(len(index))).all():
        raise RuntimeError("feature extraction returned rows out of order")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feats.astype(np.float16))
    tag_path.write_text(index_sha + "\n")
    log.info("cached %s features %s", backbone, feats.shape)
    return feats


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "q025": float(np.quantile(arr, 0.025)),
        "q975": float(np.quantile(arr, 0.975)),
    }


def _evaluate_probes(
    feats: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train_pool: np.ndarray,
    test: np.ndarray,
    center: np.ndarray,
    n_classes: int,
    C: float,
    cfg: FewShotConfig,
) -> dict[str, Any]:
    """Draws per k (same supports for every backbone and C), plus the full-pool probe."""
    out: dict[str, Any] = {}
    for k in cfg.shots:
        draws = []
        for d in range(cfg.draws):
            rng = np.random.default_rng(derive_seed(cfg.seed, k, d))
            support = sample_shots(labels, groups, train_pool, k, rng)
            probe = fit_probe(feats[support], labels[support], n_classes, C, center, cfg.seed)
            probs = probs_from_logits(probe.logits(feats[test]))
            draws.append(
                {"support_size": len(support), **compute_metrics(probs, labels[test], PROBE_METRICS)}
            )
        out[str(k)] = {
            "draws": draws,
            "summary": {m: _summary([dr[m] for dr in draws]) for m in PROBE_METRICS},
        }
    full = fit_probe(feats[train_pool], labels[train_pool], n_classes, C, center, cfg.seed)
    out["all"] = {
        "summary": compute_metrics(probs_from_logits(full.logits(feats[test])), labels[test], PROBE_METRICS)
    }
    return out


def run_fewshot(exp: FewShotExperiment) -> dict[str, Any]:
    cfg, paths = exp.fewshot, exp.paths
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    index, class_names = load_kvasir(paths)
    labels, groups = index["label_idx"].to_numpy(), index["video_id"].to_numpy()
    folds = index["official_fold"].to_numpy()
    train_pool = np.flatnonzero(folds != cfg.test_fold)
    test = np.flatnonzero(folds == cfg.test_fold)

    result: dict[str, Any] = {
        "question": "Q3",
        "config": to_dict(exp),
        "provenance": provenance(),
        "test_frames": len(test),
        "backbones": {},
    }
    for backbone in cfg.backbones:
        feats = cached_features(
            paths,
            index,
            backbone,
            pretrained=cfg.pretrained,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            amp=cfg.amp,
            device=device,
        )
        center = feats[train_pool].mean(0)  # unlabelled pool: no label information
        per_k = _evaluate_probes(
            feats, labels, groups, train_pool, test, center, len(class_names), cfg.C, cfg
        )
        for k in cfg.shots:
            log.info(
                "Q3 %s k=%d C=%g: ECE %.3f, macro-F1 %.3f",
                backbone,
                k,
                cfg.C,
                per_k[str(k)]["summary"]["ece"]["mean"],
                per_k[str(k)]["summary"]["macro_f1"]["mean"],
            )
        per_k["C_sensitivity"] = {
            str(c): {
                k: v["summary"]
                for k, v in _evaluate_probes(
                    feats, labels, groups, train_pool, test, center, len(class_names), c, cfg
                ).items()
            }
            for c in cfg.C_sensitivity
        }
        result["backbones"][backbone] = per_k

    smallest = str(min(cfg.shots))
    ranking = sorted(cfg.backbones, key=lambda b: result["backbones"][b][smallest]["summary"]["ece"]["mean"])
    result["ranking_by_ece_at_smallest_k"] = {"k": int(smallest), "order": ranking}
    write_json(paths.runs_root / "analysis" / f"q3_fewshot_fold{cfg.test_fold}.json", result)
    return result
