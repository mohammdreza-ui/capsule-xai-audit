"""One training run: split, fine-tune, then save predictions on val / test / external.

Run directory: ``<runs_root>/train/<protocol>/<backbone>/<tag>/fold<test_fold>/seed<seed>``.
A run with ``run.json["complete"]`` is skipped, so a Kaggle notebook can simply rerun the
same command list after a session restart.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from capsule_xai_audit.config import PathsConfig, TrainConfig, TrainExperiment, to_dict
from capsule_xai_audit.data.cache import open_cache
from capsule_xai_audit.data.class_mapping import load_class_mapping
from capsule_xai_audit.data.datasets import CachedImages, make_loader
from capsule_xai_audit.data.splits import SPLIT_VERSION, class_video_counts, make_split
from capsule_xai_audit.eval.predict import Predictions, predict_logits
from capsule_xai_audit.models.backbones import Classifier, build_classifier
from capsule_xai_audit.training.trainer import Trainer
from capsule_xai_audit.utils.io import read_json, sha256_strings, write_json
from capsule_xai_audit.utils.logging import get_logger
from capsule_xai_audit.utils.runtime import provenance, resolve_device
from capsule_xai_audit.utils.seed import seed_everything

log = get_logger(__name__)

PREDICTION_SPLITS = ("val", "test", "external")


# Settings that change only speed or logging, not the trained model.
_RUNTIME_ONLY = frozenset({"device", "num_workers", "eval_batch_size", "log_every_steps", "save_val_logits"})


def training_identity(train: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in train.items() if k not in _RUNTIME_ONLY}


def is_current(meta: dict[str, Any], exp: TrainExperiment) -> bool:
    """Was this run.json written by the current split algorithm and training config?"""
    if meta.get("split_version") != SPLIT_VERSION:
        return False
    return training_identity(meta["config"]["train"]) == training_identity(to_dict(exp)["train"])


def run_dir_for(paths: PathsConfig, cfg: TrainConfig) -> Path:
    return (
        paths.runs_root
        / "train"
        / cfg.split_protocol
        / cfg.backbone
        / cfg.tag
        / f"fold{cfg.test_fold}"
        / f"seed{cfg.seed}"
    )


def load_kvasir(paths: PathsConfig) -> tuple[pd.DataFrame, tuple[str, ...]]:
    mapping = load_class_mapping(paths.class_mapping)
    index = pd.read_csv(paths.kvasir_index)
    expected = np.array(mapping.kvasir_labels)[index["label_idx"].to_numpy()]
    if not (expected == index["label"].to_numpy()).all():
        raise ValueError(f"{paths.kvasir_index}: label_idx does not follow configs/class_mapping.yaml order")
    open_cache(paths.kvasir_cache, index["image_id"].tolist())  # validates the cache against the index
    return index, mapping.kvasir_labels


def load_external(paths: PathsConfig) -> pd.DataFrame | None:
    if paths.external_index is None or paths.external_cache is None or not paths.external_index.exists():
        return None
    return pd.read_csv(paths.external_index)


def load_run_model(run_dir: Path, device: torch.device) -> Classifier:
    """Rebuild a trained classifier from ``run.json`` and ``best.pt``."""
    meta = read_json(run_dir / "run.json")
    cfg = meta["config"]["train"]
    # channels_last is a memory layout only, so it is used for inference whatever training used.
    model = build_classifier(
        cfg["backbone"],
        len(meta["class_names"]),
        pretrained=False,
        img_size=meta["img_size"],
        channels_last=True,
    )
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def _predict_split(
    model: Classifier,
    cache: Path,
    positions: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    groups: np.ndarray,
    class_names: tuple[str, ...],
    cfg: TrainConfig,
    device: torch.device,
) -> Predictions:
    loader = make_loader(CachedImages(cache, positions, labels), cfg.eval_batch_size, False, cfg.num_workers)
    logits, y, pos = predict_logits(model, loader, device, cfg.amp)
    if not ((pos == positions).all() and (y == labels).all()):
        raise RuntimeError("loader returned samples out of order")
    return Predictions(logits, y, pos, image_ids, groups, class_names)


def run_training(exp: TrainExperiment) -> Path:
    cfg, paths = exp.train, exp.paths
    run_dir = run_dir_for(paths, cfg)
    meta_path = run_dir / "run.json"
    if meta_path.exists():
        old = read_json(meta_path)
        if old.get("complete") and is_current(old, exp):
            log.info("skipping completed run %s", run_dir)
            return run_dir
        if old.get("complete") or not is_current(old, exp):
            # Finished with another recipe, or partial with another config: never mix the two.
            log.warning("discarding %s: it was trained with a different split or config", run_dir)
            shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    index, class_names = load_kvasir(paths)
    split = make_split(index, cfg.split_protocol, cfg.test_fold, cfg.split_seed, cfg.val_fraction)
    np.savez(run_dir / "split.npz", fit=split.fit, val=split.val, test=split.test)
    labels = index["label_idx"].to_numpy()
    img_size = int(open_cache(paths.kvasir_cache)[0].shape[1])
    meta = {
        "config": to_dict(exp),
        "class_names": list(class_names),
        "img_size": img_size,
        "kvasir_index_sha256": sha256_strings(index["image_id"]),
        "split_version": SPLIT_VERSION,
        "split_sizes": {"fit": len(split.fit), "val": len(split.val), "test": len(split.test)},
        "split_classes": class_video_counts(index, split),
        "provenance": provenance(),
        "complete": False,
    }
    write_json(meta_path, meta)

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    model = build_classifier(
        cfg.backbone, len(class_names), cfg.pretrained, img_size=img_size, channels_last=cfg.channels_last
    )
    trainer = Trainer(
        model,
        CachedImages(paths.kvasir_cache, split.fit, labels[split.fit]),
        CachedImages(paths.kvasir_cache, split.val, labels[split.val]),
        cfg,
        run_dir,
        device,
        num_classes=len(class_names),
    )
    start = time.perf_counter()
    state = trainer.fit()
    trainer.load_best()

    ids, videos = index["image_id"].to_numpy(), index["video_id"].to_numpy()
    for name, pos in (("val", split.val), ("test", split.test)):
        preds = _predict_split(
            model,
            paths.kvasir_cache,
            pos,
            labels[pos],
            ids[pos],
            videos[pos],
            class_names,
            cfg,
            device,
        )
        preds.save(run_dir / f"preds_{name}.npz")

    external = load_external(paths)
    if external is not None and paths.external_cache is not None:
        preds = _predict_split(
            model,
            paths.external_cache,
            external["cache_pos"].to_numpy(),
            external["label_idx"].to_numpy(),
            external["image_id"].to_numpy(),
            external["image_id"].to_numpy(),
            class_names,
            cfg,
            device,
        )
        preds.save(run_dir / "preds_external.npz")
    else:
        log.warning("no external index configured/found; skipping external predictions")

    meta.update(
        complete=True,
        best_epoch=state.best_epoch,
        epochs_run=state.epoch,
        stopped_early=state.stopped_early,
        train_seconds=time.perf_counter() - start,
    )
    write_json(meta_path, meta)
    # The optimizer state is only needed to resume; drop it to save Kaggle disk.
    trainer.last_path.unlink(missing_ok=True)
    return run_dir
