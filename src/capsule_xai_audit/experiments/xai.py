"""Q4: faithfulness, sanity check and pointing game for attribution methods.

Images: test-fold frames that carry at least one Kvasir-Capsule bounding box, sampled
round-robin over classes (seeded). The target of every attribution is the ground-truth
class. Metrics are reported for all frames and, separately, for the frames the model
classifies correctly and wrongly: on a wrong frame the map explains a class the model
does not predict, which says little about how it finds lesions. Outputs go to
``<runs_root>/xai/<run path>/``: per-image CSV, per-method maps (float16 npz, never
committed) and ``summary.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from capsule_xai_audit.config import XaiConfig, XaiExperiment, to_dict
from capsule_xai_audit.data.cache import open_cache, transform_boxes
from capsule_xai_audit.data.datasets import to_unit_float
from capsule_xai_audit.eval.bootstrap import paired_bootstrap
from capsule_xai_audit.experiments.train_run import load_run_model
from capsule_xai_audit.models.backbones import Classifier
from capsule_xai_audit.utils.io import read_json, write_csv, write_json
from capsule_xai_audit.utils.logging import get_logger
from capsule_xai_audit.utils.runtime import provenance, resolve_device
from capsule_xai_audit.utils.seed import seed_everything
from capsule_xai_audit.xai.agreement import METRIC_DIRECTION, kendall_agreement
from capsule_xai_audit.xai.attribution import attribute
from capsule_xai_audit.xai.faithfulness import deletion_insertion
from capsule_xai_audit.xai.pointing import Box, energy_in_boxes, pointing_hit
from capsule_xai_audit.xai.sanity import cascading_randomization

log = get_logger(__name__)


@dataclass(frozen=True)
class BoxedImages:
    positions: np.ndarray  # rows of the Kvasir index / cache
    labels: np.ndarray
    groups: np.ndarray  # video ids
    image_ids: np.ndarray
    boxes: list[list[Box]]  # in cache pixel coordinates


def select_boxed_images(
    index: pd.DataFrame,
    candidates: np.ndarray,
    sizes: np.ndarray,
    cache_size: int,
    max_images: int,
    seed: int,
) -> BoxedImages:
    """Up to ``max_images`` candidates with a box, balanced over classes round-robin."""
    boxes_all = [
        transform_boxes(json.loads(index["bboxes"].iloc[p]), int(sizes[p, 0]), int(sizes[p, 1]), cache_size)
        for p in candidates
    ]
    with_box = [(int(p), b) for p, b in zip(candidates, boxes_all, strict=True) if b]
    if not with_box:
        raise ValueError("no candidate frame has a bounding box; rebuild the Kvasir index with metadata.csv")
    rng = np.random.default_rng(seed)
    labels = index["label_idx"].to_numpy()
    by_class: dict[int, list[tuple[int, list[Box]]]] = {}
    for item in with_box:
        by_class.setdefault(int(labels[item[0]]), []).append(item)
    queues = [[q[i] for i in rng.permutation(len(q))] for _, q in sorted(by_class.items())]
    picked: list[tuple[int, list[Box]]] = []
    while len(picked) < max_images and any(queues):
        for q in queues:
            if q and len(picked) < max_images:
                picked.append(q.pop())
    picked.sort()
    pos = np.array([p for p, _ in picked])
    return BoxedImages(
        positions=pos,
        labels=labels[pos],
        groups=index["video_id"].to_numpy()[pos],
        image_ids=index["image_id"].to_numpy()[pos],
        boxes=[b for _, b in picked],
    )


def mean_ci(values: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    """Mean of per-image scores with a video-cluster bootstrap interval."""

    def stat(rows: np.ndarray) -> float:
        return float(values[rows].mean())

    return paired_bootstrap(stat, len(values), groups, n_boot, seed).as_dict()


# A subset needs this many frames from this many videos for a cluster-bootstrap interval.
MIN_SUBSET_IMAGES, MIN_SUBSET_VIDEOS = 10, 2


def metrics_by_correctness(
    per_image: dict[str, dict[str, np.ndarray]],
    correct: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mask in (("correct", correct), ("wrong", ~correct)):
        n, n_videos = int(mask.sum()), len(set(groups[mask]))
        entry: dict[str, Any] = {"n_images": n, "n_videos": n_videos}
        if n >= MIN_SUBSET_IMAGES and n_videos >= MIN_SUBSET_VIDEOS:
            entry["metrics"] = {
                metric: {
                    method: mean_ci(values[mask], groups[mask], n_boot, seed)
                    for method, values in by_method.items()
                }
                for metric, by_method in per_image.items()
            }
        out[name] = entry
    return out


def _batches(n: int, size: int) -> list[slice]:
    return [slice(s, min(n, s + size)) for s in range(0, n, size)]


def run_xai(exp: XaiExperiment) -> dict[str, Any]:
    cfg, paths = exp.xai, exp.paths
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    run_dir = paths.runs_root / cfg.run
    out_dir = paths.runs_root / "xai" / cfg.run
    model = load_run_model(run_dir, device)
    for p in model.parameters():
        p.requires_grad_(False)
    methods = [m for m in cfg.methods if not (m == "gradcam" and model.spec.cam_layer is None)]

    index = pd.read_csv(paths.kvasir_index)
    pixels, sizes = open_cache(paths.kvasir_cache, index["image_id"].tolist())
    test = np.load(run_dir / "split.npz")["test"]
    sel = select_boxed_images(index, test, sizes, pixels.shape[1], cfg.max_images, cfg.seed)
    log.info("Q4: %d boxed test images, methods %s", len(sel.positions), methods)

    images = to_unit_float(torch.from_numpy(np.array(pixels[sel.positions])).permute(0, 3, 1, 2))
    targets = torch.from_numpy(sel.labels)
    with torch.no_grad():
        logits = torch.cat(
            [model(images[s].to(device)).float().cpu() for s in _batches(len(images), cfg.batch_size)]
        )
    correct = (logits.argmax(1) == targets).numpy()

    per_image: dict[str, dict[str, np.ndarray]] = {m: {} for m in METRIC_DIRECTION}
    rows = []
    for method in methods:
        maps = []
        for s in _batches(len(images), cfg.batch_size):
            seeds = [cfg.seed * 1_000_003 + int(p) for p in sel.positions[s]]
            maps.append(
                attribute(
                    method,
                    model,
                    images[s].to(device),
                    targets[s].to(device),
                    ig_steps=cfg.ig_steps,
                    seeds=seeds,
                ).cpu()
            )
        attr = torch.cat(maps)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"maps_{method}.npz", maps=attr.numpy().astype(np.float16), positions=sel.positions
        )

        curves_del, curves_ins = [], []
        for s in _batches(len(images), cfg.batch_size):
            c = deletion_insertion(
                model,
                images[s].to(device),
                targets[s].to(device),
                attr[s].to(device),
                steps=cfg.perturbation_steps,
                batch_size=cfg.batch_size * 2,
                seed=cfg.seed,
            )
            curves_del.append(c.deletion_auc)
            curves_ins.append(c.insertion_auc)
        a_np = attr.numpy()
        per_image["deletion_auc"][method] = np.concatenate(curves_del)
        per_image["insertion_auc"][method] = np.concatenate(curves_ins)
        per_image["pointing"][method] = np.array(
            [pointing_hit(a, b, cfg.pointing_tolerance_px) for a, b in zip(a_np, sel.boxes, strict=True)],
            dtype=float,
        )
        per_image["energy"][method] = np.array(
            [energy_in_boxes(a, b) for a, b in zip(a_np, sel.boxes, strict=True)]
        )
        for i in range(len(sel.positions)):
            rows.append(
                {
                    "image_id": sel.image_ids[i],
                    "video_id": sel.groups[i],
                    "label_idx": int(sel.labels[i]),
                    "correct": bool(correct[i]),
                    "method": method,
                    **{metric: float(per_image[metric][method][i]) for metric in METRIC_DIRECTION},
                }
            )
        log.info(
            "Q4 %s: pointing %.3f, deletion %.3f, insertion %.3f",
            method,
            per_image["pointing"][method].mean(),
            per_image["deletion_auc"][method].mean(),
            per_image["insertion_auc"][method].mean(),
        )
    write_csv(out_dir / "per_image.csv", pd.DataFrame(rows))

    sanity = _sanity(model, images, targets, sel, methods, cfg.sanity_images, cfg, device)
    summary = {
        "question": "Q4",
        "config": to_dict(exp),
        "provenance": provenance(),
        "model_run": read_json(run_dir / "run.json")["config"]["train"],
        "n_images": len(sel.positions),
        "n_videos": len(set(sel.groups)),
        "accuracy_on_images": float(correct.mean()),
        "metrics": {
            metric: {
                method: mean_ci(values, sel.groups, cfg.n_bootstrap, cfg.seed)
                for method, values in per_image[metric].items()
            }
            for metric in METRIC_DIRECTION
        },
        "by_correctness": metrics_by_correctness(per_image, correct, sel.groups, cfg.n_bootstrap, cfg.seed),
        "agreement": kendall_agreement(per_image, sel.groups, cfg.n_bootstrap, cfg.seed),
        "sanity": sanity,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(paths.runs_root / "analysis" / f"q4_xai_{cfg.run.replace('/', '_')}.json", summary)
    return summary


def _sanity(
    model: Classifier,
    images: torch.Tensor,
    targets: torch.Tensor,
    sel: BoxedImages,
    methods: list[str],
    n: int,
    cfg: XaiConfig,
    device: torch.device,
) -> dict[str, Any]:
    x, t = images[:n].to(device), targets[:n].to(device)
    seeds = [cfg.seed * 1_000_003 + int(p) for p in sel.positions[:n]]
    out: dict[str, Any] = {"n_images": len(x), "methods": {}}
    for method in methods:

        def fn(m: Classifier, method: str = method) -> torch.Tensor:
            return torch.cat(
                [
                    attribute(method, m, x[s], t[s], ig_steps=cfg.ig_steps, seeds=seeds[s])
                    for s in _batches(len(x), cfg.batch_size)
                ]
            )

        steps = cascading_randomization(model, fn, seed=cfg.seed)
        out["methods"][method] = [
            {"randomized_through": name, "spearman_mean": float(np.mean(v)), "spearman_sd": float(np.std(v))}
            for name, v in steps
        ]
    return out
