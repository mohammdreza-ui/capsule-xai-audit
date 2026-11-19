"""Slice A: do superpixel units point better than patch units on the same frozen model?

The probe is trained on the full training fold (not few-shot); accuracy is identical for
both units by construction and reported once. Region adjacency graphs (node features,
centroids, areas, edges, scores) are saved per image as an intermediate output for
later graph-based work; they are not used by any result here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from capsule_xai_audit.config import SuperpixelExperiment, to_dict
from capsule_xai_audit.data.cache import open_cache
from capsule_xai_audit.data.datasets import to_unit_float
from capsule_xai_audit.eval.bootstrap import paired_bootstrap
from capsule_xai_audit.eval.metrics import compute_metrics, probs_from_logits
from capsule_xai_audit.experiments.fewshot import cached_features
from capsule_xai_audit.experiments.train_run import load_kvasir
from capsule_xai_audit.experiments.xai import select_boxed_images
from capsule_xai_audit.fewshot.probe import fit_probe
from capsule_xai_audit.models.backbones import build_classifier
from capsule_xai_audit.slices.superpixel import (
    area_weighted,
    overlap_matrix,
    patch_contributions,
    patch_unit_map,
    region_adjacency,
    segment_centroids,
    slic_segments,
)
from capsule_xai_audit.utils.io import write_csv, write_json
from capsule_xai_audit.utils.logging import get_logger
from capsule_xai_audit.utils.runtime import provenance, resolve_device
from capsule_xai_audit.utils.seed import seed_everything
from capsule_xai_audit.xai.pointing import energy_in_boxes, unit_pointing_hit

log = get_logger(__name__)


def run_superpixel(exp: SuperpixelExperiment) -> dict[str, Any]:
    cfg, paths = exp.superpixel, exp.paths
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    index, class_names = load_kvasir(paths)
    labels = index["label_idx"].to_numpy()
    folds = index["official_fold"].to_numpy()
    train, test = np.flatnonzero(folds != cfg.test_fold), np.flatnonzero(folds == cfg.test_fold)

    feats = cached_features(
        paths,
        index,
        cfg.backbone,
        pretrained=cfg.pretrained,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        amp=True,
        device=device,
    )
    probe = fit_probe(feats[train], labels[train], len(class_names), cfg.C, feats[train].mean(0), cfg.seed)
    probe_metrics = compute_metrics(probs_from_logits(probe.logits(feats[test])), labels[test])

    pixels, sizes = open_cache(paths.kvasir_cache, index["image_id"].tolist())
    size = pixels.shape[1]
    model = build_classifier(cfg.backbone, 0, pretrained=cfg.pretrained, img_size=size).to(device).eval()
    if model.spec.kind != "vit":
        raise ValueError("Slice A needs a ViT backbone with patch tokens")
    sel = select_boxed_images(index, test, sizes, size, cfg.max_images, cfg.seed)
    grid = model.patch_grid
    units = patch_unit_map(grid, size, size)
    n_units = grid[0] * grid[1]
    out_dir = paths.runs_root / "superpixel" / cfg.backbone / f"fold{cfg.test_fold}"
    graph_dir = out_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for start in range(0, len(sel.positions), cfg.batch_size):
        pos = sel.positions[start : start + cfg.batch_size]
        batch_u8 = np.array(pixels[pos])
        with torch.no_grad():
            tokens = model.patch_tokens(
                to_unit_float(torch.from_numpy(batch_u8).permute(0, 3, 1, 2).to(device))
            )
        tokens_np = tokens.float().cpu().numpy().astype(np.float64)
        for j, p in enumerate(pos):
            i = start + j
            target = int(sel.labels[i])
            contrib = patch_contributions(tokens_np[j], probe.weight, target, probe.center)
            segments = slic_segments(batch_u8[j], cfg.n_segments, cfg.compactness)
            n_seg = int(segments.max()) + 1
            overlap = overlap_matrix(segments, units, n_seg, n_units)
            sp_scores = area_weighted(overlap, contrib)
            boxes = sel.boxes[i]
            rows.append(
                {
                    "image_id": sel.image_ids[i],
                    "video_id": sel.groups[i],
                    "label_idx": target,
                    "n_superpixels": n_seg,
                    "pointing_patch": unit_pointing_hit(units, contrib, boxes, cfg.pointing_tolerance_px),
                    "pointing_superpixel": unit_pointing_hit(
                        segments, sp_scores, boxes, cfg.pointing_tolerance_px
                    ),
                    "energy_patch": energy_in_boxes(contrib[units], boxes),
                    "energy_superpixel": energy_in_boxes(sp_scores[segments], boxes),
                }
            )
            if cfg.save_graphs:
                np.savez_compressed(
                    graph_dir / f"{sel.image_ids[i]}.npz",
                    segments=segments.astype(np.uint16),
                    node_features=area_weighted(overlap, tokens_np[j]).astype(np.float16),
                    node_scores=sp_scores.astype(np.float32),
                    centroids=segment_centroids(segments, n_seg).astype(np.float32),
                    areas=overlap.sum(1).astype(np.int32),
                    edges=region_adjacency(segments),
                    target=np.int64(target),
                    cache_position=np.int64(p),
                )
    table = pd.DataFrame(rows)
    write_csv(out_dir / "per_image.csv", table)

    def paired(metric: str) -> dict[str, float]:
        sp = table[f"{metric}_superpixel"].to_numpy(dtype=float)
        pa = table[f"{metric}_patch"].to_numpy(dtype=float)
        return paired_bootstrap(
            lambda r: float(sp[r].mean() - pa[r].mean()), len(table), sel.groups, cfg.n_bootstrap, cfg.seed
        ).as_dict()

    summary = {
        "question": "Slice A",
        "config": to_dict(exp),
        "provenance": provenance(),
        "probe_test_metrics": probe_metrics,
        "n_images": len(table),
        "n_videos": int(table["video_id"].nunique()),
        "mean_superpixels": float(table["n_superpixels"].mean()),
        "rates": {c: float(table[c].mean()) for c in table.columns if c.startswith(("pointing", "energy"))},
        "superpixel_minus_patch": {"pointing": paired("pointing"), "energy": paired("energy")},
    }
    write_json(out_dir / "summary.json", summary)
    write_json(
        paths.runs_root / "analysis" / f"sliceA_superpixel_{cfg.backbone}_fold{cfg.test_fold}.json", summary
    )
    log.info("Slice A: %s", summary["superpixel_minus_patch"])
    return summary
