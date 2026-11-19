"""End-to-end training and Q1/Q2 analyses on the synthetic datasets (CPU, tiny model)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from capsule_xai_audit.config import PathsConfig, TrainConfig, TrainExperiment
from capsule_xai_audit.data.datasets import GeometricAugment
from capsule_xai_audit.eval.predict import Predictions
from capsule_xai_audit.experiments.calibration import calibration_analysis
from capsule_xai_audit.experiments.leakage import leakage_analysis
from capsule_xai_audit.experiments.prepare import build_external_set
from capsule_xai_audit.experiments.train_run import run_training
from capsule_xai_audit.training.trainer import class_weights, warmup_cosine
from capsule_xai_audit.utils.io import read_json

from .conftest import REPO, FakeWorld


@pytest.fixture(scope="module")
def paths(world: FakeWorld, tmp_path_factory: pytest.TempPathFactory) -> PathsConfig:
    root = tmp_path_factory.mktemp("runs")
    k_index, cv_index = root / "kvasir.csv", root / "cv.csv"
    world.kvasir_index.to_csv(k_index, index=False)
    world.cv_index.to_csv(cv_index, index=False)
    external = root / "external.csv"
    build_external_set(
        k_index,
        world.kvasir_cache,
        cv_index,
        world.cv_cache,
        world.mapping,
        external,
        root / "manifests",
        max_hamming=12,
        min_ncc=0.9,
    )
    return PathsConfig(
        kvasir_index=k_index,
        kvasir_cache=world.kvasir_cache,
        external_index=external,
        external_cache=world.cv_cache,
        class_mapping=REPO / "configs" / "class_mapping.yaml",
        runs_root=root / "runs",
    )


def _cfg(**kw: object) -> TrainConfig:
    base: dict[str, object] = {
        "backbone": "resnet18",
        "pretrained": False,
        "epochs": 2,
        "batch_size": 16,
        "eval_batch_size": 32,
        "max_steps_per_epoch": 3,
        "num_workers": 0,
        "amp": False,
        "device": "cpu",
        "val_fraction": 0.25,
    }
    base.update(kw)
    return TrainConfig(**base)  # type: ignore[arg-type]


def test_external_set_excludes_kvasir_frames_and_two_label_images(
    paths: PathsConfig, world: FakeWorld
) -> None:
    assert paths.external_index is not None
    external = pd.read_csv(paths.external_index)
    ids = set(external["image_id"])
    # Removed by source folder (including the copy pHash cannot match) and by image match.
    assert not ids & set(world.paths.cv_kvasir_source)
    assert not ids & set(world.paths.cv_multilabel)
    assert set(external["source"]) <= {"KID", "SEE-AI"}
    assert set(external["label"]) <= set(world.mapping.shared.values())
    manifests = paths.runs_root.parent / "manifests"
    dups = pd.read_csv(manifests / "cv2024_kvasir_duplicates.csv")
    assert set(dups["cv_image_id"]) == set(world.paths.planted_duplicates)
    multi = pd.read_csv(manifests / "cv2024_multilabel_excluded.csv")
    assert set(multi["image_id"]) == set(world.paths.cv_multilabel)
    summary = read_json(manifests / "cv2024_dedup_summary.json")
    assert summary["kvasir_by_both"] == len(world.paths.planted_duplicates)
    assert summary["kvasir_source_only"] == 1
    assert summary["kvasir_image_match_only"] == 0


def test_training_run_writes_predictions_and_is_idempotent(paths: PathsConfig) -> None:
    run_dir = run_training(TrainExperiment(paths=paths, train=_cfg(seed=0)))
    meta = read_json(run_dir / "run.json")
    assert meta["complete"]
    assert (run_dir / "best.pt").exists()
    assert not (run_dir / "last.pt").exists()
    test = Predictions.load(run_dir / "preds_test.npz")
    assert test.logits.shape == (meta["split_sizes"]["test"], 11)
    assert np.isfinite(test.logits).all()
    external = Predictions.load(run_dir / "preds_external.npz")
    assert len(external.labels) == len(pd.read_csv(paths.external_index))  # type: ignore[arg-type]
    # Training logs that let any curve or metric be recomputed later.
    steps = [json.loads(line) for line in (run_dir / "train_steps.jsonl").read_text().splitlines()]
    assert [r["global_step"] for r in steps] == [3, 6]  # 2 epochs x 3 steps, logged at each epoch end
    assert all(np.isfinite(r["loss"]) and r["images_per_s"] > 0 for r in steps)
    epochs = sorted((run_dir / "val_logits").glob("epoch*.npz"))
    assert [e.name for e in epochs] == ["epoch00.npz", "epoch01.npz"]
    with np.load(epochs[0]) as z:
        assert z["logits"].shape == (meta["split_sizes"]["val"], 11)
        assert len(z["labels"]) == len(z["positions"]) == meta["split_sizes"]["val"]
    mtime = (run_dir / "run.json").stat().st_mtime_ns
    run_training(TrainExperiment(paths=paths, train=_cfg(seed=0)))
    assert (run_dir / "run.json").stat().st_mtime_ns == mtime


def test_run_with_another_recipe_is_retrained(paths: PathsConfig) -> None:
    run_dir = run_training(TrainExperiment(paths=paths, train=_cfg(seed=5, epochs=1, tag="stale")))
    meta = read_json(run_dir / "run.json")
    assert meta["split_version"] == 2
    assert set(meta["split_classes"]) == {"fit", "val", "test"}
    # Same recipe with another worker count: reused.
    mtime = (run_dir / "run.json").stat().st_mtime_ns
    run_training(TrainExperiment(paths=paths, train=_cfg(seed=5, epochs=1, tag="stale", num_workers=1)))
    assert (run_dir / "run.json").stat().st_mtime_ns == mtime
    # Another loss: retrained from scratch.
    run_training(
        TrainExperiment(paths=paths, train=_cfg(seed=5, epochs=1, tag="stale", class_weighting="none"))
    )
    meta = read_json(run_dir / "run.json")
    assert meta["config"]["train"]["class_weighting"] == "none"
    assert meta["complete"]
    assert len((run_dir / "history.jsonl").read_text().splitlines()) == 1


def test_class_weights() -> None:
    w = class_weights(np.array([0] * 100 + [1] * 4 + [2]), 4, "inv_sqrt")
    assert w is not None
    counts = np.array([100, 4, 1, 0])
    assert float((w.numpy() * counts).sum()) == pytest.approx(counts.sum())  # mean 1 per frame
    assert w[1] / w[0] == pytest.approx(5.0)
    assert w[3] == 0
    assert class_weights(np.array([0, 1]), 2, "none") is None


def test_resume_continues_from_last_checkpoint(paths: PathsConfig) -> None:
    exp = TrainExperiment(paths=paths, train=_cfg(seed=7, epochs=1, tag="resume"))
    run_dir = run_training(exp)
    meta = read_json(run_dir / "run.json")
    assert meta["epochs_run"] == 1


def test_q1_and_q2_analyses(paths: PathsConfig, world: FakeWorld) -> None:
    for protocol in ("official", "random", "video"):
        for seed in (0, 1):
            run_training(TrainExperiment(paths=paths, train=_cfg(seed=seed, split_protocol=protocol)))
    q1 = leakage_analysis(paths.runs_root, "resnet18", test_fold=1, n_boot=50)
    assert q1["protocols"] == ["random", "official", "video"]
    assert q1["common_classes"]
    assert "macro_f1_common_classes" in q1["metrics"]
    diffs = q1["metrics"]["macro_f1"]["differences"]
    assert set(diffs) == {"random_minus_official", "random_minus_video", "official_minus_video"}
    for d in diffs.values():
        assert d["low"] <= d["point"] <= d["high"]
    q2 = calibration_analysis(paths.runs_root, world.mapping, "resnet18", test_fold=1, n_boot=50)
    assert set(q2["domains"]) == {"id_all", "id_shared", "external"}
    for cond in ("single", "single_ts", "ensemble", "ensemble_ts"):
        ece = q2["domains"]["external"]["conditions"][cond]["ece"]
        assert 0 <= ece["point"] <= 1
    assert all(t > 0 for t in q2["temperatures"].values())


def test_warmup_cosine_schedule() -> None:
    f = warmup_cosine(total_steps=100, warmup_steps=10)
    assert f(0) == pytest.approx(0.1)
    assert f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0)
    assert f(100) == pytest.approx(0.0, abs=1e-9)


def test_geometric_augment_preserves_content() -> None:
    x = torch.rand(8, 3, 16, 16)
    out = GeometricAugment()(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out.sum(dim=(1, 2, 3)), x.sum(dim=(1, 2, 3)))


def test_run_dir_layout(paths: PathsConfig) -> None:
    from capsule_xai_audit.experiments.train_run import run_dir_for

    assert run_dir_for(paths, _cfg(seed=3, tag="lab-aug")) == Path(
        paths.runs_root, "train", "official", "resnet18", "lab-aug", "fold1", "seed3"
    )


def test_report_covers_q1_and_q2(paths: PathsConfig, tmp_path: Path) -> None:
    """Runs after test_q1_and_q2_analyses (pytest keeps definition order)."""
    from capsule_xai_audit.report import build_report

    out = tmp_path / "results"
    text = build_report(paths.runs_root, out).read_text()
    for heading in ("## Training", "## Q1", "## Q2"):
        assert heading in text
    assert "classes present in every protocol's test set" in text
    runs = pd.read_csv(out / "training_runs.csv")
    assert {"official/resnet18/base/fold1", "random/resnet18/base/fold1", "video/resnet18/base/fold1"} <= set(
        runs["group"]
    )
    for figure in ("training_resnet18.png", "q1_macro_f1.png"):
        assert (out / "figures" / figure).stat().st_size > 0
