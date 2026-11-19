"""XAI metrics, few-shot probes and both slices: unit tests plus end-to-end runs on fake data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from capsule_xai_audit.config import (
    ColorConfig,
    ColorExperiment,
    FewShotConfig,
    FewShotExperiment,
    PathsConfig,
    SuperpixelConfig,
    SuperpixelExperiment,
    TrainConfig,
    TrainExperiment,
    XaiConfig,
    XaiExperiment,
)
from capsule_xai_audit.experiments.color import run_color, stratified_subsample
from capsule_xai_audit.experiments.fewshot import run_fewshot
from capsule_xai_audit.experiments.prepare import build_external_set
from capsule_xai_audit.experiments.superpixel import run_superpixel
from capsule_xai_audit.experiments.train_run import run_training
from capsule_xai_audit.experiments.xai import run_xai
from capsule_xai_audit.fewshot.probe import fit_probe, l2_normalize, sample_shots
from capsule_xai_audit.models.backbones import build_classifier
from capsule_xai_audit.slices.superpixel import (
    area_weighted,
    overlap_matrix,
    patch_contributions,
    patch_unit_map,
    region_adjacency,
)
from capsule_xai_audit.xai.agreement import kendall_agreement
from capsule_xai_audit.xai.attribution import METHODS, attribute
from capsule_xai_audit.xai.faithfulness import auc, deletion_insertion, pixel_ranks
from capsule_xai_audit.xai.pointing import energy_in_boxes, pointing_hit, unit_pointing_hit
from capsule_xai_audit.xai.sanity import cascading_randomization, randomize_groups

from .conftest import REPO, FakeWorld

# --------------------------------------------------------------------------- pointing


def test_pointing_and_energy() -> None:
    attr = np.zeros((20, 20))
    attr[5, 5] = 1.0
    assert pointing_hit(attr, [(4, 4, 8, 8)], tolerance=0)
    assert not pointing_hit(attr, [(10, 10, 15, 15)], tolerance=0)
    assert pointing_hit(attr, [(10, 10, 15, 15)], tolerance=5)
    attr[15, 15] = 1.0
    assert energy_in_boxes(attr, [(4, 4, 8, 8)]) == pytest.approx(0.5)
    units = np.zeros((20, 20), dtype=int)
    units[10:, 10:] = 1
    assert unit_pointing_hit(units, np.array([0.1, 0.9]), [(10, 10, 20, 20)], tolerance=0)
    assert not unit_pointing_hit(units, np.array([0.9, 0.1]), [(10, 10, 20, 20)], tolerance=0)


# --------------------------------------------------------------------------- faithfulness


class QuadrantModel(nn.Module):
    """Class-1 logit is driven by the top-left quadrant only."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        evidence = 20 * (x[:, :, : h // 2, : w // 2].mean(dim=(1, 2, 3)) - 0.25)
        return torch.stack([torch.zeros_like(evidence), evidence], 1)


def test_pixel_ranks_order_by_importance() -> None:
    maps = torch.tensor([[[0.1, 0.9], [0.5, 0.3]]])
    assert pixel_ranks(maps).tolist() == [[[3, 0], [1, 2]]]


def test_auc_of_constant_curve() -> None:
    assert auc(np.full((2, 11), 0.7)) == pytest.approx([0.7, 0.7])


def test_faithful_map_beats_unfaithful_map() -> None:
    x = torch.full((1, 3, 16, 16), 0.9)
    good = torch.zeros(1, 16, 16)
    good[:, :8, :8] = 1.0
    bad = 1.0 - good
    target = torch.tensor([1])
    c_good = deletion_insertion(QuadrantModel(), x, target, good, steps=16, amp=False, baseline="zero")
    c_bad = deletion_insertion(QuadrantModel(), x, target, bad, steps=16, amp=False, baseline="zero")
    assert c_good.deletion_auc[0] < c_bad.deletion_auc[0]
    assert c_good.insertion_auc[0] > c_bad.insertion_auc[0]


# --------------------------------------------------------------------------- attribution + sanity


@pytest.fixture(scope="module")
def tiny_cnn() -> torch.nn.Module:
    torch.manual_seed(0)
    return build_classifier("resnet18", num_classes=3, pretrained=False).eval()


@pytest.mark.parametrize("method", METHODS)
def test_attribution_shapes(method: str, tiny_cnn: torch.nn.Module) -> None:
    x = torch.rand(2, 3, 64, 64)
    maps = attribute(method, tiny_cnn, x, torch.tensor([0, 2]), ig_steps=4)  # type: ignore[arg-type]
    assert maps.shape == (2, 64, 64)
    assert (maps >= 0).all()
    assert torch.isfinite(maps).all()


def test_randomization_changes_only_selected_layers(tiny_cnn: torch.nn.Module) -> None:
    import copy

    model = copy.deepcopy(tiny_cnn)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    randomize_groups(model, ("fc",), seed=0)  # type: ignore[arg-type]
    after = model.state_dict()
    assert not torch.equal(before["net.fc.weight"], after["net.fc.weight"])
    assert torch.equal(before["net.layer1.0.conv1.weight"], after["net.layer1.0.conv1.weight"])


def test_random_control_is_model_independent(tiny_cnn: torch.nn.Module) -> None:
    x = torch.rand(3, 3, 64, 64)
    t = torch.tensor([0, 1, 2])
    steps = cascading_randomization(
        tiny_cnn,  # type: ignore[arg-type]
        lambda m: attribute("random", m, x, t, seeds=[1, 2, 3]),
    )
    assert len(steps) == 6
    assert all(np.allclose(v, 1.0) for _, v in steps)
    grad_steps = cascading_randomization(
        tiny_cnn,  # type: ignore[arg-type]
        lambda m: attribute("saliency", m, x, t),
    )
    assert grad_steps[-1][1].mean() < 0.9


def test_kendall_detects_reversed_rankings() -> None:
    rng = np.random.default_rng(0)
    n = 40
    base = {"a": 0.9, "b": 0.6, "c": 0.3}
    per_image = {
        "pointing": {m: np.clip(v + rng.normal(0, 0.01, n), 0, 1) for m, v in base.items()},
        # Deletion is lower-is-better; high values for "a" mean "a" ranks last.
        "deletion_auc": {m: v + rng.normal(0, 0.01, n) for m, v in base.items()},
    }
    out = kendall_agreement(per_image, n_boot=100)
    assert out["pairs"]["pointing~deletion_auc"]["tau"] == pytest.approx(-1.0)
    assert out["rankings"]["pointing"] == ["a", "b", "c"]


# --------------------------------------------------------------------------- probes


def test_sample_shots_spreads_over_videos() -> None:
    labels = np.array([0] * 12 + [1] * 12)
    groups = np.array([f"v{i % 4}" for i in range(24)])
    shots = sample_shots(labels, groups, np.arange(24), 4, np.random.default_rng(0))
    assert len(shots) == 8
    for cls in (0, 1):
        assert len(set(groups[shots][labels[shots] == cls])) == 4
    few = sample_shots(labels, groups, np.arange(3), 5, np.random.default_rng(0))
    assert len(few) == 3


def test_binary_probe_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 5))
    y = (x[:, 0] > 0).astype(int) * 2  # classes 0 and 2 of 3
    center = x.mean(0)
    probe = fit_probe(x, y, n_classes=3, C=1.0, center=center)
    xn = l2_normalize(x - center)
    ref = LogisticRegression(C=1.0, max_iter=5000).fit(xn, y).predict_proba(xn)
    z = probe.logits(x)
    probs = np.exp(z - z.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    np.testing.assert_allclose(probs[:, [0, 2]], ref, atol=1e-6)
    assert probs[:, 1].max() < 1e-10


# --------------------------------------------------------------------------- superpixels


def test_patch_contributions_decompose_the_margin_exactly() -> None:
    rng = np.random.default_rng(0)
    patches = rng.normal(size=(16, 8))
    weight, bias, center = rng.normal(size=(4, 8)), rng.normal(size=4), rng.normal(size=8)
    feature = patches.mean(0) - center
    logits = weight @ (feature / np.linalg.norm(feature)) + bias
    for c in range(4):
        contrib = patch_contributions(patches, weight, c, center)
        margin = (logits[c] - logits.mean()) - (bias[c] - bias.mean())
        assert contrib.sum() == pytest.approx(margin)


def test_superpixel_pooling_preserves_total_contribution() -> None:
    units = patch_unit_map((4, 4), 16, 16)
    assert np.bincount(units.ravel()).tolist() == [16] * 16
    segments = (np.indices((16, 16)).sum(0) // 6).astype(np.int32)  # diagonal bands
    n_seg = int(segments.max()) + 1
    overlap = overlap_matrix(segments, units, n_seg, 16)
    contrib = np.random.default_rng(0).normal(size=16)
    scores = area_weighted(overlap, contrib)
    # sum over pixels of the piecewise-constant maps is identical
    assert (scores[segments]).sum() == pytest.approx(contrib[units].sum())


def test_region_adjacency() -> None:
    seg = np.array([[0, 0, 1], [2, 2, 1], [2, 3, 3]])
    assert region_adjacency(seg).tolist() == [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]


def test_stratified_subsample_keeps_rare_classes() -> None:
    labels = np.array([0] * 95 + [1] * 5)
    sub = stratified_subsample(labels, np.arange(100), 20, seed=0)
    assert set(labels[sub]) == {0, 1}
    assert len(sub) == 20


# --------------------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def trained(world: FakeWorld, tmp_path_factory: pytest.TempPathFactory) -> PathsConfig:
    root = tmp_path_factory.mktemp("xai_runs")
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
    paths = PathsConfig(
        kvasir_index=k_index,
        kvasir_cache=world.kvasir_cache,
        external_index=external,
        external_cache=world.cv_cache,
        class_mapping=REPO / "configs" / "class_mapping.yaml",
        runs_root=root / "runs",
    )
    for tag, de in (("base", 0.0), ("lab-aug", 10.0)):
        cfg = TrainConfig(
            backbone="resnet18",
            pretrained=False,
            epochs=1,
            batch_size=16,
            eval_batch_size=32,
            max_steps_per_epoch=2,
            num_workers=0,
            amp=False,
            device="cpu",
            val_fraction=0.25,
            tag=tag,
            color_aug_max_delta_e=de,
        )
        run_training(TrainExperiment(paths=paths, train=cfg))
    return paths


def test_fewshot_end_to_end(trained: PathsConfig) -> None:
    cfg = FewShotConfig(
        backbones=("resnet18", "dinov2_vits14"),
        shots=(1, 2),
        draws=2,
        pretrained=False,
        batch_size=32,
        num_workers=0,
        amp=False,
        device="cpu",
    )
    out = run_fewshot(FewShotExperiment(paths=trained, fewshot=cfg))
    assert set(out["backbones"]) == {"resnet18", "dinov2_vits14"}
    assert len(out["backbones"]["resnet18"]["1"]["draws"]) == 2
    assert out["ranking_by_ece_at_smallest_k"]["k"] == 1


def test_xai_end_to_end(trained: PathsConfig) -> None:
    cfg = XaiConfig(
        run="train/official/resnet18/base/fold1/seed0",
        max_images=12,
        sanity_images=4,
        ig_steps=4,
        perturbation_steps=8,
        n_bootstrap=50,
        batch_size=8,
        device="cpu",
    )
    out = run_xai(XaiExperiment(paths=trained, xai=cfg))
    assert out["n_images"] == 12
    assert set(out["metrics"]["pointing"]) == set(METHODS)
    assert "pointing~deletion_auc" in out["agreement"]["pairs"]
    random_steps = out["sanity"]["methods"]["random"]
    assert all(s["spearman_mean"] == pytest.approx(1.0) for s in random_steps)
    by = out["by_correctness"]
    assert by["correct"]["n_images"] + by["wrong"]["n_images"] == 12
    for entry in by.values():
        if "metrics" in entry:
            assert set(entry["metrics"]["pointing"]) == set(METHODS)


def test_superpixel_end_to_end(trained: PathsConfig) -> None:
    cfg = SuperpixelConfig(
        n_segments=12,
        max_images=10,
        pretrained=False,
        n_bootstrap=50,
        batch_size=8,
        num_workers=0,
        device="cpu",
    )
    out = run_superpixel(SuperpixelExperiment(paths=trained, superpixel=cfg))
    assert out["n_images"] == 10
    assert set(out["superpixel_minus_patch"]) == {"pointing", "energy"}
    graphs = list((trained.runs_root / "superpixel" / "dinov2_vits14" / "fold1" / "graphs").glob("*.npz"))
    assert len(graphs) == 10
    with np.load(graphs[0]) as g:
        assert g["edges"].shape[1] == 2
        assert g["node_features"].shape[0] == g["areas"].shape[0]


def test_color_end_to_end(trained: PathsConfig) -> None:
    cfg = ColorConfig(
        runs=("train/official/resnet18/*/fold1/seed0",),
        delta_e=(0.0, 5.0, 20.0),
        hue_angles=(0.0, 180.0),
        max_images=40,
        batch_size=32,
        num_workers=0,
        amp=False,
        device="cpu",
    )
    out = run_color(ColorExperiment(paths=trained, color=cfg))
    assert len(out["runs"]) == 2
    entry = next(iter(out["runs"].values()))
    assert [c["delta_e"] for c in entry["curve"]] == [0.0, 5.0, 20.0]
    assert entry["curve"][0]["realised_delta_e76"] == pytest.approx(0.0, abs=1e-9)
    assert 0 < entry["curve"][1]["realised_delta_e76"] <= 5.0 + 1e-6
    assert "gap_recovered_fraction" in entry["reinhard"]
    assert "lab-aug_minus_base_ece_slope" in out["by_tag"]


def test_report_collects_finished_analyses(trained: PathsConfig, tmp_path: Path) -> None:
    """Runs after the end-to-end tests above (pytest keeps definition order)."""
    from capsule_xai_audit.report import build_report

    path = build_report(trained.runs_root, tmp_path / "results")
    text = path.read_text()
    for heading in ("## Q3", "## Q4", "## Slice A", "## Slice B"):
        assert heading in text
    assert "split by whether the model classifies the frame correctly" in text
    assert "| random | n/a |" in text
    figures = {f.name for f in (tmp_path / "results" / "figures").glob("*.png")}
    assert "q4_xai_train_official_resnet18_base_fold1_seed0.png" in figures
    assert "sliceA_superpixel_dinov2_vits14_fold1.png" in figures
    assert "q3_fewshot_fold1.png" in figures
