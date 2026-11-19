from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from capsule_xai_audit.calibration.ensemble import ensemble_log_probs, ensemble_probs
from capsule_xai_audit.calibration.temperature import fit_temperature
from capsule_xai_audit.eval.bootstrap import bootstrap_metrics, cluster_members, difference, paired_bootstrap
from capsule_xai_audit.eval.metrics import (
    balanced_accuracy,
    brier,
    classwise_ece,
    ece,
    macro_f1,
    macro_f1_over,
    nll,
    probs_from_logits,
    reliability_bins,
)


def _random_problem(n: int = 2000, k: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, k)) * 2
    labels = np.array([rng.choice(k, p=p) for p in probs_from_logits(logits)])
    return logits, labels


def test_macro_f1_and_balanced_accuracy_match_sklearn() -> None:
    logits, labels = _random_problem(k=6)
    labels[labels == 5] = 0  # one class absent from the labels but still predicted
    probs = probs_from_logits(logits)
    pred = probs.argmax(1)
    present = np.unique(labels)
    assert macro_f1(probs, labels) == pytest.approx(f1_score(labels, pred, labels=present, average="macro"))
    assert balanced_accuracy(probs, labels) == pytest.approx(balanced_accuracy_score(labels, pred))


def test_macro_f1_over_a_fixed_class_set() -> None:
    logits, labels = _random_problem(k=6)
    probs = probs_from_logits(logits)
    pred = probs.argmax(1)
    keep = [0, 2, 3]
    expected = f1_score(labels, pred, labels=keep, average="macro")
    assert macro_f1_over(keep)(probs, labels) == pytest.approx(expected)
    assert macro_f1_over(list(range(6)))(probs, labels) == pytest.approx(macro_f1(probs, labels))


def test_ece_hand_computed() -> None:
    # Two bins used: conf 0.9 (2 samples, 1 correct) and conf 0.6 (2 samples, 2 correct).
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1, 0, 1])
    expected = 0.5 * abs(0.5 - 0.9) + 0.5 * abs(1.0 - 0.6)
    assert ece(probs, labels, n_bins=10) == pytest.approx(expected)
    bins = reliability_bins(probs, labels, n_bins=10)
    assert bins.count.sum() == 4


def test_perfectly_calibrated_model_has_small_ece() -> None:
    logits, labels = _random_problem(n=50_000)
    probs = probs_from_logits(logits)
    assert ece(probs, labels) < 0.01
    assert classwise_ece(probs, labels) < 0.01


def test_nll_and_brier_on_onehot() -> None:
    probs = np.eye(3)[[0, 1, 2]]
    labels = np.array([0, 1, 2])
    assert nll(probs, labels) == pytest.approx(0.0, abs=1e-9)
    assert brier(probs, labels) == pytest.approx(0.0)
    assert brier(np.eye(3)[[1, 2, 0]], labels) == pytest.approx(2.0)


@pytest.mark.parametrize("true_t", [0.5, 1.0, 2.5])
def test_temperature_scaling_recovers_planted_temperature(true_t: float) -> None:
    logits, labels = _random_problem(n=20_000, seed=1)
    # Labels follow softmax(logits); the model reports logits * true_t, i.e. miscalibrated by 1/true_t.
    assert fit_temperature(logits * true_t, labels) == pytest.approx(true_t, rel=0.05)


def test_ensemble_log_probs_is_log_mean_prob() -> None:
    a, _ = _random_problem(n=10, seed=2)
    b, _ = _random_problem(n=10, seed=3)
    expected = (probs_from_logits(a) + probs_from_logits(b)) / 2
    np.testing.assert_allclose(np.exp(ensemble_log_probs([a, b])), expected, rtol=1e-10)
    np.testing.assert_allclose(ensemble_probs([a, a]), probs_from_logits(a), rtol=1e-10)


def test_cluster_members_partition_rows() -> None:
    groups = np.array(["b", "a", "b", "c", "a"])
    members = cluster_members(groups)
    assert sorted(np.concatenate(members).tolist()) == [0, 1, 2, 3, 4]
    assert sorted(len(m) for m in members) == [1, 2, 2]


def test_cluster_bootstrap_is_wider_than_frame_bootstrap_for_correlated_frames() -> None:
    rng = np.random.default_rng(0)
    # 20 videos, 100 near-identical frames each: correctness is a per-video property.
    video_ok = rng.random(20) < 0.6
    labels = np.zeros(2000, dtype=int)
    correct = np.repeat(video_ok, 100)
    probs = np.where(correct[:, None], [[0.9, 0.1]], [[0.1, 0.9]])
    groups = np.repeat(np.arange(20), 100)
    acc = {"accuracy": lambda p, y: float((p.argmax(1) == y).mean())}
    clustered = bootstrap_metrics(acc, [probs], labels, groups, n_boot=500)["accuracy"]
    frame = bootstrap_metrics(acc, [probs], labels, None, n_boot=500)["accuracy"]
    assert clustered.low <= clustered.point <= clustered.high
    assert (clustered.high - clustered.low) > 3 * (frame.high - frame.low)


def test_difference_and_paired_bootstrap() -> None:
    rng = np.random.default_rng(0)
    values = rng.random(200)
    iv = paired_bootstrap(
        lambda rows: float(values[rows].mean() - 0.5 * values[rows].mean()), 200, n_boot=300
    )
    assert iv.point == pytest.approx(0.5 * values.mean())
    assert iv.excludes_zero
    acc = {"accuracy": lambda p, y: float((p.argmax(1) == y).mean())}
    good = bootstrap_metrics(acc, [np.tile([[0.9, 0.1]], (100, 1))], np.zeros(100, int), n_boot=200)[
        "accuracy"
    ]
    bad = bootstrap_metrics(acc, [np.tile([[0.1, 0.9]], (100, 1))], np.zeros(100, int), n_boot=200)[
        "accuracy"
    ]
    assert difference(good, bad).point == pytest.approx(1.0)
