from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from capsule_xai_audit.data.cache import (
    CacheError,
    build_cache,
    cache_paths,
    open_cache,
    square_crop_box,
    transform_boxes,
)
from capsule_xai_audit.data.class_mapping import ClassMapping, ClassMappingError, restrict_to_classes
from capsule_xai_audit.data.dedup import hamming_candidates, match_duplicates, phash, popcount64
from capsule_xai_audit.data.kvasir import (
    KvasirIndexError,
    build_kvasir_index,
    discover_images,
    multilabel_frames,
    read_bounding_boxes,
    video_id_from_stem,
    videos_in_both_folds,
)
from capsule_xai_audit.data.splits import LeakageError, Split, grouped_inner_split, make_split, video_folds

from .conftest import FakeWorld

# --------------------------------------------------------------------------- Kvasir index


def test_kvasir_index_joins_png_files_to_jpg_split_names(world: FakeWorld) -> None:
    idx = world.kvasir_index
    # Extra frames outside the official split and the one two-label frame are excluded.
    assert len(idx) == 2 * 4 * 24 - 1
    assert idx["image_id"].is_unique
    assert all(Path(p).suffix == ".png" for p in idx["path"])
    assert set(idx["official_fold"]) == {0, 1}
    labels = np.array(world.mapping.kvasir_labels)
    assert (labels[idx["label_idx"]] == idx["label"]).all()


def test_two_label_frames_are_dropped_and_listed(world: FakeWorld) -> None:
    multi = multilabel_frames(world.paths.kvasir_splits)
    assert set(multi["image_id"]) == {world.paths.kvasir_multilabel}
    assert sorted(multi["label"]) == ["Erosion", "Pylorus"]
    assert world.paths.kvasir_multilabel not in set(world.kvasir_index["image_id"])


def test_identical_copies_are_accepted_but_different_images_are_not(world: FakeWorld, tmp_path: Path) -> None:
    found = discover_images(world.paths.kvasir_images)
    assert world.paths.kvasir_multilabel in found  # stored in two folders, byte-identical
    a, b = tmp_path / "A", tmp_path / "B"
    a.mkdir()
    b.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(a / "v_1.png")
    Image.fromarray(np.full((8, 8, 3), 255, np.uint8)).save(b / "v_1.png")
    with pytest.raises(KvasirIndexError, match="two different images"):
        discover_images(tmp_path)


def test_shared_video_is_reported_not_rejected(world: FakeWorld) -> None:
    shared = videos_in_both_folds(world.kvasir_index)
    assert list(shared.index) == [world.paths.shared_video]


def test_bounding_boxes_present_exactly_for_findings(world: FakeWorld) -> None:
    idx = world.kvasir_index
    has_box = idx["bboxes"].map(lambda b: len(json.loads(b)) > 0)
    assert not has_box[idx["label"] == "Normal"].any()
    assert has_box[idx["label"] != "Normal"].all()
    box = json.loads(idx.loc[has_box.idxmax(), "bboxes"])[0]
    assert box[0] < box[2]
    assert box[1] < box[3]


def test_missing_images_fail_loudly_unless_allowed(world: FakeWorld, tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    fold0 = pd.read_csv(world.paths.kvasir_splits / "split_0.csv")
    missing = pd.DataFrame([{"filename": "deadbeefdeadbeef_1.jpg", "label": "Normal"}])
    pd.concat([fold0, missing]).to_csv(split_dir / "split_0.csv", index=False)
    pd.read_csv(world.paths.kvasir_splits / "split_1.csv").to_csv(split_dir / "split_1.csv", index=False)
    with pytest.raises(KvasirIndexError, match="no image file"):
        build_kvasir_index(world.paths.kvasir_images, split_dir)
    idx = build_kvasir_index(world.paths.kvasir_images, split_dir, allow_missing=True)
    assert "deadbeefdeadbeef_1" not in set(idx["image_id"])


def test_video_id_parsing() -> None:
    assert video_id_from_stem("bca26705313a4644_20404") == "bca26705313a4644"
    with pytest.raises(KvasirIndexError):
        video_id_from_stem("no-frame-number")


def test_metadata_with_minmax_columns_and_commas(tmp_path: Path) -> None:
    path = tmp_path / "meta.csv"
    pd.DataFrame(
        {
            "Filename": ["a_1.png", "a_2.png"],
            "x_min": [1, None],
            "y_min": [2, None],
            "x_max": [5, None],
            "y_max": [9, None],
        }
    ).to_csv(path, index=False)
    assert read_bounding_boxes(path) == {"a_1": [[1.0, 2.0, 5.0, 9.0]]}


# --------------------------------------------------------------------------- splits


def test_official_protocol_follows_the_published_folds(world: FakeWorld) -> None:
    idx = world.kvasir_index
    split = make_split(idx, "official", test_fold=1, split_seed=0, val_fraction=0.25)
    videos = idx["video_id"]
    assert (idx["official_fold"].iloc[split.test] == 1).all()
    assert len(split.test) == int((idx["official_fold"] == 1).sum())
    # As published, one video is on both sides; the inner split is still grouped by video.
    assert set(videos.iloc[split.fit]) & set(videos.iloc[split.test]) <= {world.paths.shared_video}
    assert not set(videos.iloc[split.fit]) & set(videos.iloc[split.val])


@pytest.mark.parametrize("test_fold", [0, 1])
def test_video_protocol_is_patient_level(world: FakeWorld, test_fold: int) -> None:
    idx = world.kvasir_index
    split = make_split(idx, "video", test_fold=test_fold, split_seed=0, val_fraction=0.25)
    videos = idx["video_id"]
    assert not set(videos.iloc[split.fit]) & set(videos.iloc[split.test])
    assert not set(videos.iloc[split.val]) & set(videos.iloc[split.test])
    assert not set(videos.iloc[split.fit]) & set(videos.iloc[split.val])
    # The shared video moves whole to the fold that holds most of its frames (fold 0 here).
    folds = video_folds(idx)
    assert set(folds[(videos == world.paths.shared_video).to_numpy()]) == {0}
    unchanged = (videos != world.paths.shared_video).to_numpy()
    np.testing.assert_array_equal(folds[unchanged], idx["official_fold"].to_numpy()[unchanged])


def test_random_protocol_matches_official_test_size_and_classes(world: FakeWorld) -> None:
    idx = world.kvasir_index
    off = make_split(idx, "official", test_fold=0, split_seed=0, val_fraction=0.2)
    rnd = make_split(idx, "random", test_fold=0, split_seed=0, val_fraction=0.2)
    labels = idx["label_idx"].to_numpy()
    assert len(rnd.test) == len(off.test)
    assert (np.bincount(labels[rnd.test], minlength=11) == np.bincount(labels[off.test], minlength=11)).all()
    # By construction the naive protocol shares videos between train and test.
    assert set(idx["video_id"].iloc[rnd.fit]) & set(idx["video_id"].iloc[rnd.test])
    # Fixed split seed -> identical split.
    again = make_split(idx, "random", test_fold=0, split_seed=0, val_fraction=0.2)
    np.testing.assert_array_equal(rnd.test, again.test)


@pytest.mark.parametrize("protocol", ["official", "video"])
def test_inner_split_puts_every_class_in_fit_and_validation(world: FakeWorld, protocol: str) -> None:
    idx = world.kvasir_index
    split = make_split(idx, protocol, test_fold=1, split_seed=0, val_fraction=0.25)
    labels = idx["label_idx"].to_numpy()
    train = np.concatenate([split.fit, split.val])
    present = {c for c in np.unique(labels[train]) if (labels[train] == c).sum() >= 2}
    assert present <= set(labels[split.fit])
    assert present <= set(labels[split.val])


def test_grouped_inner_split_keeps_rare_classes_in_fit() -> None:
    # Video a: Normal plus the only Blood frames. Videos b, c: Normal plus Ulcer,
    # c holding most of the Ulcer frames. Videos d-f: Normal only.
    rows = (
        [("a", 0)] * 20
        + [("a", 1)] * 10
        + [("b", 0)] * 20
        + [("b", 2)] * 2
        + [("c", 0)] * 20
        + [("c", 2)] * 30
        + [(v, 0) for v in "def" for _ in range(20)]
    )
    videos = np.array([v for v, _ in rows])
    labels = np.array([c for _, c in rows])
    frames = np.arange(len(rows))
    fit, val = grouped_inner_split(labels, videos, frames, val_fraction=0.2, seed=0)
    assert not set(fit) & set(val)
    assert len(fit) + len(val) == len(rows)
    for c in (0, 1, 2):
        assert (labels[fit] == c).any()
        assert (labels[val] == c).any()
        assert (labels[val] == c).sum() < (labels[fit] == c).sum()
    # Blood lives in one video: validation gets its last frames, and that video stays in fit.
    blood = np.flatnonzero(labels == 1)
    assert set(blood[-2:]) == set(val) & set(blood)
    # Ulcer: video c (30 of 32 frames) is never moved whole into validation.
    assert (videos[fit] == "c").any()


def test_split_rejects_overlapping_parts() -> None:
    with pytest.raises(LeakageError):
        Split(fit=np.array([0, 1]), val=np.array([1]), test=np.array([2]))


# --------------------------------------------------------------------------- cache


def test_cache_roundtrip_and_index_guard(world: FakeWorld) -> None:
    ids = world.kvasir_index["image_id"].tolist()
    pixels, sizes = open_cache(world.kvasir_cache, ids)
    assert pixels.shape == (len(ids), 64, 64, 3)
    assert pixels.dtype == np.uint8
    assert (sizes == 96).all()
    with pytest.raises(CacheError):
        open_cache(world.kvasir_cache, ids[::-1])


def test_cache_resumes_after_interruption(world: FakeWorld, tmp_path: Path) -> None:
    paths = world.kvasir_index["path"].tolist()[:10]
    ids = world.kvasir_index["image_id"].tolist()[:10]
    prefix = tmp_path / "c"
    build_cache(paths[:10], ids, prefix, size=32, num_workers=1, chunk=4)
    full, _ = open_cache(prefix, ids)
    reference = np.array(full)
    meta_path = cache_paths(prefix)[2]
    meta = json.loads(meta_path.read_text())
    meta["filled"] = 4
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(CacheError, match="incomplete"):
        open_cache(prefix, ids)
    build_cache(paths, ids, prefix, size=32, num_workers=1, chunk=4)
    resumed, _ = open_cache(prefix, ids)
    np.testing.assert_array_equal(resumed, reference)


def test_boxes_follow_centre_crop_and_resize() -> None:
    assert square_crop_box(400, 300) == (50, 0, 350, 300)
    (box,) = transform_boxes([[50, 0, 350, 300]], 400, 300, 224)
    np.testing.assert_allclose(box, (0, 0, 224, 224))
    assert transform_boxes([[0, 0, 40, 40]], 400, 300, 224) == []  # cropped away


# --------------------------------------------------------------------------- dedup


def test_popcount_matches_python() -> None:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2**63, size=100, dtype=np.uint64)
    assert popcount64(x).tolist() == [bin(int(v)).count("1") for v in x]


def test_phash_is_stable_under_recompression(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    img = np.asarray(Image.fromarray(img).resize((224, 224)))
    Image.fromarray(img).save(tmp_path / "a.jpg", quality=85)
    recompressed = np.asarray(Image.open(tmp_path / "a.jpg"))
    q, _r, d = hamming_candidates(np.array([phash(img)]), np.array([phash(recompressed)]), max_hamming=6)
    assert len(q) == 1
    assert d[0] <= 6


def test_dedup_finds_planted_copies_without_false_positives(world: FakeWorld) -> None:
    k_pixels, _ = open_cache(world.kvasir_cache)
    cv_pixels, _ = open_cache(world.cv_cache)
    result = match_duplicates(cv_pixels, k_pixels, max_hamming=12, min_ncc=0.9)
    found = set(world.cv_index["image_id"].iloc[result.duplicate_positions])
    assert found == set(world.paths.planted_duplicates)


def test_capsule_vision_index_records_the_source_folder(world: FakeWorld) -> None:
    cv = world.cv_index
    assert set(cv["source"]) == {"KID", "SEE-AI", "KVASIR"}
    assert set(cv.loc[cv["source"] == "KVASIR", "image_id"]) == set(world.paths.cv_kvasir_source)
    assert (cv["label"] == cv["image_id"].str.split("/").str[1]).all()


# --------------------------------------------------------------------------- class mapping


def test_class_mapping_is_consistent(mapping: ClassMapping) -> None:
    assert len(mapping.kvasir_labels) == 11
    assert len(mapping.capsule_vision_labels) == 10
    assert len(mapping.shared) == 8
    inv = mapping.capsule_vision_to_kvasir_index()
    assert inv["Bleeding"] == mapping.kvasir_labels.index("Blood")
    with pytest.raises(ClassMappingError):
        ClassMapping(("a", "b"), ("x",), {"a": "x", "b": "x"})


def test_restrict_renormalises_over_kept_classes() -> None:
    logits = np.log(np.array([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]]))
    labels = np.array([0, 2])
    sub_logits, sub_labels, mask = restrict_to_classes(logits, labels, np.array([0, 1]))
    assert mask.tolist() == [True, False]
    probs = np.exp(sub_logits) / np.exp(sub_logits).sum(1, keepdims=True)
    np.testing.assert_allclose(probs, [[2 / 3, 1 / 3]])
    assert sub_labels.tolist() == [0]
