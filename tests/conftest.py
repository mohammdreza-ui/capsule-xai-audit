from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from capsule_xai_audit.data.cache import build_cache
from capsule_xai_audit.data.capsule_vision import build_capsule_vision_index
from capsule_xai_audit.data.class_mapping import ClassMapping, load_class_mapping
from capsule_xai_audit.data.kvasir import build_kvasir_index
from capsule_xai_audit.testing.fake_data import FakeDataPaths, make_fake_datasets

REPO = Path(__file__).resolve().parents[1]
CACHE_SIZE = 64


@dataclass(frozen=True)
class FakeWorld:
    root: Path
    paths: FakeDataPaths
    mapping: ClassMapping
    kvasir_index: pd.DataFrame
    cv_index: pd.DataFrame
    kvasir_cache: Path
    cv_cache: Path


@pytest.fixture(scope="session")
def mapping() -> ClassMapping:
    return load_class_mapping(REPO / "configs" / "class_mapping.yaml")


@pytest.fixture(scope="session")
def world(tmp_path_factory: pytest.TempPathFactory, mapping: ClassMapping) -> FakeWorld:
    root = tmp_path_factory.mktemp("fake")
    paths = make_fake_datasets(root, mapping, size=96)
    kvasir = build_kvasir_index(
        paths.kvasir_images, paths.kvasir_splits, paths.kvasir_metadata, mapping.kvasir_labels
    )
    cv = build_capsule_vision_index(paths.capsule_vision_root, mapping.capsule_vision_labels)
    k_cache, cv_cache = root / "cache" / "kvasir", root / "cache" / "cv"
    build_cache(kvasir["path"].tolist(), kvasir["image_id"].tolist(), k_cache, size=CACHE_SIZE, num_workers=2)
    build_cache(cv["path"].tolist(), cv["image_id"].tolist(), cv_cache, size=CACHE_SIZE, num_workers=2)
    return FakeWorld(root, paths, mapping, kvasir, cv, k_cache, cv_cache)
