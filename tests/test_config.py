from __future__ import annotations

from pathlib import Path

import pytest

from capsule_xai_audit.config import ConfigError, TrainExperiment, XaiExperiment, load_config, to_dict

REPO = Path(__file__).resolve().parents[1]


def test_defaults_and_overrides() -> None:
    cfg = load_config(
        TrainExperiment, None, ["train.seed=3", "train.backbone=convnext_tiny", "train.lr=1e-3"]
    )
    assert cfg.train.seed == 3
    assert cfg.train.backbone == "convnext_tiny"
    assert cfg.train.lr == pytest.approx(1e-3)
    assert isinstance(cfg.paths.runs_root, Path)


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(TrainExperiment, None, ["train.sed=3"])


def test_literal_and_type_validation() -> None:
    with pytest.raises(ConfigError, match="not one of"):
        load_config(TrainExperiment, None, ["train.split_protocol=frame"])
    with pytest.raises(ConfigError, match="integer"):
        load_config(TrainExperiment, None, ["train.seed=1.5"])
    with pytest.raises(ConfigError, match="test_fold"):
        load_config(TrainExperiment, None, ["train.test_fold=2"])


def test_tuple_fields_and_roundtrip() -> None:
    cfg = load_config(XaiExperiment, None, ["xai.methods=[gradcam, random]"])
    assert cfg.xai.methods == ("gradcam", "random")
    assert to_dict(cfg)["xai"]["methods"] == ["gradcam", "random"]


@pytest.mark.parametrize("path", sorted((REPO / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_load(path: Path) -> None:
    from capsule_xai_audit import cli

    cls = cli.CONFIG_FILES.get(path.name)
    if cls is None:
        pytest.skip(f"{path.name} is not an experiment config")
    load_config(cls, path)
