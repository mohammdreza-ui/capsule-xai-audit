"""Typed experiment configuration.

Configs are frozen dataclasses loaded from YAML. Loading is strict: unknown keys, wrong
types and invalid ``Literal`` values raise ``ConfigError``. Command-line overrides use
dotted paths, e.g. ``--set train.seed=3 train.backbone=convnext_tiny``.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, Union

import yaml

T = TypeVar("T")

SplitProtocol = Literal["official", "random", "video"]
SelectMetric = Literal["macro_f1", "nll"]
ClassWeighting = Literal["none", "inv_sqrt"]


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------- sections


@dataclass(frozen=True)
class PathsConfig:
    """Where indices, caches and outputs live. All other configs share this section."""

    kvasir_index: Path = Path("data/index/kvasir.csv")
    kvasir_cache: Path = Path("cache/kvasir_224")
    external_index: Path | None = Path("data/index/cv2024_external.csv")
    external_cache: Path | None = Path("cache/cv2024_224")
    class_mapping: Path = Path("configs/class_mapping.yaml")
    runs_root: Path = Path("runs")


@dataclass(frozen=True)
class TrainConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    split_protocol: SplitProtocol = "official"
    test_fold: int = 1
    # Seed of the random frame-level split; kept fixed across training seeds so every
    # seed of one protocol is evaluated on the same test frames.
    split_seed: int = 0
    seed: int = 0
    # Free-form variant label, part of the run directory (e.g. "base", "lab-aug").
    tag: str = "base"
    epochs: int = 12
    batch_size: int = 64
    eval_batch_size: int = 128
    # Memory layout only (NHWC for CNNs); faster fp16 convolutions, same numerics.
    channels_last: bool = True
    lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    grad_clip: float = 1.0
    val_fraction: float = 0.15
    early_stopping_patience: int = 4
    select_metric: SelectMetric = "macro_f1"
    # Cross-entropy class weights from fit-set frequencies: "inv_sqrt" weights class c by
    # 1/sqrt(n_c), normalised to mean 1 over fit frames (72% of Kvasir-Capsule is Normal).
    class_weighting: ClassWeighting = "inv_sqrt"
    amp: bool = True
    num_workers: int = 4
    color_aug_max_delta_e: float = 0.0
    color_aug_prob: float = 0.8
    max_steps_per_epoch: int | None = None
    # Training log: one train_steps.jsonl record every this many steps (window means).
    log_every_steps: int = 25
    # Keep the validation logits of every epoch (val_logits/epochNN.npz), not only of the best.
    save_val_logits: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.test_fold not in (0, 1):
            raise ConfigError(f"test_fold must be 0 or 1, got {self.test_fold}")
        if not 0.0 < self.val_fraction < 0.5:
            raise ConfigError(f"val_fraction must be in (0, 0.5), got {self.val_fraction}")
        if self.color_aug_max_delta_e < 0:
            raise ConfigError("color_aug_max_delta_e must be >= 0")


@dataclass(frozen=True)
class TrainExperiment:
    paths: PathsConfig = field(default_factory=PathsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass(frozen=True)
class FewShotConfig:
    backbones: tuple[str, ...] = ("dinov2_vits14", "resnet50", "convnext_tiny")
    shots: tuple[int, ...] = (1, 5, 20)
    draws: int = 10
    test_fold: int = 1
    seed: int = 0
    # Inverse L2 strength of the logistic-regression probe. Fixed in advance because a
    # 1-shot support set leaves nothing to cross-validate on; calibration depends on it,
    # so the other values in C_sensitivity are reported as well.
    C: float = 10.0
    C_sensitivity: tuple[float, ...] = (1.0, 100.0)
    pretrained: bool = True
    batch_size: int = 128
    num_workers: int = 4
    amp: bool = True
    device: str = "auto"


@dataclass(frozen=True)
class FewShotExperiment:
    paths: PathsConfig = field(default_factory=PathsConfig)
    fewshot: FewShotConfig = field(default_factory=FewShotConfig)


@dataclass(frozen=True)
class XaiConfig:
    # Run directory of the trained model to explain, relative to paths.runs_root.
    run: str = "train/official/resnet50/base/fold1/seed0"
    methods: tuple[str, ...] = ("gradcam", "integrated_gradients", "saliency", "grad_x_input", "random")
    max_images: int = 300
    sanity_images: int = 50
    ig_steps: int = 32
    perturbation_steps: int = 50
    pointing_tolerance_px: int = 15
    n_bootstrap: int = 2000
    batch_size: int = 32
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class XaiExperiment:
    paths: PathsConfig = field(default_factory=PathsConfig)
    xai: XaiConfig = field(default_factory=XaiConfig)


@dataclass(frozen=True)
class SuperpixelConfig:
    backbone: str = "dinov2_vits14"
    test_fold: int = 1
    n_segments: int = 64
    compactness: float = 10.0
    max_images: int = 300
    pointing_tolerance_px: int = 15
    C: float = 10.0
    pretrained: bool = True
    n_bootstrap: int = 2000
    save_graphs: bool = True
    seed: int = 0
    batch_size: int = 64
    num_workers: int = 4
    device: str = "auto"


@dataclass(frozen=True)
class SuperpixelExperiment:
    paths: PathsConfig = field(default_factory=PathsConfig)
    superpixel: SuperpixelConfig = field(default_factory=SuperpixelConfig)


@dataclass(frozen=True)
class ColorConfig:
    # Run directories (relative to runs_root) evaluated under colour shift. Glob patterns allowed.
    runs: tuple[str, ...] = (
        "train/official/resnet50/base/fold1/seed*",
        "train/official/resnet50/lab-aug/fold1/seed*",
    )
    delta_e: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0)
    # Hue angles (degrees) of the shift direction in the a*b* plane.
    hue_angles: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    max_images: int = 5000
    batch_size: int = 128
    num_workers: int = 4
    amp: bool = True
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class ColorExperiment:
    paths: PathsConfig = field(default_factory=PathsConfig)
    color: ColorConfig = field(default_factory=ColorConfig)


# --------------------------------------------------------------------------- loading


def _is_optional(tp: Any) -> tuple[bool, Any]:
    origin = typing.get_origin(tp)
    if origin in (Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(typing.get_args(tp)) == 2:
            return True, args[0]
    return False, tp


def _coerce(value: Any, tp: Any, where: str) -> Any:
    optional, inner = _is_optional(tp)
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{where}: null is not allowed")
    tp = inner
    origin = typing.get_origin(tp)

    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        return _build(tp, value, where)
    if origin is Literal:
        allowed = typing.get_args(tp)
        if value not in allowed:
            raise ConfigError(f"{where}: {value!r} is not one of {allowed}")
        return value
    if origin is tuple:
        args = typing.get_args(tp)
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0], f"{where}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
        return tuple(_coerce(v, a, f"{where}[{i}]") for i, (v, a) in enumerate(zip(args, value, strict=True)))
    if tp is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"{where}: expected a path string")
        return Path(value)
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{where}: expected true/false, got {value!r}")
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, str):
            # PyYAML (YAML 1.1) reads "1e-3" as a string; accept plain numeric strings.
            try:
                return float(value)
            except ValueError:
                raise ConfigError(f"{where}: expected a number, got {value!r}") from None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported config type {tp!r}")


def _build(cls: type[T], data: Mapping[str, Any], where: str = "") -> T:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where or cls.__name__}: unknown keys {sorted(unknown)}")
    kwargs = {k: _coerce(v, hints[k], f"{where}.{k}" if where else k) for k, v in data.items()}
    return cls(**kwargs)


def _parse_override(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise ConfigError(f"override {item!r} must look like section.key=value")
    key, raw = item.split("=", 1)
    # YAML parsing gives ints, floats, bools, null and [lists] for free.
    return key.strip().split("."), yaml.safe_load(raw)


def _set_nested(data: dict[str, Any], keys: list[str], value: Any) -> None:
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ConfigError(f"cannot override inside non-mapping key {key!r}")
    node[keys[-1]] = value


def load_config(cls: type[T], path: Path | None = None, overrides: Sequence[str] = ()) -> T:
    data: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        data = loaded
    for item in overrides:
        keys, value = _parse_override(item)
        _set_nested(data, keys, value)
    return _build(cls, data)


def to_dict(cfg: Any) -> dict[str, Any]:
    """Plain-JSON view of a config (paths as strings, tuples as lists)."""

    def convert(v: Any) -> Any:
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (tuple, list)):
            return [convert(x) for x in v]
        if isinstance(v, dict):
            return {k: convert(x) for k, x in v.items()}
        return v

    return typing.cast(dict[str, Any], convert(dataclasses.asdict(cfg)))
