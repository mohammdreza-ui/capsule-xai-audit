"""Command-line entry point: ``cxa <command> ...``.

Data preparation
    fetch-splits, fetch-metadata, unpack-kvasir, index-kvasir, index-cv, cache, dedup
Experiments (YAML config + ``--set section.key=value`` overrides)
    train, fewshot, xai, superpixel, color
Analyses of finished training runs
    leakage (Q1), calibration (Q2)
Reporting / testing
    report, make-fake-data
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from capsule_xai_audit.config import (
    ColorExperiment,
    FewShotExperiment,
    SuperpixelExperiment,
    TrainExperiment,
    XaiExperiment,
    load_config,
)
from capsule_xai_audit.data.class_mapping import load_class_mapping

# Shipped config file -> experiment type (used by `cxa` and by the config tests).
CONFIG_FILES: dict[str, type[Any]] = {
    "train.yaml": TrainExperiment,
    "fewshot.yaml": FewShotExperiment,
    "xai.yaml": XaiExperiment,
    "superpixel.yaml": SuperpixelExperiment,
    "color.yaml": ColorExperiment,
}

DEFAULT_MAPPING = Path("configs/class_mapping.yaml")


def _experiment_parser(sub: Any, name: str, help_text: str, default_config: str) -> argparse.ArgumentParser:
    p: argparse.ArgumentParser = sub.add_parser(name, help=help_text)
    p.add_argument("--config", type=Path, default=Path("configs") / default_config)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override, e.g. train.seed=3")
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cxa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-splits", help="download Kvasir-Capsule official split CSVs")
    p.add_argument("--out", type=Path, default=Path("data/kvasir/official_splits"))

    p = sub.add_parser(
        "fetch-metadata", help="download Kvasir-Capsule metadata.csv (bounding boxes) from OSF"
    )
    p.add_argument("--out", type=Path, default=Path("data/kvasir/metadata.csv"))

    p = sub.add_parser("unpack-kvasir", help="extract the per-class archives of the Simula Kvasir zip")
    p.add_argument("--src", type=Path, required=True, help="folder holding the *.tar.gz class archives")
    p.add_argument("--out", type=Path, required=True, help="folder to extract the class folders into")

    p = sub.add_parser("index-kvasir", help="build the Kvasir-Capsule frame index")
    p.add_argument("--images", type=Path, required=True, help="labelled_images root (class folders)")
    p.add_argument("--splits", type=Path, default=Path("data/kvasir/official_splits"))
    p.add_argument("--metadata", type=Path, default=None, help="metadata.csv with bounding boxes")
    p.add_argument("--out", type=Path, default=Path("data/index/kvasir.csv"))
    p.add_argument("--manifests", type=Path, default=Path("manifests"))
    p.add_argument("--class-mapping", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--allow-missing", action="store_true")

    p = sub.add_parser("index-cv", help="build the Capsule Vision 2024 train+val index")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/index/cv2024.csv"))
    p.add_argument("--class-mapping", type=Path, default=DEFAULT_MAPPING)

    p = sub.add_parser("cache", help="decode + resize images of an index into a uint8 memmap")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="cache prefix, e.g. cache/kvasir_224")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)

    p = sub.add_parser("dedup", help="remove Kvasir frames from Capsule Vision; write the external index")
    p.add_argument("--kvasir-index", type=Path, default=Path("data/index/kvasir.csv"))
    p.add_argument("--kvasir-cache", type=Path, default=Path("cache/kvasir_224"))
    p.add_argument("--cv-index", type=Path, default=Path("data/index/cv2024.csv"))
    p.add_argument("--cv-cache", type=Path, default=Path("cache/cv2024_224"))
    p.add_argument("--out", type=Path, default=Path("data/index/cv2024_external.csv"))
    p.add_argument("--manifests", type=Path, default=Path("manifests"))
    p.add_argument("--class-mapping", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--max-hamming", type=int, default=10)
    p.add_argument("--min-ncc", type=float, default=0.95)

    _experiment_parser(sub, "train", "fine-tune one model (Q1/Q2/Slice B)", "train.yaml")
    _experiment_parser(sub, "fewshot", "few-shot linear probes (Q3)", "fewshot.yaml")
    _experiment_parser(sub, "xai", "attribution evaluation (Q4)", "xai.yaml")
    _experiment_parser(sub, "superpixel", "patch vs superpixel units (Slice A)", "superpixel.yaml")
    _experiment_parser(sub, "color", "CIELAB colour shift (Slice B)", "color.yaml")

    for name, help_text in (("leakage", "Q1 analysis"), ("calibration", "Q2 analysis")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--runs-root", type=Path, default=Path("runs"))
        p.add_argument("--backbone", required=True)
        p.add_argument("--test-fold", type=int, default=1)
        p.add_argument("--tag", default="base")
        p.add_argument("--n-boot", type=int, default=2000)
        p.add_argument("--class-mapping", type=Path, default=DEFAULT_MAPPING)

    p = sub.add_parser("report", help="write results/RESULTS.md from runs/analysis")
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("results"))

    p = sub.add_parser("make-fake-data", help="write small synthetic Kvasir/Capsule Vision trees")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--class-mapping", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--videos-per-fold", type=int, default=4)
    p.add_argument("--frames-per-video", type=int, default=40)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cmd = args.command

    # Imports are local so that `cxa --help` and data commands start fast.
    if cmd == "fetch-splits":
        from capsule_xai_audit.experiments.prepare import fetch_official_splits

        fetch_official_splits(args.out)
    elif cmd == "fetch-metadata":
        from capsule_xai_audit.experiments.prepare import fetch_kvasir_metadata

        fetch_kvasir_metadata(args.out)
    elif cmd == "unpack-kvasir":
        from capsule_xai_audit.experiments.prepare import unpack_class_archives

        print(f"{unpack_class_archives(args.src, args.out)} images in {args.out}")
    elif cmd == "index-kvasir":
        from capsule_xai_audit.experiments.prepare import index_kvasir

        index_kvasir(
            args.images,
            args.splits,
            args.metadata,
            load_class_mapping(args.class_mapping),
            args.out,
            args.manifests,
            args.allow_missing,
        )
    elif cmd == "index-cv":
        from capsule_xai_audit.experiments.prepare import index_capsule_vision

        index_capsule_vision(args.root, load_class_mapping(args.class_mapping), args.out)
    elif cmd == "cache":
        from capsule_xai_audit.experiments.prepare import cache_index

        cache_index(args.index, args.out, args.size, args.workers)
    elif cmd == "dedup":
        from capsule_xai_audit.experiments.prepare import build_external_set

        build_external_set(
            args.kvasir_index,
            args.kvasir_cache,
            args.cv_index,
            args.cv_cache,
            load_class_mapping(args.class_mapping),
            args.out,
            args.manifests,
            args.max_hamming,
            args.min_ncc,
        )
    elif cmd == "train":
        from capsule_xai_audit.experiments.train_run import run_training

        run_training(load_config(TrainExperiment, args.config, args.set))
    elif cmd == "fewshot":
        from capsule_xai_audit.experiments.fewshot import run_fewshot

        run_fewshot(load_config(FewShotExperiment, args.config, args.set))
    elif cmd == "xai":
        from capsule_xai_audit.experiments.xai import run_xai

        run_xai(load_config(XaiExperiment, args.config, args.set))
    elif cmd == "superpixel":
        from capsule_xai_audit.experiments.superpixel import run_superpixel

        run_superpixel(load_config(SuperpixelExperiment, args.config, args.set))
    elif cmd == "color":
        from capsule_xai_audit.experiments.color import run_color

        run_color(load_config(ColorExperiment, args.config, args.set))
    elif cmd == "leakage":
        from capsule_xai_audit.experiments.leakage import leakage_analysis

        leakage_analysis(args.runs_root, args.backbone, args.test_fold, args.tag, args.n_boot)
    elif cmd == "calibration":
        from capsule_xai_audit.experiments.calibration import calibration_analysis

        calibration_analysis(
            args.runs_root,
            load_class_mapping(args.class_mapping),
            args.backbone,
            args.test_fold,
            args.tag,
            args.n_boot,
        )
    elif cmd == "report":
        from capsule_xai_audit.report import build_report

        build_report(args.runs_root, args.out)
    elif cmd == "make-fake-data":
        from capsule_xai_audit.testing.fake_data import make_fake_datasets

        paths = make_fake_datasets(
            args.out, load_class_mapping(args.class_mapping), args.videos_per_fold, args.frames_per_video
        )
        print(paths)
    else:  # pragma: no cover - argparse enforces the choices
        raise AssertionError(cmd)


if __name__ == "__main__":
    main()
