#!/usr/bin/env python3
"""End-to-end smoke test on the real Kvasir-Capsule and Capsule Vision 2024 downloads.

Runs every stage of the pipeline with tiny settings (few training steps, few images) so
that problems in the data or the code show up in minutes rather than after hours on Kaggle.
It does not produce results; everything it writes goes to ``--work`` (default data/smoke),
except the image caches, which are shared with real runs (``--cache-dir``, default cache/)
because building them is the slow part.

Each stage reports PASS, WARN (works, but something in the data needs a decision), FAIL,
or SKIP (a stage it depends on failed). The exit code is 1 if any stage failed.

    source ~/VENV/AI/bin/activate
    python scripts/smoke_real_data.py \\
        --kvasir-zip data/kvasir-capsule-labeled-images.zip \\
        --cv-zip "data/Capsule Vision 2024.zip"
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
METADATA_URL = "https://osf.io/download/kzc8w/"  # Kvasir-Capsule metadata.csv (OSF, Google Drive storage)
SPLIT_URL = "https://raw.githubusercontent.com/simula/kvasir-capsule/master/official_splits/split_{}.csv"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# Published numbers the downloads must reproduce.
KVASIR_LABELLED_IMAGES = 47_238
KVASIR_CLASS_FOLDERS = 14
CV_FRAMES = {"training": 37_607, "validation": 16_132}
CV_KVASIR_FRAMES = {"training": 27_011, "validation": 11_581}


class StageFailed(Exception):
    pass


@dataclass
class Stage:
    name: str
    status: str = "PENDING"
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class Context:
    args: argparse.Namespace
    work: Path
    kvasir_images: Path | None = None
    kvasir_metadata: Path | None = None
    cv_root: Path | None = None
    stages: dict[str, Stage] = field(default_factory=dict)

    @property
    def index_dir(self) -> Path:
        return self.work / "index"

    @property
    def runs(self) -> Path:
        return self.work / "runs"

    def path_overrides(self) -> list[str]:
        cache = self.args.cache_dir
        return [
            f"paths.kvasir_index={self.index_dir / 'kvasir.csv'}",
            f"paths.kvasir_cache={cache / 'kvasir_224'}",
            f"paths.external_index={self.index_dir / 'cv2024_external.csv'}",
            f"paths.external_cache={cache / 'cv2024_224'}",
            f"paths.class_mapping={REPO / 'configs' / 'class_mapping.yaml'}",
            f"paths.runs_root={self.runs}",
        ]


# --------------------------------------------------------------------------- helpers


def note(stage: Stage, message: str) -> None:
    stage.notes.append(message)
    print(f"    {message}", flush=True)


def warn(stage: Stage, message: str) -> None:
    stage.status = "WARN"
    note(stage, f"WARN: {message}")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise StageFailed(message)


def run_cxa(ctx: Context, stage: Stage, *args: str) -> str:
    """Run one ``cxa`` command; its full output goes to <work>/logs/<stage>.log."""
    log = ctx.work / "logs" / f"{stage.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "capsule_xai_audit", *args]
    with log.open("a") as fh:
        fh.write(f"$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = log.read_text().strip().splitlines()[-6:]
        raise StageFailed(
            f"`cxa {args[0]}` exited with {proc.returncode}; last lines of {log}:\n      "
            + "\n      ".join(tail)
        )
    return log.read_text()


def safe_extract_zip(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                raise StageFailed(f"{archive.name}: unsafe member path {name!r}")
        zf.extractall(target)


def images_below(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()]


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        target.write_bytes(response.read())


# --------------------------------------------------------------------------- stages


def stage_extract(ctx: Context, stage: Stage) -> None:
    raw = ctx.args.raw_dir
    k_root = raw / "kvasir" / "labelled_images"
    if len(images_below(k_root)) != KVASIR_LABELLED_IMAGES:
        check(ctx.args.kvasir_zip.exists(), f"Kvasir zip not found: {ctx.args.kvasir_zip}")
        note(stage, f"extracting {ctx.args.kvasir_zip.name}")
        safe_extract_zip(ctx.args.kvasir_zip, raw / "kvasir")
        from capsule_xai_audit.experiments.prepare import unpack_class_archives

        unpack_class_archives(k_root, k_root)
    else:
        note(stage, f"Kvasir already extracted in {k_root}")
    ctx.kvasir_images = k_root

    cv_dataset = raw / "capsule_vision" / "Dataset"
    if len(images_below(cv_dataset)) != sum(CV_FRAMES.values()):
        check(ctx.args.cv_zip.exists(), f"Capsule Vision zip not found: {ctx.args.cv_zip}")
        note(stage, f"extracting {ctx.args.cv_zip.name}")
        safe_extract_zip(ctx.args.cv_zip, raw / "capsule_vision")
    else:
        note(stage, f"Capsule Vision already extracted in {cv_dataset}")
    ctx.cv_root = cv_dataset

    metadata = ctx.args.metadata
    if not metadata.exists():
        from capsule_xai_audit.experiments.prepare import fetch_kvasir_metadata

        note(stage, f"metadata.csv is not part of the zip; downloading it from {METADATA_URL}")
        fetch_kvasir_metadata(metadata)
    ctx.kvasir_metadata = metadata


def stage_structure(ctx: Context, stage: Stage) -> None:
    assert ctx.kvasir_images is not None
    assert ctx.cv_root is not None
    assert ctx.kvasir_metadata is not None
    folders = sorted(d.name for d in ctx.kvasir_images.iterdir() if d.is_dir())
    files = images_below(ctx.kvasir_images)
    note(stage, f"Kvasir: {len(folders)} class folders, {len(files)} images")
    check(
        len(folders) == KVASIR_CLASS_FOLDERS,
        f"expected {KVASIR_CLASS_FOLDERS} class folders, found {folders}",
    )
    check(
        len(files) == KVASIR_LABELLED_IMAGES, f"expected {KVASIR_LABELLED_IMAGES} images, found {len(files)}"
    )
    by_stem: dict[str, list[Path]] = collections.defaultdict(list)
    for p in files:
        by_stem[p.stem].append(p)
    dups = {s: ps for s, ps in by_stem.items() if len(ps) > 1}
    if dups:
        identical = all(len({p.read_bytes() for p in ps}) == 1 for ps in dups.values())
        pairs = collections.Counter(tuple(sorted(p.parent.name for p in ps)) for ps in dups.values())
        warn(
            stage,
            f"{len(dups)} Kvasir file names are in two class folders "
            f"(byte-identical: {identical}): {dict(pairs)}",
        )

    meta = pd.read_csv(ctx.kvasir_metadata, sep=None, engine="python")
    boxes = meta[[f"{a}{i}" for i in range(1, 5) for a in "xy"]].notna().all(axis=1)
    note(stage, f"metadata.csv: {len(meta)} rows, {int(boxes.sum())} with a bounding box")
    check(int(boxes.sum()) > 4000, "metadata.csv has fewer than 4000 boxes")

    cv_files = images_below(ctx.cv_root)
    parts = [p.relative_to(ctx.cv_root).parts for p in cv_files]
    check(all(len(x) == 4 for x in parts), "Capsule Vision layout is not <split>/<class>/<source>/<file>")
    per_split = collections.Counter(x[0] for x in parts)
    kv_split = collections.Counter(x[0] for x in parts if x[2] == "KVASIR")
    note(stage, f"Capsule Vision: {dict(per_split)}; KVASIR-source frames {dict(kv_split)}")
    check(dict(per_split) == CV_FRAMES, f"Capsule Vision frame counts {dict(per_split)} != {CV_FRAMES}")
    check(dict(kv_split) == CV_KVASIR_FRAMES, f"KVASIR-source counts {dict(kv_split)} != {CV_KVASIR_FRAMES}")
    names = collections.Counter((x[0], x[3]) for x in parts)
    multi = sum(1 for n in names.values() if n > 1)
    if multi:
        warn(
            stage,
            f"{multi} Capsule Vision images sit in two class folders under one name (multi-label frames)",
        )


def stage_splits(ctx: Context, stage: Stage) -> None:
    split_dir = ctx.work / "official_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for s in (0, 1):
        target = split_dir / f"split_{s}.csv"
        if not target.exists():
            download(SPLIT_URL.format(s), target)
    frames = []
    for s in (0, 1):
        df = pd.read_csv(split_dir / f"split_{s}.csv")
        check(list(df.columns) == ["filename", "label"], f"split_{s}.csv header is {list(df.columns)}")
        frames.append(df.assign(fold=s))
    sp = pd.concat(frames, ignore_index=True)
    sp["stem"] = sp["filename"].map(lambda f: Path(f).stem)
    sp["video"] = sp["stem"].str.rsplit("_", n=1).str[0]
    note(stage, f"{len(sp)} rows, {sp['stem'].nunique()} distinct frames, {sp['video'].nunique()} videos")
    repeated = sp[sp["stem"].duplicated(keep=False)]
    if len(repeated):
        combos = repeated.groupby("stem")["label"].agg(lambda s: "+".join(sorted(s))).value_counts().to_dict()
        warn(stage, f"{repeated['stem'].nunique()} frames are listed twice with different labels: {combos}")
    folds = sp.groupby("video")["fold"].nunique()
    shared = folds[folds > 1].index
    if len(shared):
        n = int(sp["video"].isin(shared).sum())
        per_pair = sp[sp["video"].isin(shared)].groupby(["video", "label"])["fold"].nunique().max()
        warn(
            stage,
            f"{len(shared)} of {sp['video'].nunique()} videos ({n} frames) are in both official folds; "
            f"within them each (video, class) pair stays in one fold: {per_pair == 1}",
        )


def stage_index(ctx: Context, stage: Stage) -> None:
    assert ctx.kvasir_images is not None
    assert ctx.kvasir_metadata is not None
    assert ctx.cv_root is not None
    run_cxa(
        ctx,
        stage,
        "index-kvasir",
        "--images",
        str(ctx.kvasir_images),
        "--splits",
        str(ctx.work / "official_splits"),
        "--metadata",
        str(ctx.kvasir_metadata),
        "--out",
        str(ctx.index_dir / "kvasir.csv"),
        "--manifests",
        str(ctx.work / "manifests"),
    )
    k = pd.read_csv(ctx.index_dir / "kvasir.csv")
    multi = pd.read_csv(ctx.work / "manifests" / "kvasir_multilabel_excluded.csv")
    rows = pd.concat(pd.read_csv(ctx.work / "official_splits" / f"split_{s}.csv") for s in (0, 1))
    expected = rows["filename"].map(lambda f: Path(f).stem).nunique() - multi["image_id"].nunique()
    note(
        stage,
        f"{multi['image_id'].nunique()} two-label frames excluded (listed in kvasir_multilabel_excluded.csv)",
    )
    check(len(k) == expected, f"Kvasir index has {len(k)} frames, expected {expected}")
    note(
        stage,
        f"Kvasir index: {len(k)} frames, folds {k.groupby('official_fold').size().to_dict()}, "
        f"{int((k['bboxes'] != '[]').sum())} with boxes",
    )
    check(len(k) > 47_000, "Kvasir index has fewer than 47,000 frames")
    check(int((k["bboxes"] != "[]").sum()) > 4000, "fewer than 4000 indexed frames carry a box")
    run_cxa(ctx, stage, "index-cv", "--root", str(ctx.cv_root), "--out", str(ctx.index_dir / "cv2024.csv"))
    cv = pd.read_csv(ctx.index_dir / "cv2024.csv")
    check(len(cv) == sum(CV_FRAMES.values()), f"Capsule Vision index has {len(cv)} frames")


def stage_cache(ctx: Context, stage: Stage) -> None:
    from capsule_xai_audit.data.cache import open_cache

    for name, index in (("kvasir_224", "kvasir.csv"), ("cv2024_224", "cv2024.csv")):
        run_cxa(
            ctx,
            stage,
            "cache",
            "--index",
            str(ctx.index_dir / index),
            "--out",
            str(ctx.args.cache_dir / name),
            "--size",
            "224",
            "--workers",
            str(ctx.args.workers),
        )
        ids = pd.read_csv(ctx.index_dir / index)["image_id"].tolist()
        pixels, sizes = open_cache(ctx.args.cache_dir / name, ids)
        note(
            stage,
            f"{name}: {pixels.shape}, original sizes {sorted({tuple(s) for s in sizes[::500]})}",
        )
        sample = np.asarray(pixels[:: max(1, len(pixels) // 200)])
        check(sample.std() > 10, f"{name} looks blank (pixel std {sample.std():.1f})")


def stage_dedup(ctx: Context, stage: Stage) -> None:
    run_cxa(
        ctx,
        stage,
        "dedup",
        "--kvasir-index",
        str(ctx.index_dir / "kvasir.csv"),
        "--kvasir-cache",
        str(ctx.args.cache_dir / "kvasir_224"),
        "--cv-index",
        str(ctx.index_dir / "cv2024.csv"),
        "--cv-cache",
        str(ctx.args.cache_dir / "cv2024_224"),
        "--out",
        str(ctx.index_dir / "cv2024_external.csv"),
        "--manifests",
        str(ctx.work / "manifests"),
    )
    cv = pd.read_csv(ctx.index_dir / "cv2024.csv")
    kv_ids = set(pd.read_csv(ctx.index_dir / "kvasir.csv")["image_id"])
    check("source" in cv and cv["source"].notna().all(), "Capsule Vision index has no source column")
    cv["stem"] = cv["image_id"].map(lambda s: s.rsplit("/", 1)[1])
    kvasir_source = set(cv.loc[cv["source"] == "KVASIR", "image_id"])
    # Answer key for image matching: KVASIR-source frames whose original frame is indexed.
    truth = set(cv.loc[(cv["source"] == "KVASIR") & cv["stem"].isin(kv_ids), "image_id"])
    dups = pd.read_csv(ctx.work / "manifests" / "cv2024_kvasir_duplicates.csv")
    found = set(dups["cv_image_id"])
    recall = len(found & truth) / max(1, len(truth))
    precision = len(found & kvasir_source) / max(1, len(found))
    note(
        stage,
        f"image matching vs source folders: recall {recall:.4f} ({len(found & truth)} of {len(truth)} "
        f"indexed Kvasir frames), precision {precision:.4f} ({len(found)} flagged)",
    )
    summary = json.loads((ctx.work / "manifests" / "cv2024_dedup_summary.json").read_text())
    note(stage, f"two-label images excluded: {summary['multilabel_images_excluded']}")
    external = pd.read_csv(ctx.index_dir / "cv2024_external.csv")
    leaked = int((external["source"] == "KVASIR").sum())
    note(
        stage,
        f"external set: {len(external)} frames, by source {external['source'].value_counts().to_dict()}, "
        f"by label {external['label'].value_counts().to_dict()}",
    )
    check(leaked == 0, f"{leaked} KVASIR-source frames are still in the external set")
    check(not set(external["image_id"]) & found, "image-matched Kvasir frames are still in the external set")
    check(recall > 0.99, f"image matching found only {recall:.1%} of the indexed Kvasir frames")
    if precision < 0.999:
        warn(stage, f"{len(found - kvasir_source)} non-KVASIR frames matched a Kvasir frame by image")


def stage_train(ctx: Context, stage: Stage) -> None:
    from capsule_xai_audit.eval.predict import Predictions

    for protocol in ("official", "random", "video"):
        for seed in (0, 1):
            run_cxa(
                ctx,
                stage,
                "train",
                "--set",
                *ctx.path_overrides(),
                f"train.backbone={ctx.args.backbone}",
                f"train.split_protocol={protocol}",
                f"train.seed={seed}",
                "train.epochs=1",
                f"train.max_steps_per_epoch={ctx.args.train_steps}",
                f"train.num_workers={ctx.args.workers}",
            )
    run = ctx.runs / "train" / "official" / ctx.args.backbone / "base" / "fold1" / "seed0"
    meta = json.loads((run / "run.json").read_text())
    check(meta["complete"], "run.json is not marked complete")
    for split in ("val", "test", "external"):
        p = Predictions.load(run / f"preds_{split}.npz")
        check(np.isfinite(p.logits).all(), f"non-finite logits in preds_{split}")
        note(stage, f"preds_{split}: {p.logits.shape}")
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    note(stage, f"1 epoch x {ctx.args.train_steps} steps: {history[-1]['seconds']:.0f} s incl. validation")


def stage_q1_q2(ctx: Context, stage: Stage) -> None:
    run_cxa(
        ctx,
        stage,
        "leakage",
        "--runs-root",
        str(ctx.runs),
        "--backbone",
        ctx.args.backbone,
        "--n-boot",
        "100",
    )
    run_cxa(
        ctx,
        stage,
        "calibration",
        "--runs-root",
        str(ctx.runs),
        "--backbone",
        ctx.args.backbone,
        "--n-boot",
        "100",
        "--class-mapping",
        str(REPO / "configs" / "class_mapping.yaml"),
    )
    q1 = json.loads((ctx.runs / "analysis" / f"q1_leakage_{ctx.args.backbone}_base_fold1.json").read_text())
    check(q1["protocols"] == ["random", "official", "video"], f"Q1 protocols: {q1['protocols']}")
    q2 = json.loads(
        (ctx.runs / "analysis" / f"q2_calibration_{ctx.args.backbone}_base_fold1.json").read_text()
    )
    check({"id_all", "id_shared", "external"} <= set(q2["domains"]), f"Q2 domains: {list(q2['domains'])}")


def stage_fewshot(ctx: Context, stage: Stage) -> None:
    run_cxa(
        ctx,
        stage,
        "fewshot",
        "--set",
        *ctx.path_overrides(),
        "fewshot.backbones=[dinov2_vits14, resnet50]",
        "fewshot.shots=[1, 5]",
        "fewshot.draws=2",
        "fewshot.C_sensitivity=[]",
        f"fewshot.num_workers={ctx.args.workers}",
    )


def stage_xai(ctx: Context, stage: Stage) -> None:
    run_cxa(
        ctx,
        stage,
        "xai",
        "--set",
        *ctx.path_overrides(),
        f"xai.run=train/official/{ctx.args.backbone}/base/fold1/seed0",
        "xai.max_images=16",
        "xai.sanity_images=4",
        "xai.ig_steps=8",
        "xai.perturbation_steps=10",
        "xai.n_bootstrap=50",
        "xai.batch_size=8",
    )


def stage_superpixel(ctx: Context, stage: Stage) -> None:
    run_cxa(
        ctx,
        stage,
        "superpixel",
        "--set",
        *ctx.path_overrides(),
        "superpixel.max_images=16",
        "superpixel.n_bootstrap=50",
        f"superpixel.num_workers={ctx.args.workers}",
    )


def stage_color(ctx: Context, stage: Stage) -> None:
    run_cxa(
        ctx,
        stage,
        "train",
        "--set",
        *ctx.path_overrides(),
        f"train.backbone={ctx.args.backbone}",
        "train.seed=0",
        "train.tag=lab-aug",
        "train.color_aug_max_delta_e=20",
        "train.epochs=1",
        f"train.max_steps_per_epoch={ctx.args.train_steps}",
        f"train.num_workers={ctx.args.workers}",
    )
    b = ctx.args.backbone
    run_cxa(
        ctx,
        stage,
        "color",
        "--set",
        *ctx.path_overrides(),
        f"color.runs=[train/official/{b}/base/fold1/seed0, train/official/{b}/lab-aug/fold1/seed0]",
        "color.delta_e=[0.0, 10.0]",
        "color.hue_angles=[0.0, 180.0]",
        "color.max_images=256",
        f"color.num_workers={ctx.args.workers}",
    )


def stage_report(ctx: Context, stage: Stage) -> None:
    run_cxa(ctx, stage, "report", "--runs-root", str(ctx.runs), "--out", str(ctx.work / "results"))
    text = (ctx.work / "results" / "RESULTS.md").read_text()
    missing = [h for h in ("## Q1", "## Q2", "## Q3", "## Q4", "## Slice A", "## Slice B") if h not in text]
    check(not missing, f"report is missing sections {missing}")


STAGES: list[tuple[str, Callable[[Context, Stage], None], tuple[str, ...]]] = [
    ("extract", stage_extract, ()),
    ("structure", stage_structure, ("extract",)),
    ("splits", stage_splits, ()),
    ("index", stage_index, ("extract", "splits")),
    ("cache", stage_cache, ("index",)),
    ("dedup", stage_dedup, ("cache",)),
    ("train", stage_train, ("dedup",)),
    ("q1_q2", stage_q1_q2, ("train",)),
    ("fewshot_q3", stage_fewshot, ("cache",)),
    ("xai_q4", stage_xai, ("train",)),
    ("slice_a", stage_superpixel, ("fewshot_q3",)),
    ("slice_b", stage_color, ("train",)),
    ("report", stage_report, ("q1_q2", "fewshot_q3", "xai_q4", "slice_a", "slice_b")),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--kvasir-zip", type=Path, default=REPO / "data" / "kvasir-capsule-labeled-images.zip"
    )
    parser.add_argument("--cv-zip", type=Path, default=REPO / "data" / "Capsule Vision 2024.zip")
    parser.add_argument(
        "--raw-dir", type=Path, default=REPO / "data" / "raw", help="where the zips are extracted"
    )
    parser.add_argument("--metadata", type=Path, default=REPO / "data" / "raw" / "kvasir" / "metadata.csv")
    parser.add_argument("--work", type=Path, default=REPO / "data" / "smoke")
    parser.add_argument("--cache-dir", type=Path, default=REPO / "cache")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fresh", action="store_true", help="delete --work first (caches are kept)")
    args = parser.parse_args()

    if args.fresh and args.work.exists():
        shutil.rmtree(args.work)
    args.work.mkdir(parents=True, exist_ok=True)
    ctx = Context(args=args, work=args.work)

    for name, fn, deps in STAGES:
        stage = ctx.stages.setdefault(name, Stage(name))
        blocked = [d for d in deps if ctx.stages[d].status in ("FAIL", "SKIP")]
        print(f"\n== {name}", flush=True)
        if blocked:
            stage.status = "SKIP"
            note(stage, f"skipped: depends on {', '.join(blocked)}")
            continue
        start = time.perf_counter()
        try:
            fn(ctx, stage)
            if stage.status == "PENDING":
                stage.status = "PASS"
        except StageFailed as err:
            stage.status = "FAIL"
            note(stage, f"FAIL: {err}")
        except Exception:  # a smoke test reports every crash and carries on
            stage.status = "FAIL"
            note(stage, "FAIL: unexpected error\n" + traceback.format_exc())
        stage.seconds = time.perf_counter() - start

    print("\n" + "=" * 72)
    for stage in ctx.stages.values():
        print(f"{stage.status:5}  {stage.name:12} {stage.seconds:7.0f} s")
    summary = {
        s.name: {"status": s.status, "seconds": round(s.seconds), "notes": s.notes}
        for s in ctx.stages.values()
    }
    (ctx.work / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"details: {ctx.work / 'smoke_summary.json'}, logs: {ctx.work / 'logs'}")
    return 1 if any(s.status == "FAIL" for s in ctx.stages.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
