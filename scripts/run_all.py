#!/usr/bin/env python3
"""Run the whole capsule-xai-audit pipeline in one go, locally or on Kaggle.

Phases, in order; each stops the run if it fails:

1. preflight  GPUs (with an fp16 test on each), CPUs, disk space, packages, pretrained
              weights (downloaded now, so a network problem shows up at minute one) and
              the input data. Unit tests run when pytest is installed.
2. prepare    Finds the inputs (folders, per-class archives or zips), downloads the official
              split and metadata.csv when missing, builds the indices and 224 px caches and
              the external set. Nothing is assumed to exist; finished steps are reused.
3. smoke      A tiny version of every experiment on the real data, through the same
              scheduler and on all GPUs, so a bug costs minutes, not hours.
4. full       All experiments of the chosen profile. Training jobs get one GPU each; with
              >= 12 GiB per GPU a light job (few-shot, XAI, slices) shares a GPU next to a
              training job; analyses run on the CPU meanwhile. Order favours the jobs that
              unblock others.

Re-running continues where it stopped: finished training runs are skipped and an analysis
is redone only when one of its inputs changed. On Kaggle no new job starts after
``--time-budget-hours`` (default 11 h), so the notebook ends inside the 12 h limit and its
output is saved; attach that output to the next run and it continues.

    # local (inputs found below data/)
    python scripts/run_all.py --profile full
    # Kaggle notebook cell (inputs found below /kaggle/input)
    !python /kaggle/working/capsule-xai-audit/scripts/run_all.py --profile full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# CXA_KAGGLE_ROOT / CXA_TMP point Kaggle mode at a simulated /kaggle and /tmp (for testing).
KAGGLE_ROOT = Path(os.environ.get("CXA_KAGGLE_ROOT", "/kaggle"))
KAGGLE_INPUT = KAGGLE_ROOT / "input"
IS_KAGGLE = KAGGLE_INPUT.is_dir()
TMP = Path(os.environ.get("CXA_TMP", "/tmp"))

# Published numbers of the releases (see README, Implementation notes).
EXPECTED_KVASIR_INDEX = 47_145
EXPECTED_CV_FRAMES = 53_739

PROFILES: dict[str, dict[str, object]] = {
    # Pre-registered minimum: Q1 and Q2, 3 seeds.
    "minimum": {
        "backbones": ["resnet50"],
        "protocols": ["official", "random"],
        "seeds": [0, 1, 2],
        "fewshot": False,
        "xai": False,
        "slice_a": False,
        "slice_b": False,
    },
    # Core: Q1-Q4.
    "core": {
        "backbones": ["resnet50", "convnext_tiny"],
        "protocols": ["official", "random", "video"],
        "seeds": [0, 1, 2, 3, 4],
        "fewshot": True,
        "xai": True,
        "slice_a": False,
        "slice_b": False,
    },
    # Core plus both slices.
    "full": {
        "backbones": ["resnet50", "convnext_tiny"],
        "protocols": ["official", "random", "video"],
        "seeds": [0, 1, 2, 3, 4],
        "fewshot": True,
        "xai": True,
        "slice_a": True,
        "slice_b": True,
    },
}
FEWSHOT_BACKBONES = ["dinov2_vits14", "resnet50", "convnext_tiny"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class PhaseFailed(Exception):
    pass


# --------------------------------------------------------------------------- layout


@dataclass
class Layout:
    work: Path
    index: Path
    cache: Path
    runs: Path
    manifests: Path
    results: Path
    logs: Path
    smoke: Path
    scratch: Path

    @property
    def splits(self) -> Path:
        return self.index / "official_splits"

    def path_overrides(self, runs: Path | None = None) -> list[str]:
        return [
            f"paths.kvasir_index={self.index / 'kvasir.csv'}",
            f"paths.kvasir_cache={self.cache / 'kvasir_224'}",
            f"paths.external_index={self.index / 'cv2024_external.csv'}",
            f"paths.external_cache={self.cache / 'cv2024_224'}",
            f"paths.class_mapping={REPO / 'configs' / 'class_mapping.yaml'}",
            f"paths.runs_root={runs or self.runs}",
        ]


def make_layout(args: argparse.Namespace) -> Layout:
    if IS_KAGGLE:
        work = args.work or KAGGLE_ROOT / "working"
        scratch = TMP / "cxa"  # not part of the notebook output (20 GB limit)
        layout = Layout(
            work=work,
            index=work / "index",
            cache=args.cache_dir or scratch / "cache",
            runs=work / "runs",
            manifests=work / "manifests",
            results=work / "results",
            logs=work / "logs",
            smoke=scratch / "smoke",
            scratch=scratch,
        )
    else:
        work = args.work or REPO
        layout = Layout(
            work=work,
            index=work / "data" / "index",
            cache=args.cache_dir or work / "cache",
            runs=work / "runs",
            manifests=work / "manifests",
            results=work / "results",
            logs=work / "data" / "logs",
            smoke=work / "data" / "smoke_run",
            scratch=work / "data" / "raw",
        )
    for d in (layout.index, layout.cache, layout.runs, layout.manifests, layout.logs, layout.scratch):
        d.mkdir(parents=True, exist_ok=True)
    return layout


# --------------------------------------------------------------------------- small helpers


def cxa(args: list[str], log_file: Path, env: dict[str, str] | None = None) -> None:
    """Run ``cxa <args>`` synchronously; raise PhaseFailed with the log tail on failure."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a") as fh:
        fh.write(f"\n$ cxa {' '.join(args)}\n")
        fh.flush()
        proc = subprocess.run(
            [sys.executable, "-m", "capsule_xai_audit", *args],
            cwd=REPO,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})},
        )
    if proc.returncode != 0:
        raise PhaseFailed(
            f"`cxa {args[0]}` failed (exit {proc.returncode}); last lines of {log_file}:\n" + tail(log_file)
        )


def tail(path: Path, n: int = 12) -> str:
    try:
        lines = path.read_text(errors="replace").strip().splitlines()[-n:]
    except OSError:
        return ""
    return "\n".join("    " + line for line in lines)


def sha256_of(*paths: Path) -> str:
    digest = hashlib.sha256()
    for p in paths:
        digest.update(p.read_bytes())
    return digest.hexdigest()


def free_gib(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 2**30


def walk_dirs(roots: list[Path], max_depth: int = 6) -> Iterator[Path]:
    for root in roots:
        if not root.is_dir():
            continue
        base = len(root.parts)
        for current, dirs, _ in os.walk(root, followlinks=True):
            path = Path(current)
            if len(path.parts) - base >= max_depth:
                dirs[:] = []
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            yield path


def has_images(folder: Path) -> bool:
    return any(p.suffix.lower() in (".jpg", ".jpeg", ".png") for p in folder.iterdir() if p.is_file())


# --------------------------------------------------------------------------- input discovery


@dataclass
class Inputs:
    kvasir_images: Path | None = None
    kvasir_archives: Path | None = None
    kvasir_zip: Path | None = None
    metadata: Path | None = None
    cv_root: Path | None = None
    cv_zip: Path | None = None
    prepared: Path | None = None  # a folder holding index/ and cache/ from an earlier run
    previous_runs: Path | None = None


def search_roots(args: argparse.Namespace) -> list[Path]:
    if args.input_root:
        return [args.input_root]
    return [KAGGLE_INPUT] if IS_KAGGLE else [REPO / "data"]


def find_inputs(args: argparse.Namespace, layout: Layout) -> Inputs:
    found = Inputs(kvasir_images=args.kvasir_images, metadata=args.metadata, cv_root=args.cv_root)
    roots = search_roots(args)
    for d in walk_dirs(roots):
        name = d.name.lower()
        if (
            found.kvasir_images is None
            and found.kvasir_archives is None
            and name in ("labelled_images", "labeled_images")
        ):
            subdirs = [s for s in d.iterdir() if s.is_dir()]
            if len(subdirs) >= 10 and all(has_images(s) for s in subdirs):
                found.kvasir_images = d  # already unpacked: one folder of images per class
            elif any(d.glob("*.tar.gz")):
                found.kvasir_archives = d
        if (
            found.cv_root is None
            and (d / "validation").is_dir()
            and (d / "training" / "Angioectasia").is_dir()
        ):
            found.cv_root = d
        if (
            found.prepared is None
            and (d / "cache" / "kvasir_224.json").is_file()
            and (d / "cache" / "cv2024_224.json").is_file()
            and (d / "index" / "cv2024_external.csv").is_file()
            and d.resolve() != layout.work.resolve()
        ):
            found.prepared = d
        if (
            IS_KAGGLE
            and found.previous_runs is None
            and d.name == "runs"
            and (d / "train").is_dir()
            and d.resolve() != layout.runs.resolve()
        ):
            found.previous_runs = d
    # Unpacked archives from an earlier run on this machine.
    unpacked = layout.scratch / "kvasir_images"
    if found.kvasir_images is None and unpacked.is_dir() and any(unpacked.glob(".*.done")):
        found.kvasir_images = unpacked
    if found.metadata is None:
        for root in roots:
            for p in root.rglob("metadata.csv") if root.is_dir() else []:
                if "x1" in p.open(errors="replace").readline().lower():
                    found.metadata = p
                    break
            if found.metadata:
                break
    if found.metadata is None and (layout.index / "metadata.csv").is_file():
        found.metadata = layout.index / "metadata.csv"
    # Zips as a last resort (Kaggle unpacks uploaded zips itself, locally they may be zipped).
    for root in roots:
        for z in root.rglob("*.zip") if root.is_dir() else []:
            try:
                names = zipfile.ZipFile(z).namelist()[:200]
            except zipfile.BadZipFile:
                continue
            if found.kvasir_zip is None and any(
                "labelled_images/" in n and n.endswith(".tar.gz") for n in names
            ):
                found.kvasir_zip = z
            if found.cv_zip is None and any(n.startswith("Dataset/training/") for n in names):
                found.cv_zip = z
    return found


def safe_unzip(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                raise PhaseFailed(f"{archive.name}: unsafe member path {name!r}")
        zf.extractall(target)


# --------------------------------------------------------------------------- phase 1: preflight


@dataclass
class Check:
    name: str
    status: str
    detail: str


def gpu_info() -> list[tuple[str, float]]:
    import torch

    if not torch.cuda.is_available():
        return []
    return [
        (torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 2**30)
        for i in range(torch.cuda.device_count())
    ]


def preflight(args: argparse.Namespace, layout: Layout, plan: dict[str, object]) -> list[Check]:
    checks: list[Check] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append(Check(name, status, detail))
        log(f"  {status:4} {name}: {detail}")

    log("phase 1/4: preflight")
    try:
        import capsule_xai_audit  # noqa: F401
    except ImportError:
        log("  installing the package (pip install -e)")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)], check=True)
    import torch

    gpus = gpu_info()
    if gpus:
        for i, (name, mem) in enumerate(gpus):
            a = torch.randn(1024, 1024, device=f"cuda:{i}", dtype=torch.float16)
            ok = bool(torch.isfinite(a @ a).all())
            add(
                f"gpu{i}",
                "PASS" if ok else "FAIL",
                f"{name}, {mem:.1f} GiB, fp16 matmul {'ok' if ok else 'failed'}",
            )
    else:
        add("gpu", "WARN", "no CUDA GPU: everything runs on the CPU and will be very slow")
    add("cpu", "PASS", f"{os.cpu_count()} cores")

    have_cache = (layout.cache / "kvasir_224.json").is_file() and (layout.cache / "cv2024_224.json").is_file()
    need_cache = not have_cache and (args.rebuild or find_inputs(args, layout).prepared is None)
    cache_free = free_gib(layout.cache)
    add(
        "disk (cache)",
        "FAIL" if need_cache and cache_free < 17 else "PASS",
        f"{cache_free:.0f} GiB free at {layout.cache}"
        + (" (17 GiB needed to build the caches)" if need_cache else ""),
    )
    work_free = free_gib(layout.runs)
    add(
        "disk (runs)",
        "FAIL" if work_free < 6 else "PASS",
        f"{work_free:.0f} GiB free at {layout.runs} (6 GiB needed)",
    )

    for module in ("timm", "captum", "sklearn", "skimage", "pandas", "scipy"):
        try:
            __import__(module)
            add(f"package {module}", "PASS", "importable")
        except ImportError as err:
            add(f"package {module}", "FAIL", str(err))

    backbones = sorted(
        set(plan["backbones"]) | (set(FEWSHOT_BACKBONES) if plan["fewshot"] or plan["slice_a"] else set())
    )
    import timm

    from capsule_xai_audit.models.backbones import get_spec

    for b in backbones:
        try:
            kwargs = {"img_size": 224} if get_spec(b).kind == "vit" else {}
            timm.create_model(get_spec(b).timm_name, pretrained=True, num_classes=0, **kwargs)
            add(f"weights {b}", "PASS", "pretrained weights available")
        except Exception as err:  # network or hub failure
            add(f"weights {b}", "FAIL", f"cannot load pretrained weights: {err}")

    if args.unit_tests:
        try:
            import pytest  # noqa: F401

            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-o", "addopts="],
                cwd=REPO,
                capture_output=True,
                text=True,
            )
            last = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
            add("unit tests", "PASS" if proc.returncode == 0 else "FAIL", last)
        except ImportError:
            add("unit tests", "WARN", "pytest not installed; skipped")
    return checks


# --------------------------------------------------------------------------- phase 2: prepare


def prepare(args: argparse.Namespace, layout: Layout) -> None:
    log("phase 2/4: prepare data")
    plog = layout.logs / "prepare.log"
    inputs = find_inputs(args, layout)

    if inputs.previous_runs is not None and not any((layout.runs / "train").glob("*")):
        log(f"  copying earlier runs from {inputs.previous_runs}")
        shutil.copytree(inputs.previous_runs, layout.runs, dirs_exist_ok=True)

    if inputs.prepared is not None and not args.rebuild:
        log(f"  using prepared index and caches from {inputs.prepared}")
        for name in ("kvasir.csv", "cv2024.csv", "cv2024_external.csv"):
            shutil.copy2(inputs.prepared / "index" / name, layout.index / name)
        layout.cache = inputs.prepared / "cache"
        validate_prepared(layout)
        return

    # Kvasir-Capsule images: folder, per-class archives, or the zip.
    if inputs.kvasir_images is None and inputs.kvasir_archives is None and inputs.kvasir_zip is not None:
        log(f"  extracting {inputs.kvasir_zip.name}")
        safe_unzip(inputs.kvasir_zip, layout.scratch / "kvasir")
        inputs.kvasir_archives = layout.scratch / "kvasir" / "labelled_images"
    if inputs.kvasir_images is None and inputs.kvasir_archives is not None:
        target = layout.scratch / "kvasir_images"  # inputs may be read-only (Kaggle)
        log(f"  unpacking class archives from {inputs.kvasir_archives}")
        cxa(["unpack-kvasir", "--src", str(inputs.kvasir_archives), "--out", str(target)], plog)
        inputs.kvasir_images = target
    if inputs.kvasir_images is None:
        raise PhaseFailed(
            "Kvasir-Capsule images not found. Attach kvasir-capsule-labeled-images.zip "
            "(https://datasets.simula.no/downloads/kvasir-capsule/kvasir-capsule-labeled-images.zip) "
            "or pass --kvasir-images."
        )
    if inputs.cv_root is None and inputs.cv_zip is not None:
        log(f"  extracting {inputs.cv_zip.name}")
        safe_unzip(inputs.cv_zip, layout.scratch / "capsule_vision")
        inputs.cv_root = layout.scratch / "capsule_vision" / "Dataset"
    if inputs.cv_root is None:
        raise PhaseFailed(
            "Capsule Vision 2024 not found. Attach Dataset.zip "
            "(https://figshare.com/ndownloader/files/48018562) or pass --cv-root."
        )
    if inputs.metadata is None:
        log("  metadata.csv not attached; downloading it from OSF")
        inputs.metadata = layout.index / "metadata.csv"
        cxa(["fetch-metadata", "--out", str(inputs.metadata)], plog)
    log(f"  Kvasir images {inputs.kvasir_images}")
    log(f"  metadata      {inputs.metadata}")
    log(f"  Capsule Vision {inputs.cv_root}")

    if not (layout.splits / "split_0.csv").is_file() or not (layout.splits / "split_1.csv").is_file():
        cxa(["fetch-splits", "--out", str(layout.splits)], plog)
    cxa(
        [
            "index-kvasir",
            "--images",
            str(inputs.kvasir_images),
            "--splits",
            str(layout.splits),
            "--metadata",
            str(inputs.metadata),
            "--out",
            str(layout.index / "kvasir.csv"),
            "--manifests",
            str(layout.manifests),
        ],
        plog,
    )
    cxa(["index-cv", "--root", str(inputs.cv_root), "--out", str(layout.index / "cv2024.csv")], plog)
    workers = str(max(1, os.cpu_count() or 1))
    for name, index in (("kvasir_224", "kvasir.csv"), ("cv2024_224", "cv2024.csv")):
        log(f"  cache {name} (reused when it matches the index)")
        cxa(
            [
                "cache",
                "--index",
                str(layout.index / index),
                "--out",
                str(layout.cache / name),
                "--workers",
                workers,
            ],
            plog,
        )

    stamp = layout.index / ".dedup_inputs.sha256"
    digest = sha256_of(layout.index / "kvasir.csv", layout.index / "cv2024.csv")
    external = layout.index / "cv2024_external.csv"
    if external.is_file() and stamp.is_file() and stamp.read_text().strip() == digest:
        log("  external set is up to date")
    else:
        log("  building the external set (dedup)")
        cxa(
            [
                "dedup",
                "--kvasir-index",
                str(layout.index / "kvasir.csv"),
                "--kvasir-cache",
                str(layout.cache / "kvasir_224"),
                "--cv-index",
                str(layout.index / "cv2024.csv"),
                "--cv-cache",
                str(layout.cache / "cv2024_224"),
                "--out",
                str(external),
                "--manifests",
                str(layout.manifests),
            ],
            plog,
        )
        stamp.write_text(digest + "\n")
    validate_prepared(layout)


def validate_prepared(layout: Layout) -> None:
    import pandas as pd

    from capsule_xai_audit.data.cache import open_cache

    k = pd.read_csv(layout.index / "kvasir.csv")
    cv = pd.read_csv(layout.index / "cv2024.csv")
    ext = pd.read_csv(layout.index / "cv2024_external.csv")
    open_cache(layout.cache / "kvasir_224", k["image_id"].tolist())
    open_cache(layout.cache / "cv2024_224", cv["image_id"].tolist())
    for label, n, expected in (
        ("Kvasir index", len(k), EXPECTED_KVASIR_INDEX),
        ("Capsule Vision", len(cv), EXPECTED_CV_FRAMES),
    ):
        log(
            f"  {label}: {n} frames"
            + ("" if n == expected else f"  (WARNING: published release gives {expected})")
        )
    leaked = int((ext.get("source", pd.Series(dtype=str)) == "KVASIR").sum())
    log(f"  external set: {len(ext)} frames, KVASIR-source frames in it: {leaked}")
    if leaked:
        raise PhaseFailed("Kvasir-Capsule frames are left in the external set")


# --------------------------------------------------------------------------- scheduler


@dataclass
class Job:
    name: str
    args: list[str]
    kind: str  # "heavy" (training, one per GPU), "light" (GPU, may share), "cpu"
    deps: list[str] = field(default_factory=list)
    done: Callable[[], bool] | None = None  # already complete from an earlier run?
    output: Path | None = None  # analysis output; redone only when an input changed
    status: str = "pending"  # pending, running, ok, reused, failed, blocked, deferred
    seconds: float = 0.0
    gpu: int | None = None


class Scheduler:
    def __init__(
        self,
        jobs: list[Job],
        n_gpus: int,
        light_per_gpu: int,
        cpu_slots: int,
        logs: Path,
        status_file: Path,
        budget_s: float | None,
    ) -> None:
        self.jobs = {j.name: j for j in jobs}
        self.order = [j.name for j in jobs]
        self.n_gpus = max(1, n_gpus)
        self.has_gpu = n_gpus > 0
        self.light_per_gpu = light_per_gpu
        self.cpu_slots = cpu_slots
        self.logs = logs
        self.status_file = status_file
        self.budget_s = budget_s
        self.start = time.monotonic()
        self.running: dict[str, tuple[subprocess.Popen[bytes], float]] = {}

    # ------------------------------------------------------------ bookkeeping

    def _slots_free(self, job: Job) -> int | None:
        """GPU index (or -1 for CPU jobs) the job can start on, or None."""
        busy = [self.jobs[n] for n in self.running]
        if job.kind == "cpu":
            return -1 if sum(j.kind == "cpu" for j in busy) < self.cpu_slots else None
        heavy = [0] * self.n_gpus
        light = [0] * self.n_gpus
        for j in busy:
            if j.gpu is not None and j.gpu >= 0:
                (heavy if j.kind == "heavy" else light)[j.gpu] += 1
        if job.kind == "heavy":
            for g in range(self.n_gpus):
                # A heavy job may also take a GPU whose only occupant is a light job when sharing is off.
                if heavy[g] == 0 and light[g] <= self.light_per_gpu:
                    return g
            return None
        # light: a GPU with a free light slot, preferring idle GPUs.
        candidates = [g for g in range(self.n_gpus) if light[g] < max(1, self.light_per_gpu)]
        if self.light_per_gpu == 0:
            candidates = [g for g in candidates if heavy[g] == 0]
        if not candidates:
            return None
        return min(candidates, key=lambda g: (heavy[g], light[g]))

    def _deps_state(self, job: Job) -> str:
        states = [self.jobs[d].status for d in job.deps if d in self.jobs]
        if any(s in ("failed", "blocked", "deferred") for s in states):
            return "blocked"
        if all(s in ("ok", "reused") for s in states):
            return "ready"
        return "wait"

    def _write_status(self) -> None:
        data = {
            "elapsed_s": round(time.monotonic() - self.start),
            "jobs": {
                n: {"status": j.status, "seconds": round(j.seconds), "gpu": j.gpu}
                for n, j in self.jobs.items()
            },
        }
        self.status_file.write_text(json.dumps(data, indent=2) + "\n")

    def _eta(self) -> str:
        heavy_done = [j.seconds for j in self.jobs.values() if j.kind == "heavy" and j.status == "ok"]
        left = sum(1 for j in self.jobs.values() if j.kind == "heavy" and j.status in ("pending", "running"))
        if not heavy_done or not left:
            return ""
        eta = sum(heavy_done) / len(heavy_done) * left / self.n_gpus
        return f", about {eta / 3600:.1f} h of training left"

    # ------------------------------------------------------------ run

    def _launch(self, job: Job, gpu: int) -> None:
        env = dict(os.environ)
        if job.kind == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""  # analyses never need a GPU
            job.gpu = None
        elif self.has_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            job.gpu = gpu
        else:
            job.gpu = 0  # no GPU: one "slot" that runs on the CPU
        logfile = self.logs / f"{job.name.replace(':', '_').replace('/', '_')}.log"
        fh = logfile.open("w")
        fh.write(f"$ cxa {' '.join(job.args)}\n")
        fh.flush()
        proc = subprocess.Popen(
            [sys.executable, "-m", "capsule_xai_audit", *job.args],
            cwd=REPO,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        fh.close()
        self.running[job.name] = (proc, time.monotonic())
        job.status = "running"
        where = "cpu" if job.gpu is None else f"gpu{job.gpu}"
        log(f"  start  {job.name:44} on {where}")

    def run(self) -> bool:
        try:
            while True:
                progressed = self._poll()
                started = self._start_ready()
                if not self.running and not started:
                    pending = [j for j in self.jobs.values() if j.status == "pending"]
                    for j in pending:
                        j.status = "blocked" if self._deps_state(j) == "blocked" else "deferred"
                    break
                if not progressed and not started:
                    time.sleep(2)
        except KeyboardInterrupt:
            log("  interrupted; stopping running jobs")
            for proc, _ in self.running.values():
                os.killpg(proc.pid, signal.SIGTERM)
            raise
        finally:
            self._write_status()
        return all(j.status in ("ok", "reused") for j in self.jobs.values())

    def _poll(self) -> bool:
        progressed = False
        for name, (proc, started) in list(self.running.items()):
            code = proc.poll()
            if code is None:
                continue
            job = self.jobs[name]
            job.seconds = time.monotonic() - started
            job.status = "ok" if code == 0 else "failed"
            del self.running[name]
            progressed = True
            done = sum(j.status in ("ok", "reused") for j in self.jobs.values())
            log(
                f"  {'done ' if code == 0 else 'FAIL '} {name:44} {job.seconds / 60:5.1f} min  "
                f"({done}/{len(self.jobs)}{self._eta()})"
            )
            if code != 0:
                logfile = self.logs / f"{name.replace(':', '_').replace('/', '_')}.log"
                log("    last lines:\n" + tail(logfile, 8))
            self._write_status()
        return progressed

    def _start_ready(self) -> bool:
        started = False
        over_budget = self.budget_s is not None and time.monotonic() - self.start > self.budget_s
        for name in self.order:
            job = self.jobs[name]
            if job.status != "pending":
                continue
            state = self._deps_state(job)
            if state == "blocked":
                job.status = "blocked"
                log(f"  skip   {name:44} (an input failed)")
                continue
            if state != "ready":
                continue
            inputs_reused = all(self.jobs[d].status == "reused" for d in job.deps if d in self.jobs)
            if (job.done and job.done()) or (job.output and job.output.is_file() and inputs_reused):
                job.status = "reused"
                started = True
                continue
            if over_budget:
                continue
            slot = self._slots_free(job)
            if slot is None:
                continue
            self._launch(job, slot)
            started = True
        return started


# --------------------------------------------------------------------------- job plans


def train_done(runs: Path, protocol: str, backbone: str, tag: str, seed: int) -> Callable[[], bool]:
    meta = runs / "train" / protocol / backbone / tag / "fold1" / f"seed{seed}" / "run.json"

    def done() -> bool:
        # A run from an older split algorithm is not done: the train job replaces it.
        # (The job itself also compares the full training config before skipping.)
        try:
            from capsule_xai_audit.data.splits import SPLIT_VERSION

            run = json.loads(meta.read_text())
        except (ImportError, OSError, ValueError):
            return False
        return bool(run.get("complete")) and run.get("split_version") == SPLIT_VERSION

    return done


def build_jobs(plan: dict[str, object], layout: Layout, runs: Path, smoke: bool, workers: int) -> list[Job]:
    backbones: list[str] = list(plan["backbones"])  # type: ignore[arg-type]
    protocols: list[str] = list(plan["protocols"])  # type: ignore[arg-type]
    seeds: list[int] = list(plan["seeds"])  # type: ignore[arg-type]
    paths = layout.path_overrides(runs)
    analysis = runs / "analysis"
    tiny_train = (
        ["train.epochs=1", "train.max_steps_per_epoch=15", "train.log_every_steps=5"] if smoke else []
    )
    # The smoke test needs two seeds only where an ensemble is formed (Q2, main backbone).
    other_seeds = seeds[:1] if smoke else seeds
    n_boot = "50" if smoke else "2000"
    jobs: list[Job] = []
    main_cnn = "resnet50" if "resnet50" in backbones else backbones[0]

    def train(protocol: str, backbone: str, seed: int, tag: str = "base", extra: tuple[str, ...] = ()) -> Job:
        return Job(
            name=f"train:{protocol}:{backbone}:{tag}:s{seed}",
            args=[
                "train",
                "--set",
                *paths,
                f"train.backbone={backbone}",
                f"train.split_protocol={protocol}",
                f"train.seed={seed}",
                f"train.tag={tag}",
                f"train.num_workers={workers}",
                *tiny_train,
                *extra,
            ],
            kind="heavy",
            done=train_done(runs, protocol, backbone, tag, seed),
        )

    # Order = start priority: runs that unblock XAI, Q2 and Slice B first.
    trains: list[Job] = []
    for seed in seeds:
        # In the smoke test this run trains longer: it is the learning check (see check_learning).
        longer = ("train.max_steps_per_epoch=100",) if smoke and seed == seeds[0] else ()
        trains.append(train("official", main_cnn, seed, extra=longer))
    if plan["slice_b"]:
        for seed in other_seeds:
            trains.append(train("official", main_cnn, seed, "lab-aug", ("train.color_aug_max_delta_e=20",)))
    for backbone in backbones:
        for protocol in protocols:
            for seed in seeds if (protocol == "official" and not smoke) else other_seeds:
                if backbone == main_cnn and protocol == "official":
                    continue
                trains.append(train(protocol, backbone, seed))
    jobs += trains

    def names(**match: object) -> list[str]:
        out = []
        for j in trains:
            _, protocol, backbone, tag, _ = j.name.split(":")
            fields = {"protocol": protocol, "backbone": backbone, "tag": tag}
            if all(fields[k] == v for k, v in match.items()):
                out.append(j.name)
        return out

    if plan["fewshot"] or plan["slice_a"]:
        fs = ["fewshot", "--set", *paths, f"fewshot.num_workers={workers}"]
        if smoke:
            fs += [
                "fewshot.backbones=[dinov2_vits14, resnet50]",
                "fewshot.shots=[1, 5]",
                "fewshot.draws=2",
                "fewshot.C_sensitivity=[]",
            ]
        jobs.append(Job("fewshot", fs, "light", output=analysis / "q3_fewshot_fold1.json"))
    if plan["xai"]:
        # Every backbone's official seed-0 model: the Q4 comparison should not hinge on one network.
        for backbone in backbones:
            run = f"train/official/{backbone}/base/fold1/seed{seeds[0]}"
            xa = ["xai", "--set", *paths, f"xai.run={run}"]
            if smoke:
                xa += [
                    "xai.max_images=16",
                    "xai.sanity_images=4",
                    "xai.ig_steps=8",
                    "xai.perturbation_steps=10",
                    "xai.n_bootstrap=50",
                    "xai.batch_size=8",
                ]
            jobs.append(
                Job(
                    f"xai:{backbone}",
                    xa,
                    "light",
                    deps=[f"train:official:{backbone}:base:s{seeds[0]}"],
                    output=analysis / f"q4_xai_{run.replace('/', '_')}.json",
                )
            )
    if plan["slice_a"]:
        sa = ["superpixel", "--set", *paths, f"superpixel.num_workers={workers}"]
        if smoke:
            sa += ["superpixel.max_images=16", "superpixel.n_bootstrap=50"]
        jobs.append(
            Job(
                "slice_a",
                sa,
                "light",
                deps=["fewshot"],
                output=analysis / "sliceA_superpixel_dinov2_vits14_fold1.json",
            )
        )
    if plan["slice_b"]:
        pattern = (
            f"[train/official/{main_cnn}/base/fold1/seed*, train/official/{main_cnn}/lab-aug/fold1/seed*]"
        )
        sb = ["color", "--set", *paths, f"color.runs={pattern}", f"color.num_workers={workers}"]
        if smoke:
            sb += ["color.delta_e=[0.0, 10.0]", "color.hue_angles=[0.0, 180.0]", "color.max_images=256"]
        jobs.append(
            Job(
                "slice_b",
                sb,
                "light",
                deps=names(protocol="official", backbone=main_cnn),
                output=analysis / "sliceB_color.json",
            )
        )

    mapping = str(REPO / "configs" / "class_mapping.yaml")
    for backbone in backbones:
        if len(protocols) >= 2:
            jobs.append(
                Job(
                    f"q1:{backbone}",
                    ["leakage", "--runs-root", str(runs), "--backbone", backbone, "--n-boot", n_boot],
                    "cpu",
                    deps=names(backbone=backbone, tag="base"),
                    output=analysis / f"q1_leakage_{backbone}_base_fold1.json",
                )
            )
        if len(names(protocol="official", backbone=backbone, tag="base")) >= 2:
            jobs.append(
                Job(
                    f"q2:{backbone}",
                    [
                        "calibration",
                        "--runs-root",
                        str(runs),
                        "--backbone",
                        backbone,
                        "--n-boot",
                        n_boot,
                        "--class-mapping",
                        mapping,
                    ],
                    "cpu",
                    deps=names(protocol="official", backbone=backbone, tag="base"),
                    output=analysis / f"q2_calibration_{backbone}_base_fold1.json",
                )
            )
    results = (layout.smoke / "results") if smoke else layout.results
    jobs.append(
        Job(
            "report",
            ["report", "--runs-root", str(runs), "--out", str(results)],
            "cpu",
            deps=[j.name for j in jobs],
        )
    )
    return jobs


def check_learning(runs: Path, backbone: str, seed: int) -> str:
    """The smoke learning check: training loss must fall clearly within 100 steps."""
    path = runs / "train" / "official" / backbone / "base" / "fold1" / f"seed{seed}" / "train_steps.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) < 4:
        raise PhaseFailed(f"learning check: only {len(records)} records in {path}")
    first = sum(r["loss"] for r in records[:2]) / 2
    last = sum(r["loss"] for r in records[-2:]) / 2
    curve = ", ".join(f"{r['loss']:.2f}" for r in records)
    if not last < 0.9 * first:
        raise PhaseFailed(f"learning check: training loss did not fall ({first:.3f} -> {last:.3f}): {curve}")
    return f"training loss {first:.3f} -> {last:.3f} over {records[-1]['global_step']} steps ({curve})"


def smoke_plan(plan: dict[str, object]) -> dict[str, object]:
    """Every kind of job of the plan, with 2 seeds (Q2 needs an ensemble) and one of each protocol."""
    return {**plan, "seeds": [0, 1]}


def run_jobs(
    jobs: list[Job], args: argparse.Namespace, layout: Layout, label: str, budget_s: float | None
) -> bool:
    gpus = gpu_info()
    light = args.light_per_gpu
    if light is None:
        light = 1 if gpus and min(m for _, m in gpus) >= 12 else 0
    logs = layout.logs / label
    logs.mkdir(parents=True, exist_ok=True)
    log(f"  {len(jobs)} jobs on {len(gpus) or 'no'} GPU(s), light jobs per GPU next to training: {light}")
    sched = Scheduler(jobs, len(gpus), light, args.cpu_slots, logs, logs / "status.json", budget_s)
    ok = sched.run()
    counts: dict[str, int] = {}
    for j in jobs:
        counts[j.status] = counts.get(j.status, 0) + 1
    log(f"  {label}: {counts}")
    for j in jobs:
        if j.status in ("failed", "blocked", "deferred"):
            log(f"    {j.status:8} {j.name}")
    return ok


def default_workers() -> int:
    """DataLoader workers per job: the CPU cores shared by up to two jobs per GPU."""
    import torch

    n_gpus = max(1, torch.cuda.device_count())
    return max(2, min(8, (os.cpu_count() or 2) // (2 * n_gpus)))


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--backbones", nargs="+", help="override the profile's backbones")
    parser.add_argument("--protocols", nargs="+", choices=["official", "random", "video"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["preflight", "prepare", "smoke", "full"],
        default=["preflight", "prepare", "smoke", "full"],
    )
    parser.add_argument(
        "--work", type=Path, help="output root (default: repo locally, /kaggle/working on Kaggle)"
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="image caches (default: cache/ locally, /tmp/cxa/cache on Kaggle)"
    )
    parser.add_argument("--input-root", type=Path, help="where to look for the datasets")
    parser.add_argument("--kvasir-images", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--cv-root", type=Path)
    parser.add_argument(
        "--rebuild", action="store_true", help="ignore a prepared index/cache found in the inputs"
    )
    parser.add_argument("--time-budget-hours", type=float, default=11.0 if IS_KAGGLE else None)
    parser.add_argument(
        "--light-per-gpu",
        type=int,
        default=None,
        help="light jobs allowed next to a training job per GPU (default: 1 if >= 12 GiB)",
    )
    parser.add_argument("--cpu-slots", type=int, default=2)
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers per job")
    parser.add_argument("--no-unit-tests", dest="unit_tests", action="store_false")
    parser.add_argument(
        "--keep-smoke", action="store_true", help="keep the smoke runs (deleted after success)"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the job plan and stop")
    args = parser.parse_args()

    plan = dict(PROFILES[args.profile])
    if args.backbones:
        plan["backbones"] = args.backbones
    if args.protocols:
        plan["protocols"] = args.protocols
    if args.seeds:
        plan["seeds"] = args.seeds
    layout = make_layout(args)
    log(f"capsule-xai-audit run_all: profile {args.profile} {'(Kaggle)' if IS_KAGGLE else '(local)'}")
    log(f"  backbones {plan['backbones']}, protocols {plan['protocols']}, seeds {plan['seeds']}")
    log(f"  outputs {layout.work}, caches {layout.cache}")

    if args.dry_run:
        workers = args.workers or 2
        for j in build_jobs(plan, layout, layout.runs, smoke=False, workers=workers):
            log(f"  {j.kind:5} {j.name:44} deps={len(j.deps)}")
        return 0

    started = time.monotonic()
    try:
        if "preflight" in args.phases:
            checks = preflight(args, layout, plan)
            failed = [c for c in checks if c.status == "FAIL"]
            if failed:
                raise PhaseFailed("preflight failed: " + "; ".join(f"{c.name}: {c.detail}" for c in failed))
        if "prepare" in args.phases:
            prepare(args, layout)
        else:
            # Use what an earlier prepare left behind (or a prepared input on Kaggle).
            inputs = find_inputs(args, layout)
            if inputs.prepared is not None and not (layout.cache / "kvasir_224.json").is_file():
                layout.cache = inputs.prepared / "cache"
        workers = args.workers or default_workers()
        if "smoke" in args.phases:
            log("phase 3/4: smoke test (tiny version of every job, real data)")
            smoke_runs = layout.smoke / "runs"
            if smoke_runs.exists():
                shutil.rmtree(smoke_runs)
            jobs = build_jobs(smoke_plan(plan), layout, smoke_runs, smoke=True, workers=workers)
            if not run_jobs(jobs, args, layout, "smoke", None):
                raise PhaseFailed(f"smoke test failed; logs in {layout.logs / 'smoke'}")
            main_cnn = "resnet50" if "resnet50" in plan["backbones"] else plan["backbones"][0]  # type: ignore[index]
            log(f"  learning check: {check_learning(smoke_runs, str(main_cnn), 0)}")
            log("  smoke test passed")
            if not args.keep_smoke:
                shutil.rmtree(smoke_runs, ignore_errors=True)
        if "full" in args.phases:
            log("phase 4/4: full run")
            budget = (
                None
                if args.time_budget_hours is None
                else args.time_budget_hours * 3600 - (time.monotonic() - started)
            )
            jobs = build_jobs(plan, layout, layout.runs, smoke=False, workers=workers)
            ok = run_jobs(jobs, args, layout, "full", budget)
            if not ok:
                deferred = [j for j in jobs if j.status == "deferred"]
                if deferred:
                    log(f"  time budget reached, {len(deferred)} jobs left: attach this output and run again")
                raise PhaseFailed(f"full run incomplete; logs in {layout.logs / 'full'}")
            log(f"  done; results in {layout.results / 'RESULTS.md'}")
    except PhaseFailed as err:
        log(f"STOPPED: {err}")
        return 1
    log(f"finished in {(time.monotonic() - started) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
