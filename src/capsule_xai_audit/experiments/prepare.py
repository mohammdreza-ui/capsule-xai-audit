"""Data preparation: indices, caches and the deduplicated external set.

Outputs that are safe to publish (no images, no local paths) go to ``manifests/``:
    kvasir_manifest.csv                frames, labels, videos, folds, boxes
    kvasir_multilabel_excluded.csv     frames the official split lists with two labels
    cv2024_kvasir_duplicates.csv       Capsule Vision frames matched to Kvasir-Capsule by image
    cv2024_multilabel_excluded.csv     Capsule Vision images stored under two labels
    cv2024_train_val_duplicates.csv    validation frames matched to training frames (sanity check)
    cv2024_external_manifest.csv       frames used as the external set
    cv2024_dedup_summary.json          counts of every exclusion and of source/image agreement
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from capsule_xai_audit.data.cache import build_cache, open_cache
from capsule_xai_audit.data.capsule_vision import build_capsule_vision_index
from capsule_xai_audit.data.class_mapping import ClassMapping
from capsule_xai_audit.data.dedup import match_duplicates, phash_all
from capsule_xai_audit.data.kvasir import (
    IMAGE_EXTENSIONS,
    METADATA_URL,
    OFFICIAL_SPLIT_URL,
    SPLIT_FILES,
    build_kvasir_index,
    manifest,
    multilabel_frames,
)
from capsule_xai_audit.utils.io import sha256_file, write_csv, write_json
from capsule_xai_audit.utils.logging import get_logger

log = get_logger(__name__)


def fetch_official_splits(out_dir: Path) -> dict[str, str]:
    """Download the official split CSVs from the Kvasir-Capsule GitHub repository."""
    out_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in SPLIT_FILES:
        target = out_dir / name
        with urllib.request.urlopen(OFFICIAL_SPLIT_URL.format(name=name), timeout=60) as response:
            target.write_bytes(response.read())
        digests[name] = sha256_file(target)
        log.info("fetched %s (sha256 %s)", target, digests[name][:12])
    write_json(out_dir / "sha256.json", digests)
    return digests


def fetch_kvasir_metadata(out: Path) -> str:
    """Download Kvasir-Capsule ``metadata.csv`` (bounding boxes) from the official OSF project."""
    out.parent.mkdir(parents=True, exist_ok=True)
    # OSF answers 403 to urllib's default User-Agent.
    request = urllib.request.Request(METADATA_URL, headers={"User-Agent": "capsule-xai-audit"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    header = data.split(b"\n", 1)[0].decode("utf-8", "replace").strip().lower()
    if "filename" not in header or "x1" not in header:
        raise RuntimeError(f"{METADATA_URL} did not return metadata.csv (first line: {header[:120]!r})")
    out.write_bytes(data)
    digest = sha256_file(out)
    log.info("fetched %s (sha256 %s)", out, digest[:12])
    return digest


def _safe_members(archive: tarfile.TarFile, name: str) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for m in members:
        if m.name.startswith("/") or ".." in Path(m.name).parts or m.issym() or m.islnk() or m.isdev():
            raise RuntimeError(f"{name}: refusing unsafe archive member {m.name!r}")
    return members


def unpack_class_archives(src: Path, out: Path) -> int:
    """Extract the per-class ``*.tar.gz`` archives of the Simula release into ``out``.

    Members with absolute paths, ``..`` components, links or devices are refused. Returns
    the number of images below ``out`` afterwards; archives already extracted (marker file
    next to the images) are skipped, so the call is cheap to repeat.
    """
    archives = sorted(p for p in Path(src).iterdir() if p.name.endswith((".tar.gz", ".tgz", ".tar")))
    if not archives:
        raise FileNotFoundError(f"no class archives (*.tar.gz) in {src}")
    out.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        marker = out / f".{archive.name}.done"
        if marker.exists():
            continue
        with tarfile.open(archive) as tf:
            tf.extractall(out, members=_safe_members(tf, archive.name))
        marker.touch()
        log.info("unpacked %s", archive.name)
    return sum(1 for p in out.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def index_kvasir(
    images_root: Path,
    split_dir: Path,
    metadata_csv: Path | None,
    mapping: ClassMapping,
    out_index: Path,
    manifest_dir: Path | None,
    allow_missing: bool = False,
) -> pd.DataFrame:
    index = build_kvasir_index(images_root, split_dir, metadata_csv, mapping.kvasir_labels, allow_missing)
    write_csv(out_index, index)
    if manifest_dir is not None:
        write_csv(manifest_dir / "kvasir_manifest.csv", manifest(index))
        write_csv(manifest_dir / "kvasir_multilabel_excluded.csv", multilabel_frames(split_dir))
    counts = index.groupby(["official_fold", "label"]).size().unstack(0, fill_value=0)
    log.info("Kvasir-Capsule frames per class and fold:\n%s", counts.to_string())
    return index


def index_capsule_vision(root: Path, mapping: ClassMapping, out_index: Path) -> pd.DataFrame:
    index = build_capsule_vision_index(root, mapping.capsule_vision_labels)
    write_csv(out_index, index)
    return index


def cache_index(index_csv: Path, prefix: Path, size: int, num_workers: int) -> None:
    index = pd.read_csv(index_csv)
    build_cache(
        index["path"].tolist(), index["image_id"].tolist(), prefix, size=size, num_workers=num_workers
    )


def multilabel_images(cv_index: pd.DataFrame, cv_pixels: np.ndarray) -> np.ndarray:
    """Rows whose image is stored under several labels (same split, source and file name).

    Only groups whose cached pixels are identical count; files that merely share a name
    are left alone.
    """
    names = cv_index["path"].map(lambda p: Path(p).name)
    keys = cv_index["cv_split"].astype(str) + "/" + cv_index["source"].fillna("").astype(str) + "/" + names
    mask = np.zeros(len(cv_index), dtype=bool)
    for _, rows in cv_index.groupby(keys).groups.items():
        positions = np.asarray(rows)
        if len(positions) < 2 or cv_index["label"].iloc[positions].nunique() < 2:
            continue
        first = cv_pixels[positions[0]]
        if all(np.array_equal(first, cv_pixels[p]) for p in positions[1:]):
            mask[positions] = True
    return mask


def build_external_set(
    kvasir_index_csv: Path,
    kvasir_cache: Path,
    cv_index_csv: Path,
    cv_cache: Path,
    mapping: ClassMapping,
    out_external_index: Path,
    manifest_dir: Path,
    max_hamming: int = 10,
    min_ncc: float = 0.95,
) -> pd.DataFrame:
    """Capsule Vision without Kvasir-Capsule frames and multi-label images, shared classes only.

    Kvasir-Capsule frames are removed if *either* their source folder is ``KVASIR`` *or*
    pHash + NCC matches them to a frame of the Kvasir index (both folds: a Kvasir frame
    in the external set would be in-distribution, whichever fold it is from). The overlap
    of the two criteria is reported as a check of each.
    """
    k_index, cv_index = pd.read_csv(kvasir_index_csv), pd.read_csv(cv_index_csv)
    if "source" not in cv_index:
        cv_index["source"] = ""
    cv_index["source"] = cv_index["source"].fillna("").astype(str)
    k_pixels, _ = open_cache(kvasir_cache, k_index["image_id"].tolist())
    cv_pixels, _ = open_cache(cv_cache, cv_index["image_id"].tolist())

    log.info("hashing %d Kvasir and %d Capsule Vision frames", len(k_pixels), len(cv_pixels))
    k_hash, cv_hash = phash_all(k_pixels), phash_all(cv_pixels)

    vs_kvasir = match_duplicates(cv_pixels, k_pixels, cv_hash, k_hash, max_hamming, min_ncc).table
    dup = vs_kvasir[vs_kvasir["is_duplicate"]]
    write_csv(
        manifest_dir / "cv2024_kvasir_duplicates.csv",
        pd.DataFrame(
            {
                "cv_image_id": cv_index["image_id"].iloc[dup["query_pos"]].to_numpy(),
                "cv_label": cv_index["label"].iloc[dup["query_pos"]].to_numpy(),
                "cv_source": cv_index["source"].iloc[dup["query_pos"]].to_numpy(),
                "kvasir_image_id": k_index["image_id"].iloc[dup["best_ref_pos"]].to_numpy(),
                "kvasir_label": k_index["label"].iloc[dup["best_ref_pos"]].to_numpy(),
                "hamming": dup["hamming"].to_numpy(),
                "ncc": dup["ncc"].round(5).to_numpy(),
            }
        ),
    )

    # Sanity check comparable to the published audit of the challenge split.
    train_pos = np.flatnonzero(cv_index["cv_split"].to_numpy() == "training")
    val_pos = np.flatnonzero(cv_index["cv_split"].to_numpy() == "validation")
    tv = match_duplicates(
        cv_pixels[val_pos], cv_pixels[train_pos], cv_hash[val_pos], cv_hash[train_pos], max_hamming, min_ncc
    ).table
    tv_dup = tv[tv["is_duplicate"]]
    write_csv(
        manifest_dir / "cv2024_train_val_duplicates.csv",
        pd.DataFrame(
            {
                "val_image_id": cv_index["image_id"].iloc[val_pos[tv_dup["query_pos"]]].to_numpy(),
                "train_image_id": cv_index["image_id"].iloc[train_pos[tv_dup["best_ref_pos"]]].to_numpy(),
                "ncc": tv_dup["ncc"].round(5).to_numpy(),
            }
        ),
    )

    by_source = (cv_index["source"] == "KVASIR").to_numpy()
    by_image = vs_kvasir["is_duplicate"].to_numpy()
    multi = multilabel_images(cv_index, cv_pixels)
    write_csv(
        manifest_dir / "cv2024_multilabel_excluded.csv",
        cv_index.loc[multi, ["image_id", "label", "cv_split", "source"]],
    )

    cv_to_k = mapping.capsule_vision_to_kvasir_index()
    shared = cv_index["label"].isin(cv_to_k).to_numpy()
    keep = ~by_source & ~by_image & ~multi & shared
    external = cv_index.assign(cache_pos=np.arange(len(cv_index)))[keep].copy()
    external["label_idx"] = external["label"].map(cv_to_k).astype(int)
    external["kvasir_label"] = np.array(mapping.kvasir_labels)[external["label_idx"]]
    write_csv(out_external_index, external)
    write_csv(manifest_dir / "cv2024_external_manifest.csv", external.drop(columns=["path"]))

    has_source = bool(cv_index["source"].ne("").any())
    summary = {
        "capsule_vision_frames": len(cv_index),
        "frames_by_source": cv_index["source"].value_counts().to_dict(),
        "kvasir_by_source_folder": int(by_source.sum()) if has_source else None,
        "kvasir_by_image_match": int(by_image.sum()),
        "kvasir_by_both": int((by_source & by_image).sum()) if has_source else None,
        "kvasir_source_only": int((by_source & ~by_image).sum()) if has_source else None,
        "kvasir_image_match_only": int((~by_source & by_image).sum()) if has_source else None,
        "matched_by_cv_label": dup.assign(label=cv_index["label"].iloc[dup["query_pos"]].to_numpy())
        .groupby("label")
        .size()
        .to_dict(),
        "multilabel_images_excluded": int(multi.sum()),
        "validation_frames_matching_training": len(tv_dup),
        "external_frames": len(external),
        "external_by_label": external["label"].value_counts().to_dict(),
        "external_by_source": external["source"].value_counts().to_dict(),
        "max_hamming": max_hamming,
        "min_ncc": min_ncc,
    }
    write_json(manifest_dir / "cv2024_dedup_summary.json", summary)
    log.info("external set: %s", summary)
    return external
