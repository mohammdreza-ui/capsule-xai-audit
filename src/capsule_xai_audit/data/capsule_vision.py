"""Capsule Vision 2024 (train + validation) index.

Layout after unzipping: ``Dataset/{training,validation}/<Class>/<Source>/<frame>``, where
``<Source>`` is the dataset a frame came from (``KID``, ``KVASIR``, ``SEE-AI``, ``AIIMS``).
Kvasir-Capsule frames are removed from the external set by this folder, and image matching
(``data.dedup``) is run as an independent check (``experiments.prepare``).

Some SEE-AI images carry two labels and are stored once per label, under the same file
name, in two class folders; ``experiments.prepare`` excludes those from the external set.

Index columns: image_id, path, label, cv_split, source
``image_id`` is the path relative to the dataset root without extension, so it is unique
even when two classes contain files with the same name. ``source`` is empty when the
release has no source folders.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from capsule_xai_audit.data.kvasir import IMAGE_EXTENSIONS
from capsule_xai_audit.utils.logging import get_logger

log = get_logger(__name__)

CV_SPLITS: tuple[str, str] = ("training", "validation")
LABEL_FILES = {"training": "training_data.xlsx", "validation": "validation_data.xlsx"}


class CapsuleVisionIndexError(RuntimeError):
    pass


def find_dataset_root(path: Path) -> Path:
    """Accept either the ``Dataset`` folder itself or any parent that contains it."""
    path = Path(path)
    if all((path / s).is_dir() for s in CV_SPLITS):
        return path
    for candidate in sorted(path.rglob("training")):
        if candidate.is_dir() and (candidate.parent / "validation").is_dir():
            return candidate.parent
    raise CapsuleVisionIndexError(f"no training/ and validation/ folders found below {path}")


def build_capsule_vision_index(root: Path, classes: Sequence[str]) -> pd.DataFrame:
    root = find_dataset_root(root)
    allowed = set(classes)
    rows = []
    for split in CV_SPLITS:
        split_dir = root / split
        for path in sorted(split_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
                continue
            parts = path.relative_to(split_dir).parts
            label = parts[0]
            if label not in allowed:
                raise CapsuleVisionIndexError(
                    f"{path}: class folder {label!r} is not one of {sorted(allowed)}"
                )
            rows.append(
                {
                    "image_id": str(path.relative_to(root).with_suffix("")),
                    "path": str(path),
                    "label": label,
                    "cv_split": split,
                    "source": parts[1] if len(parts) > 2 else "",
                }
            )
    if not rows:
        raise CapsuleVisionIndexError(f"no images found below {root}")
    index = pd.DataFrame(rows)
    log.info(
        "Capsule Vision: %d frames; per split %s; per source %s",
        len(index),
        index["cv_split"].value_counts().to_dict(),
        index["source"].value_counts().to_dict(),
    )
    _cross_check_label_files(root, index)
    return index


def _cross_check_label_files(root: Path, index: pd.DataFrame) -> None:
    """Compare folder labels with the challenge's xlsx files when they are present.

    The column layout of those files is not documented, so the check looks for a column of
    image paths and reports disagreements instead of failing.
    """
    for split, name in LABEL_FILES.items():
        xlsx = root / split / name
        if not xlsx.exists():
            log.info("label file %s not found; skipping cross-check", xlsx)
            continue
        try:
            sheet = pd.read_excel(xlsx)
        except (ImportError, ValueError, OSError) as err:
            log.warning("could not read %s: %s", xlsx, err)
            continue
        path_cols = [
            c
            for c in sheet.columns
            if sheet[c].astype(str).str.lower().str.endswith(IMAGE_EXTENSIONS).mean() > 0.9
        ]
        if not path_cols:
            log.warning("%s: no column of image paths found (columns %s)", xlsx, list(sheet.columns))
            continue
        listed = {Path(p.replace("\\", "/")).name for p in sheet[path_cols[0]].astype(str)}
        on_disk = {Path(p).name for p in index.loc[index["cv_split"] == split, "path"]}
        log.info(
            "%s: %d listed, %d on disk, %d listed but missing, %d on disk but unlisted",
            name,
            len(listed),
            len(on_disk),
            len(listed - on_disk),
            len(on_disk - listed),
        )
