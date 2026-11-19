"""Kvasir-Capsule index: one row per labelled frame in the official split.

The official split CSVs (``official_splits/split_{0,1}.csv``, header ``filename,label``)
list frames as ``<video_id>_<frame>.jpg``. Releases differ in image format (the paper
describes PNG, the Simula zip ships JPEG in one archive per class), so frames are joined
on the file stem. Frames on disk that are not in the official split (Ampulla of Vater,
Polyp, Blood-hematin) are excluded and counted.

Two quirks of the released data are handled explicitly:

* The official split lists 8 frames twice, once as Erosion and once as Pylorus. The
  pipeline is single-label, so these frames are dropped (``multilabel_frames``).
* The same frames are stored, byte-identical, in two class folders. Duplicate file names
  are accepted when the files are identical and rejected otherwise.

The official split is *not* video-disjoint: 7 of 43 videos have frames in both folds
(each (video, class) pair sits in one fold). ``videos_in_both_folds`` reports this; the
``video`` split protocol in ``data.splits`` provides a patient-level alternative.

Index columns:
    image_id, path, label, label_idx, video_id, official_fold, bboxes
where ``bboxes`` is a JSON list of ``[x0, y0, x1, y1]`` in original-image pixels.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from capsule_xai_audit.utils.logging import get_logger

log = get_logger(__name__)

SPLIT_FILES: tuple[str, str] = ("split_0.csv", "split_1.csv")
OFFICIAL_SPLIT_URL = "https://raw.githubusercontent.com/simula/kvasir-capsule/master/official_splits/{name}"
# metadata.csv (bounding boxes) is not in the Simula zip; the official OSF project hosts it.
METADATA_URL = "https://osf.io/download/kzc8w/"
IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")

INDEX_COLUMNS: tuple[str, ...] = (
    "image_id",
    "path",
    "label",
    "label_idx",
    "video_id",
    "official_fold",
    "bboxes",
)


class KvasirIndexError(RuntimeError):
    pass


def video_id_from_stem(stem: str) -> str:
    """``bca26705313a4644_20404`` -> ``bca26705313a4644``."""
    video, sep, frame = stem.rpartition("_")
    if not sep or not video or not frame.isdigit():
        raise KvasirIndexError(f"unexpected Kvasir-Capsule frame name {stem!r}")
    return video


def _read_split_rows(split_dir: Path) -> pd.DataFrame:
    frames = []
    for fold, name in enumerate(SPLIT_FILES):
        path = Path(split_dir) / name
        if not path.exists():
            raise KvasirIndexError(f"missing official split file {path}; run `cxa fetch-splits`")
        df = pd.read_csv(path)
        if list(df.columns) != ["filename", "label"]:
            raise KvasirIndexError(f"{path}: expected header 'filename,label', got {list(df.columns)}")
        df["official_fold"] = fold
        frames.append(df)
    rows = pd.concat(frames, ignore_index=True)
    rows["image_id"] = rows["filename"].map(lambda f: Path(f).stem)
    return rows.drop_duplicates(["image_id", "label", "official_fold"])


def multilabel_frames(split_dir: Path) -> pd.DataFrame:
    """Frames the official split lists with more than one label (one row per listing)."""
    rows = _read_split_rows(split_dir)
    n_labels = rows.groupby("image_id")["label"].transform("nunique")
    return rows.loc[n_labels > 1, ["image_id", "label", "official_fold"]].sort_values(["image_id", "label"])


def read_official_splits(split_dir: Path) -> pd.DataFrame:
    """Single-label frames of the official split: image_id, label, official_fold, video_id."""
    rows = _read_split_rows(split_dir)
    multi = set(multilabel_frames(split_dir)["image_id"])
    if multi:
        log.warning("dropping %d frames that the official split lists with two labels", len(multi))
        rows = rows[~rows["image_id"].isin(multi)]
    repeated = rows["image_id"].duplicated(keep=False)
    if repeated.any():
        raise KvasirIndexError(
            f"{rows.loc[repeated, 'image_id'].nunique()} frames are listed in both official folds"
        )
    rows = rows.assign(video_id=rows["image_id"].map(video_id_from_stem))
    return rows[["image_id", "label", "official_fold", "video_id"]].reset_index(drop=True)


def discover_images(root: Path, extensions: Sequence[str] = IMAGE_EXTENSIONS) -> dict[str, Path]:
    """Map file stem -> path for every image below ``root``.

    A stem found in several folders is accepted only if all copies are byte-identical
    (the first path in sorted order is kept); differing copies raise ``KvasirIndexError``.
    """
    found: dict[str, Path] = {}
    identical_copies = 0
    exts = {e.lower() for e in extensions}
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in exts or not path.is_file():
            continue
        if path.stem in found:
            if path.read_bytes() != found[path.stem].read_bytes():
                raise KvasirIndexError(
                    f"two different images share the name {path.stem!r}: {found[path.stem]} and {path}"
                )
            identical_copies += 1
            continue
        found[path.stem] = path
    if not found:
        raise KvasirIndexError(f"no images found below {root}")
    if identical_copies:
        log.info("%d image files are byte-identical copies stored in a second class folder", identical_copies)
    return found


@dataclass(frozen=True)
class _BoxColumns:
    filename: str
    xs: tuple[str, ...]
    ys: tuple[str, ...]


def _locate_box_columns(columns: Sequence[str]) -> _BoxColumns:
    lower = {c.strip().lower(): c for c in columns}
    filename = next((lower[k] for k in ("filename", "file_name", "image", "file") if k in lower), None)
    if filename is None:
        raise KvasirIndexError(f"metadata has no filename column; columns are {list(columns)}")
    corner_x = tuple(lower[f"x{i}"] for i in range(1, 5) if f"x{i}" in lower)
    corner_y = tuple(lower[f"y{i}"] for i in range(1, 5) if f"y{i}" in lower)
    if len(corner_x) >= 2 and len(corner_x) == len(corner_y):
        return _BoxColumns(filename, corner_x, corner_y)
    minmax = [k for k in ("x_min", "y_min", "x_max", "y_max") if k in lower]
    if len(minmax) == 4:
        return _BoxColumns(filename, (lower["x_min"], lower["x_max"]), (lower["y_min"], lower["y_max"]))
    raise KvasirIndexError(f"metadata has no recognisable bounding-box columns; columns are {list(columns)}")


def read_bounding_boxes(metadata_csv: Path) -> dict[str, list[list[float]]]:
    """Axis-aligned boxes per frame from Kvasir-Capsule ``metadata.csv``.

    The delimiter is sniffed (the released file is not comma-separated) and box corners
    are reduced to ``[min x, min y, max x, max y]``. Frames may carry several boxes.
    """
    df = pd.read_csv(metadata_csv, sep=None, engine="python")
    cols = _locate_box_columns(list(df.columns))
    xs = df[list(cols.xs)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    ys = df[list(cols.ys)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(xs).all(axis=1) & np.isfinite(ys).all(axis=1)
    boxes: dict[str, list[list[float]]] = {}
    for name, x, y in zip(df.loc[valid, cols.filename], xs[valid], ys[valid], strict=True):
        box = [float(x.min()), float(y.min()), float(x.max()), float(y.max())]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        boxes.setdefault(Path(str(name)).stem, []).append(box)
    log.info("metadata: %d frames with at least one valid bounding box", len(boxes))
    return boxes


def build_kvasir_index(
    images_root: Path,
    split_dir: Path,
    metadata_csv: Path | None = None,
    classes: Sequence[str] | None = None,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Join official splits, images on disk and (optional) bounding boxes."""
    splits = read_official_splits(split_dir)
    images = discover_images(images_root)

    missing = sorted(set(splits["image_id"]) - set(images))
    if missing:
        msg = f"{len(missing)} official-split frames have no image file (e.g. {missing[:3]})"
        if not allow_missing:
            raise KvasirIndexError(msg)
        log.warning("%s; dropping them", msg)
        splits = splits[~splits["image_id"].isin(missing)]
    not_in_split = len(set(images) - set(splits["image_id"]))
    log.info("%d image files are not part of the official splits and are excluded", not_in_split)

    class_names = list(classes) if classes is not None else sorted(splits["label"].unique())
    unknown = sorted(set(splits["label"]) - set(class_names))
    if unknown:
        raise KvasirIndexError(f"official-split labels {unknown} are not in the configured class list")
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    boxes = read_bounding_boxes(metadata_csv) if metadata_csv is not None else {}
    index = splits.assign(
        path=splits["image_id"].map(lambda s: str(images[s])),
        label_idx=splits["label"].map(class_to_idx).astype(int),
        bboxes=splits["image_id"].map(lambda s: json.dumps(boxes.get(s, []))),
    )
    index = index.sort_values(["official_fold", "video_id", "image_id"]).reset_index(drop=True)
    shared = videos_in_both_folds(index)
    if len(shared):
        log.warning(
            "the official split is not video-disjoint: %d of %d videos (%d frames) have frames in both folds",
            len(shared),
            index["video_id"].nunique(),
            int(shared.to_numpy().sum()),
        )
    return index[list(INDEX_COLUMNS)]


def videos_in_both_folds(index: pd.DataFrame) -> pd.DataFrame:
    """Frames per official fold for every video that appears in both folds."""
    counts = index.groupby(["video_id", "official_fold"]).size().unstack(fill_value=0)
    return counts[(counts > 0).sum(axis=1) > 1]


def parse_bboxes(index: pd.DataFrame) -> list[list[list[float]]]:
    return [json.loads(b) for b in index["bboxes"]]


def manifest(index: pd.DataFrame) -> pd.DataFrame:
    """Publishable view of an index: no local paths, no image content."""
    return index.drop(columns=["path"])
