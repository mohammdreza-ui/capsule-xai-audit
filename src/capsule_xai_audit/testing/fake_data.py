"""Synthetic stand-ins for Kvasir-Capsule and Capsule Vision 2024.

Same file layout and CSV formats as the real releases, small enough for unit tests and
the end-to-end smoke run. Each non-normal frame has a coloured blob whose colour depends
on the class and whose bounding box is written to ``metadata.csv`` (semicolon-separated,
four corner points, like the real file), so classifiers and pointing-game code have a
learnable, localised signal.

The quirks of the real releases are reproduced on purpose:

* one frame listed twice in the official split with two labels and stored in both class
  folders (the real split has 8 Erosion + Pylorus frames);
* one frame stored byte-identically in a second, unsplit class folder;
* one video with frames in both official folds (the real split has 7 of 43);
* Capsule Vision ``<split>/<Class>/<Source>/`` folders, with Kvasir copies under ``KVASIR``,
  one ``KVASIR`` frame that is not a copy of an indexed frame, and one ``SEE-AI`` image
  stored under two labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from capsule_xai_audit.data.class_mapping import ClassMapping

KVASIR_FOLDER_NAMES = {
    "Angiectasia": "Angiectasia",
    "Blood": "Blood - fresh",
    "Erosion": "Erosion",
    "Erythematous": "Erythema",
    "Foreign Bodies": "Foreign body",
    "Ileo-cecal valve": "Ileocecal valve",
    "Lymphangiectasia": "Lymphangiectasia",
    "Normal": "Normal clean mucosa",
    "Pylorus": "Pylorus",
    "Reduced Mucosal View": "Reduced mucosal view",
    "Ulcer": "Ulcer",
}


@dataclass(frozen=True)
class FakeDataPaths:
    kvasir_images: Path
    kvasir_splits: Path
    kvasir_metadata: Path
    capsule_vision_root: Path
    # Capsule Vision image ids that are copies of Kvasir frames (ground truth for dedup tests).
    planted_duplicates: tuple[str, ...]
    # Kvasir frame listed with two labels in the official split.
    kvasir_multilabel: str
    # Video with frames in both official folds.
    shared_video: str
    # Capsule Vision image ids in KVASIR source folders (planted copies plus one non-copy).
    cv_kvasir_source: tuple[str, ...]
    # Capsule Vision image ids of the one image stored under two labels.
    cv_multilabel: tuple[str, ...]


def _class_colour(label_idx: int, n_classes: int) -> np.ndarray:
    hue = label_idx / n_classes
    rgb = np.array([np.cos(2 * np.pi * (hue + k / 3)) for k in range(3)]) * 0.5 + 0.5
    return (rgb * 255).astype(np.float64)


def synth_frame(
    rng: np.random.Generator, label_idx: int, n_classes: int, normal_idx: int, size: int = 336
) -> tuple[np.ndarray, list[float] | None]:
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.stack(
        [np.full((size, size), 170.0), np.full((size, size), 80.0), np.full((size, size), 60.0)], -1
    )
    img = base + rng.normal(0, 12, size=base.shape)
    box = None
    if label_idx != normal_idx:
        r = int(rng.integers(size // 10, size // 5))
        cx, cy = (int(v) for v in rng.integers(r + 5, size - r - 5, size=2))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        img[mask] = _class_colour(label_idx, n_classes) + rng.normal(0, 8, size=(int(mask.sum()), 3))
        box = [float(cx - r), float(cy - r), float(cx + r), float(cy + r)]
    return np.clip(img, 0, 255).astype(np.uint8), box


def make_fake_datasets(
    root: Path,
    mapping: ClassMapping,
    videos_per_fold: int = 4,
    frames_per_video: int = 24,
    extra_unsplit: int = 3,
    cv_per_class: int = 8,
    cv_duplicates: int = 10,
    size: int = 336,
    seed: int = 0,
) -> FakeDataPaths:
    rng = np.random.default_rng(seed)
    root = Path(root)
    classes = list(mapping.kvasir_labels)
    n_classes = len(classes)
    normal_idx = classes.index("Normal")
    img_root, split_dir = root / "kvasir" / "labelled_images", root / "kvasir" / "official_splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    split_rows: list[list[dict[str, str]]] = [[], []]
    meta_rows = []
    kvasir_frames: list[tuple[np.ndarray, str]] = []
    for fold in (0, 1):
        for v in range(videos_per_fold):
            video = f"{fold:x}{v:03x}{rng.integers(0, 16**12):012x}"
            for f in range(frames_per_video):
                # Every video covers every class at least once, then random labels.
                label_idx = f if f < n_classes else int(rng.integers(0, n_classes))
                img, box = synth_frame(rng, label_idx, n_classes, normal_idx, size)
                stem = f"{video}_{f}"
                folder = img_root / KVASIR_FOLDER_NAMES[classes[label_idx]]
                folder.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(folder / f"{stem}.png")
                split_rows[fold].append({"filename": f"{stem}.jpg", "label": classes[label_idx]})
                kvasir_frames.append((img, classes[label_idx]))
                corners = [np.nan] * 8
                if box is not None:
                    x0, y0, x1, y1 = box
                    corners = [x0, y0, x1, y0, x1, y1, x0, y1]
                meta_rows.append(
                    dict(
                        zip(
                            [
                                "filename",
                                "video_id",
                                "frame_number",
                                "finding_class",
                                "x1",
                                "y1",
                                "x2",
                                "y2",
                                "x3",
                                "y3",
                                "x4",
                                "y4",
                            ],
                            [f"{stem}.png", video, f, classes[label_idx], *corners],
                            strict=True,
                        )
                    )
                )
    # A frame with two labels: listed twice (the second time as Pylorus) and stored in both folders.
    multi_stem = Path(split_rows[1][classes.index("Erosion")]["filename"]).stem
    multi_src = img_root / KVASIR_FOLDER_NAMES["Erosion"] / f"{multi_stem}.png"
    (img_root / KVASIR_FOLDER_NAMES["Pylorus"] / multi_src.name).write_bytes(multi_src.read_bytes())
    split_rows[1].append({"filename": f"{multi_stem}.jpg", "label": "Pylorus"})
    # A byte-identical copy in a folder outside the split (like Ampulla of vater/Pylorus).
    pylorus_stem = Path(split_rows[0][classes.index("Pylorus")]["filename"]).stem
    ampulla = img_root / "Ampulla of vater"
    ampulla.mkdir(parents=True, exist_ok=True)
    src = img_root / KVASIR_FOLDER_NAMES["Pylorus"] / f"{pylorus_stem}.png"
    (ampulla / src.name).write_bytes(src.read_bytes())
    # One video's Ulcer frames go to the other fold, as the official split does per (video, class).
    shared_video = Path(split_rows[0][0]["filename"]).stem.rsplit("_", 1)[0]
    moved = [r for r in split_rows[0] if r["filename"].startswith(shared_video) and r["label"] == "Ulcer"]
    split_rows[0] = [r for r in split_rows[0] if r not in moved]
    split_rows[1].extend(moved)

    # Frames on disk that are not part of the official split (like Ampulla/Polyp/hematin).
    for e in range(extra_unsplit):
        img, _ = synth_frame(rng, 0, n_classes, normal_idx, size)
        folder = img_root / "Polyp"
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(folder / f"ffffffffffffffff_{e}.png")

    for fold in (0, 1):
        pd.DataFrame(split_rows[fold]).to_csv(split_dir / f"split_{fold}.csv", index=False)
    metadata = root / "kvasir" / "metadata.csv"
    pd.DataFrame(meta_rows).to_csv(metadata, sep=";", index=False)

    cv_root = root / "capsule_vision" / "Dataset"
    cv_to_kvasir = {v: k for k, v in mapping.shared.items()}
    for split in ("training", "validation"):
        for cv_label in mapping.capsule_vision_labels:
            k_label = cv_to_kvasir.get(cv_label)
            for i in range(cv_per_class):
                folder = cv_root / split / cv_label / ("KID" if i % 4 == 0 else "SEE-AI")
                folder.mkdir(parents=True, exist_ok=True)
                # External-domain look: different mucosa colour and lighting.
                label_idx = classes.index(k_label) if k_label else int(rng.integers(0, n_classes))
                img, _ = synth_frame(rng, label_idx, n_classes, normal_idx, 320)
                img = np.clip(img.astype(np.float64) * [0.9, 1.1, 1.15], 0, 255).astype(np.uint8)
                Image.fromarray(img).save(folder / f"cv_{split}_{i}.jpg", quality=92)

    planted = []
    shared_frames = [(img, lab) for img, lab in kvasir_frames if lab in mapping.shared]
    picks = rng.choice(len(shared_frames), size=min(cv_duplicates, len(shared_frames)), replace=False)
    for j, pick in enumerate(picks):
        img, label = shared_frames[int(pick)]
        rel = Path("training") / mapping.shared[label] / "KVASIR" / f"kvasir_copy_{j}"
        (cv_root / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).resize((320, 320)).save(cv_root / f"{rel}.jpg", quality=90)
        planted.append(str(rel))
    # A KVASIR-source frame that matches no indexed frame (like the unsplit Polyp frames).
    unmatched = Path("validation") / "Erosion" / "KVASIR" / "kvasir_unindexed_0"
    (cv_root / unmatched).parent.mkdir(parents=True, exist_ok=True)
    img, _ = synth_frame(rng, classes.index("Erosion"), n_classes, normal_idx, 320)
    Image.fromarray(img).save(cv_root / f"{unmatched}.jpg", quality=92)
    # One SEE-AI image stored under two labels with the same file name.
    img, _ = synth_frame(rng, classes.index("Erosion"), n_classes, normal_idx, 320)
    multi_ids = []
    for cv_label in ("Erosion", "Bleeding"):
        rel = Path("training") / cv_label / "SEE-AI" / "seeai_two_labels"
        Image.fromarray(img).save(cv_root / f"{rel}.jpg", quality=92)
        multi_ids.append(str(rel))

    return FakeDataPaths(
        kvasir_images=img_root,
        kvasir_splits=split_dir,
        kvasir_metadata=metadata,
        capsule_vision_root=cv_root,
        planted_duplicates=tuple(planted),
        kvasir_multilabel=multi_stem,
        shared_video=shared_video,
        cv_kvasir_source=(*planted, str(unmatched)),
        cv_multilabel=tuple(multi_ids),
    )
