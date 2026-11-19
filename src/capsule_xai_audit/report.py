"""Collect analysis JSONs into ``results/RESULTS.md``, small JSON copies and figures.

Only aggregate numbers leave ``runs/``: no images, no per-frame predictions.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from capsule_xai_audit.utils.io import atomic_write_bytes, read_json, write_csv
from capsule_xai_audit.utils.logging import get_logger

log = get_logger(__name__)


def fmt_ci(d: dict[str, float], digits: int = 3) -> str:
    return f"{d['point']:.{digits}f} [{d['low']:.{digits}f}, {d['high']:.{digits}f}]"


def fmt_draws(d: dict[str, float], digits: int = 3) -> str:
    return f"{d['mean']:.{digits}f} [{d['q025']:.{digits}f}, {d['q975']:.{digits}f}]"


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def training_runs(runs_root: Path) -> list[dict[str, Any]]:
    """One record per completed training run, with its per-epoch and per-step logs."""
    runs = []
    for meta_path in sorted((runs_root / "train").glob("*/*/*/fold*/seed*/run.json")):
        meta = read_json(meta_path)
        if not meta.get("complete"):
            continue
        cfg = meta["config"]["train"]
        history = _read_jsonl(meta_path.parent / "history.jsonl")
        fallback = history[-1] if history else {}
        best = next((h for h in history if h["epoch"] == meta.get("best_epoch")), fallback)
        runs.append(
            {
                "group": f"{cfg['split_protocol']}/{cfg['backbone']}/{cfg['tag']}/fold{cfg['test_fold']}",
                "backbone": cfg["backbone"],
                "label": f"{cfg['split_protocol']}/{cfg['tag']}",
                "seed": cfg["seed"],
                "epochs_run": meta.get("epochs_run"),
                "best_epoch": meta.get("best_epoch"),
                "stopped_early": meta.get("stopped_early"),
                "minutes": round(meta.get("train_seconds", float("nan")) / 60, 2),
                "best_val_macro_f1": best.get("val_macro_f1"),
                "best_val_nll": best.get("val_nll"),
                "best_val_ece": best.get("val_ece"),
                "history": history,
                "steps": _read_jsonl(meta_path.parent / "train_steps.jsonl"),
            }
        )
    return runs


def training_section(runs_root: Path, out_dir: Path, figures: Path) -> str:
    runs = training_runs(runs_root)
    if not runs:
        return ""
    keys = (
        "group",
        "seed",
        "epochs_run",
        "best_epoch",
        "stopped_early",
        "minutes",
        "best_val_macro_f1",
        "best_val_nll",
        "best_val_ece",
    )
    write_csv(out_dir / "training_runs.csv", pd.DataFrame([{k: r[k] for k in keys} for r in runs]))
    out = ["## Training", ""]
    rows = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        groups.setdefault(r["group"], []).append(r)
    for group, rs in sorted(groups.items()):
        f1 = np.array([r["best_val_macro_f1"] for r in rs], dtype=float)
        rows.append(
            [
                group,
                str(len(rs)),
                f"{np.mean([r['epochs_run'] for r in rs]):.1f}",
                f"{np.mean([r['best_epoch'] for r in rs]):.1f}",
                f"{f1.mean():.3f} ± {f1.std(ddof=1) if len(f1) > 1 else 0:.3f}",
                f"{np.mean([r['minutes'] for r in rs]):.1f}",
            ]
        )
    header = ["run group", "seeds", "epochs run", "best epoch", "best val macro-F1", "minutes"]
    out.append(table(header, rows))
    out += ["", "Per-run values: `training_runs.csv`. Curves: mean over seeds.", ""]

    for backbone in sorted({r["backbone"] for r in runs}):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for label in sorted({r["label"] for r in runs if r["backbone"] == backbone}):
            rs = [r for r in runs if r["backbone"] == backbone and r["label"] == label]
            steps = [pd.DataFrame(r["steps"]) for r in rs if r["steps"]]
            if steps:
                curve = pd.concat(steps).groupby("global_step")["loss"].mean()
                axes[0].plot(curve.index, curve.to_numpy(), label=label)
            hist = pd.concat([pd.DataFrame(r["history"]) for r in rs if r["history"]])
            if len(hist):
                mean = hist.groupby("epoch")[["val_macro_f1", "val_nll"]].mean()
                axes[1].plot(mean.index, mean["val_macro_f1"], marker="o", label=label)
                axes[2].plot(mean.index, mean["val_nll"], marker="o", label=label)
        axes[0].set(title=f"{backbone}: training loss", xlabel="step", ylabel="cross-entropy (window mean)")
        axes[1].set(title="validation macro-F1", xlabel="epoch", ylabel="macro-F1")
        axes[2].set(title="validation NLL", xlabel="epoch", ylabel="NLL")
        for ax in axes:
            ax.legend(fontsize=7)
        for ax in axes[1:]:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _save(fig, figures / f"training_{backbone}.png")
        out += [f"![training {backbone}]({figures.name}/training_{backbone}.png)", ""]
    return "\n".join(out)


def q1_section(files: list[Path], figures: Path) -> str:
    out = ["## Q1 — split leakage", ""]
    rows, diff_rows = [], []
    for f in files:
        r = read_json(f)
        m = r["metrics"]["macro_f1"]
        for p in r["protocols"]:
            rows.append([r["backbone"], str(r["test_fold"]), p, str(r["n_seeds"][p]), fmt_ci(m[p])])
        for name, d in m["differences"].items():
            diff_rows.append([r["backbone"], str(r["test_fold"]), name.replace("_minus_", " − "), fmt_ci(d)])
    out.append(table(["backbone", "test fold", "protocol", "seeds", "macro-F1"], rows))
    out += ["", table(["backbone", "test fold", "difference", "macro-F1 [95% CI]"], diff_rows)]
    out += [
        "",
        "95% intervals: video-cluster bootstrap, seeds averaged within each replicate. "
        "`random − official` is the pre-registered contrast (H1); `official − video` measures the "
        "leakage left in the official split (7 of 43 videos in both folds).",
        "",
    ]
    common_rows, common_classes = [], None
    for f in files:
        r = read_json(f)
        m = r["metrics"].get("macro_f1_common_classes")
        if m is None:
            continue
        common_classes = r["common_classes"]
        for p in r["protocols"]:
            common_rows.append([r["backbone"], p, fmt_ci(m[p])])
        for name, d in m["differences"].items():
            common_rows.append([r["backbone"], name.replace("_minus_", " − "), fmt_ci(d)])
    if common_rows:
        out += [
            f"Macro-F1 averaged over the {len(common_classes or [])} classes present in every protocol's "
            f"test set ({', '.join(common_classes or [])}), so that protocols are compared on one class set:",
            "",
            table(["backbone", "protocol or difference", "macro-F1 [95% CI]"], common_rows),
            "",
        ]
    fig, ax = plt.subplots(figsize=(6, 1 + 0.45 * sum(len(read_json(f)["protocols"]) for f in files)))
    labels, y = [], 0
    for f in files:
        r = read_json(f)
        m = r["metrics"]["macro_f1"]
        for p in r["protocols"]:
            iv = m[p]
            ax.errorbar(
                iv["point"],
                y,
                xerr=[[iv["point"] - iv["low"]], [iv["high"] - iv["point"]]],
                fmt="o",
                capsize=3,
            )
            labels.append(f"{r['backbone']} · {p}")
            y += 1
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set(xlabel="test macro-F1 (95% CI)", title="Q1: split protocol")
    _save(fig, figures / "q1_macro_f1.png")
    out += [f"![Q1]({figures.name}/q1_macro_f1.png)", ""]
    return "\n".join(out)


def q2_section(files: list[Path], figures: Path) -> str:
    out = ["## Q2 — calibration under shift", ""]
    for f in files:
        r = read_json(f)
        out += [f"### {r['group']} (seeds {r['seeds']})", ""]
        rows = []
        for domain, d in r["domains"].items():
            for cond, metrics in d["conditions"].items():
                rows.append(
                    [domain, cond, *(fmt_ci(metrics[m]) for m in ("ece", "nll", "brier", "macro_f1"))]
                )
        out.append(table(["domain", "condition", "ECE", "NLL", "Brier", "macro-F1"], rows))
        if "h2b_external_minus_id_shared_ece" in r:
            h = r["h2b_external_minus_id_shared_ece"]
            out += [
                "",
                f"H2b, external − ID ECE after scaling: single {fmt_ci(h['single_ts'])}, "
                f"ensemble {fmt_ci(h['ensemble_ts'])}.",
            ]
        if "external" in r["domains"]:
            e = r["domains"]["external"]["ensemble_minus_single"]
            out += [
                f"H2a, ensemble − single on external: ECE {fmt_ci(e['ece'])}, NLL {fmt_ci(e['nll'])}, "
                f"Brier {fmt_ci(e['brier'])}."
            ]
        out.append("")
        _reliability_figure(r, figures / f"{f.stem}_reliability.png")
        out += [f"![reliability]({figures.name}/{f.stem}_reliability.png)", ""]
    return "\n".join(out)


def _reliability_figure(r: dict[str, Any], path: Path) -> None:
    domains = list(r["domains"])
    fig, axes = plt.subplots(1, len(domains), figsize=(4 * len(domains), 4), squeeze=False)
    for ax, domain in zip(axes[0], domains, strict=True):
        for cond, style in (("single", "o-"), ("ensemble_ts", "s--")):
            rel = r["domains"][domain]["reliability"][cond]
            pts = [
                (c, a)
                for c, a, n in zip(rel["confidence"], rel["accuracy"], rel["count"], strict=True)
                if n > 0
            ]
            if pts:
                ax.plot(*zip(*pts, strict=True), style, label=cond, markersize=4)
        ax.plot([0, 1], [0, 1], color="0.6", linewidth=1)
        ax.set(title=domain, xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def q3_section(files: list[Path], figures: Path) -> str:
    out = ["## Q3 — few-shot linear probes", ""]
    for f in files:
        r = read_json(f)
        shots = [str(k) for k in r["config"]["fewshot"]["shots"]]
        rows = []
        for backbone, per_k in r["backbones"].items():
            for k in [*shots, "all"]:
                s = per_k[k]["summary"]
                if k == "all":
                    rows.append([backbone, "all", f"{s['ece']:.3f}", f"{s['macro_f1']:.3f}"])
                else:
                    rows.append([backbone, k, fmt_draws(s["ece"]), fmt_draws(s["macro_f1"])])
        out.append(table(["backbone", "shots/class", "ECE", "macro-F1"], rows))
        rank = r["ranking_by_ece_at_smallest_k"]
        out += [
            "",
            f"Lowest ECE at k={rank['k']}: {' < '.join(rank['order'])} (mean over draws; "
            "intervals are 2.5–97.5% of draws).",
            "",
        ]
        fig, ax = plt.subplots(figsize=(5, 4))
        for backbone, per_k in r["backbones"].items():
            ax.errorbar(
                [int(k) for k in shots],
                [per_k[k]["summary"]["ece"]["mean"] for k in shots],
                yerr=[per_k[k]["summary"]["ece"]["sd"] for k in shots],
                marker="o",
                capsize=3,
                label=backbone,
            )
        ax.set(xscale="log", xlabel="labelled frames per class", ylabel="ECE")
        ax.set_xticks([int(k) for k in shots], [str(k) for k in shots])
        ax.minorticks_off()
        ax.legend()
        fig.tight_layout()
        figures.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures / f"{f.stem}.png", dpi=130)
        plt.close(fig)
        out += [f"![few-shot]({figures.name}/{f.stem}.png)", ""]
    return "\n".join(out)


def q4_section(files: list[Path], figures: Path) -> str:
    out = ["## Q4 — explanation methods", ""]
    for f in files:
        r = read_json(f)
        out += [f"### {r['config']['xai']['run']} ({r['n_images']} boxed frames, {r['n_videos']} videos)", ""]
        metrics = r["metrics"]
        methods = sorted(metrics["pointing"])
        rows = [
            [m, *(fmt_ci(metrics[k][m]) for k in ("pointing", "energy", "deletion_auc", "insertion_auc"))]
            for m in methods
        ]
        out.append(table(["method", "pointing ↑", "energy ↑", "deletion AUC ↓", "insertion AUC ↑"], rows))
        out += ["", "Kendall tau between method rankings:", ""]
        out.append(
            table(
                ["metric pair", "tau [95% CI]"],
                [
                    [k, f"{v['tau']:.2f} [{v['low']:.2f}, {v['high']:.2f}]"]
                    for k, v in r["agreement"]["pairs"].items()
                ],
            )
        )
        by = r.get("by_correctness")
        if by is not None:
            out += [
                "",
                f"Model accuracy on these frames: {r['accuracy_on_images']:.2f}. Pointing and deletion AUC "
                "split by whether the model classifies the frame correctly:",
                "",
            ]
            sub_rows = []
            for subset in ("correct", "wrong"):
                e = by[subset]
                head = f"{subset} ({e['n_images']} frames, {e['n_videos']} videos)"
                if "metrics" not in e:
                    sub_rows.append([head, "too few frames", "", ""])
                    continue
                for m in methods:
                    sub_rows.append(
                        [
                            head,
                            m,
                            fmt_ci(e["metrics"]["pointing"][m]),
                            fmt_ci(e["metrics"]["deletion_auc"][m]),
                        ]
                    )
            out.append(table(["frames", "method", "pointing ↑", "deletion AUC ↓"], sub_rows))
        out += ["", "Sanity check (Spearman with original map after randomising all layer groups):", ""]
        out.append(
            table(
                ["method", "Spearman (last step)"],
                [
                    # The random baseline ignores the model, so the test does not apply to it.
                    [m, "n/a" if m == "random" else f"{steps[-1]['spearman_mean']:.3f}"]
                    for m, steps in r["sanity"]["methods"].items()
                ],
            )
        )
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
        names = ("pointing", "energy", "deletion_auc", "insertion_auc")
        titles = ("pointing ↑", "energy ↑", "deletion AUC ↓", "insertion AUC ↑")
        for ax, metric, title in zip(axes, names, titles, strict=True):
            points = [metrics[metric][m]["point"] for m in methods]
            err = [
                [metrics[metric][m]["point"] - metrics[metric][m]["low"] for m in methods],
                [metrics[metric][m]["high"] - metrics[metric][m]["point"] for m in methods],
            ]
            ax.barh(methods, points, xerr=err, capsize=3, color="0.7")
            ax.set(title=title)
        stem = f.stem
        _save(fig, figures / f"{stem}.png")
        out += ["", f"![Q4]({figures.name}/{stem}.png)", ""]
    return "\n".join(out)


def slice_a_section(files: list[Path], figures: Path) -> str:
    out = ["## Slice A — superpixel units", ""]
    for f in files:
        r = read_json(f)
        d = r["superpixel_minus_patch"]
        rates = r["rates"]
        out.append(
            table(
                ["metric", "patch", "superpixel", "superpixel − patch"],
                [
                    [
                        "pointing",
                        f"{rates['pointing_patch']:.3f}",
                        f"{rates['pointing_superpixel']:.3f}",
                        fmt_ci(d["pointing"]),
                    ],
                    [
                        "energy",
                        f"{rates['energy_patch']:.3f}",
                        f"{rates['energy_superpixel']:.3f}",
                        fmt_ci(d["energy"]),
                    ],
                ],
            )
        )
        out += [
            "",
            f"Probe test macro-F1 {r['probe_test_metrics']['macro_f1']:.3f} (identical for both units); "
            f"{r['n_images']} frames, mean {r['mean_superpixels']:.0f} superpixels.",
            "",
        ]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        for ax, metric in zip(axes, ("pointing", "energy"), strict=True):
            values = [rates[f"{metric}_patch"], rates[f"{metric}_superpixel"]]
            ax.bar(["patch", "superpixel"], values, color="0.7")
            diff = d[metric]
            ax.set(title=f"{metric}: Δ {diff['point']:+.3f} [{diff['low']:+.3f}, {diff['high']:+.3f}]")
        _save(fig, figures / f"{f.stem}.png")
        out += [f"![Slice A]({figures.name}/{f.stem}.png)", ""]
    return "\n".join(out)


# Below this ID - external difference the recovered fraction is a ratio of two noise terms.
MIN_GAP = 0.05


def _gap_recovered(reinhard: dict[str, Any], metric: str) -> str:
    try:
        gap = reinhard["id_shared"][metric] - reinhard["external"][metric]
        value = reinhard["gap_recovered_fraction"][metric]
    except KeyError:
        return "n/a"
    return f"n/a (gap {gap:+.3f})" if abs(gap) < MIN_GAP else f"{value:.2f}"


def slice_b_section(files: list[Path], figures: Path) -> str:
    out = ["## Slice B — colour shift (simulation)", ""]
    for f in files:
        r = read_json(f)
        rows = []
        fig, ax = plt.subplots(figsize=(5, 4))
        for run, e in r["runs"].items():
            rows.append(
                [
                    run,
                    f"{e['slopes']['ece']:.4f}",
                    f"{e['spearman_ece_vs_delta_e']:.2f}",
                    _gap_recovered(e.get("reinhard", {}), "macro_f1"),
                    _gap_recovered(e.get("reinhard", {}), "ece"),
                ]
            )
            ax.plot(
                [c["delta_e"] for c in e["curve"]],
                [c["ece"] for c in e["curve"]],
                marker="o",
                label=f"{e['tag']} s{e['seed']}",
                alpha=0.8,
            )
        out.append(
            table(
                ["run", "ECE slope / ΔE", "Spearman(ΔE, ECE)", "gap recovered F1", "gap recovered ECE"], rows
            )
        )
        for key, v in r["by_tag"].items():
            if key.endswith("_minus_base_ece_slope"):
                out += ["", f"{key}: {fmt_ci(v, 4)} ({v['note']})."]
        ax.set(xlabel="nominal ΔE76", ylabel="ECE")
        ax.legend(fontsize=7)
        fig.tight_layout()
        figures.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures / f"{f.stem}.png", dpi=130)
        plt.close(fig)
        out += ["", f"![colour]({figures.name}/{f.stem}.png)", "", r["note"], ""]
    return "\n".join(out)


def build_report(runs_root: Path, out_dir: Path) -> Path:
    analysis = runs_root / "analysis"
    figures = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    def files(prefix: str) -> list[Path]:
        return sorted(analysis.glob(f"{prefix}*.json"))

    sections = ["# Results", "", "Generated by `cxa report` from `runs/`. Do not edit by hand.", ""]
    training = training_section(runs_root, out_dir, figures)
    if training:
        sections.append(training)
    builders = [
        ("q1_", lambda fs: q1_section(fs, figures)),
        ("q2_", lambda fs: q2_section(fs, figures)),
        ("q3_", lambda fs: q3_section(fs, figures)),
        ("q4_", lambda fs: q4_section(fs, figures)),
        ("sliceA_", lambda fs: slice_a_section(fs, figures)),
        ("sliceB_", lambda fs: slice_b_section(fs, figures)),
    ]
    for prefix, build in builders:
        found = files(prefix)
        if found:
            sections.append(build(found))
            for f in found:
                shutil.copy2(f, out_dir / f.name)
        else:
            log.info("no %s results yet", prefix.rstrip("_"))
    path = out_dir / "RESULTS.md"
    atomic_write_bytes(path, ("\n".join(sections).rstrip() + "\n").encode())
    log.info("wrote %s", path)
    return path
