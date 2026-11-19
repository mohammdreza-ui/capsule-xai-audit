"""The run_all orchestrator: scheduling, reuse, failure isolation, time budget, learning check."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run_all() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_all", REPO / "scripts" / "run_all.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_all"] = module
    spec.loader.exec_module(module)
    return module


# `cxa --help` succeeds in a second; an unknown command fails just as fast.
OK, BAD = ["--help"], ["no-such-command"]


def test_scheduler_parallel_gpus_reuse_and_failure_isolation(run_all: ModuleType, tmp_path: Path) -> None:
    out = tmp_path / "existing.json"
    out.write_text("{}")
    J = run_all.Job
    jobs = [
        J("h1", OK, "heavy"),
        J("h2", OK, "heavy"),
        J("h_done", OK, "heavy", done=lambda: True),
        J("h_fail", BAD, "heavy"),
        J("light", OK, "light", deps=["h1"]),
        J("light_reused", OK, "light", deps=["h_done"], output=out),
        J("cpu_blocked", OK, "cpu", deps=["h_fail"]),
        J("cpu_ok", OK, "cpu", deps=["h1", "h2", "light"]),
    ]
    sched = run_all.Scheduler(jobs, 2, 1, 2, tmp_path, tmp_path / "status.json", None)
    assert not sched.run()  # one job failed
    status = {j.name: j.status for j in jobs}
    assert status == {
        "h1": "ok",
        "h2": "ok",
        "h_done": "reused",
        "h_fail": "failed",
        "light": "ok",
        "light_reused": "reused",
        "cpu_blocked": "blocked",
        "cpu_ok": "ok",
    }
    assert {jobs[0].gpu, jobs[1].gpu} == {0, 1}  # the first two trainings ran on different GPUs
    assert json.loads((tmp_path / "status.json").read_text())["jobs"]["h_fail"]["status"] == "failed"


def test_no_job_starts_after_the_time_budget(run_all: ModuleType, tmp_path: Path) -> None:
    job = run_all.Job("late", OK, "heavy")
    run_all.Scheduler([job], 1, 0, 1, tmp_path, tmp_path / "s.json", budget_s=0.0).run()
    assert job.status == "deferred"


def test_learning_check(run_all: ModuleType, tmp_path: Path) -> None:
    run = tmp_path / "train" / "official" / "resnet50" / "base" / "fold1" / "seed0"
    run.mkdir(parents=True)

    def write(losses: list[float]) -> None:
        records = [{"loss": v, "global_step": 10 * (i + 1)} for i, v in enumerate(losses)]
        (run / "train_steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    write([2.4, 2.3, 1.6, 1.2, 1.0])
    assert "2.350 -> 1.100" in run_all.check_learning(tmp_path, "resnet50", 0)
    write([2.4, 2.4, 2.39, 2.38, 2.38])
    with pytest.raises(run_all.PhaseFailed, match="did not fall"):
        run_all.check_learning(tmp_path, "resnet50", 0)


def test_full_profile_plan(run_all: ModuleType, tmp_path: Path) -> None:
    layout = run_all.Layout(*(tmp_path / n for n in ("w", "i", "c", "r", "m", "res", "l", "s", "x")))
    jobs = run_all.build_jobs(run_all.PROFILES["full"], layout, layout.runs, smoke=False, workers=2)
    kinds = [j.kind for j in jobs]
    assert kinds.count("heavy") == 2 * 3 * 5 + 5  # backbones x protocols x seeds + lab-aug seeds
    assert {j.name for j in jobs if j.kind == "light"} == {
        "fewshot",
        "xai:resnet50",
        "xai:convnext_tiny",
        "slice_a",
        "slice_b",
    }
    report = next(j for j in jobs if j.name == "report")
    assert len(report.deps) == len(jobs) - 1
