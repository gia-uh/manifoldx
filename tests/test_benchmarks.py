"""Every benchmark case runs at its smoke size, so the suites keep working as the code changes.

GPU cases skip on a machine without an adapter (tests/conftest.py); the web case reports an
error without a browser."""

import re
import shutil
from pathlib import Path

import pytest

from benchmarks import harness
from benchmarks.__main__ import SUITES, modules
from manifoldx.bench import micro
from manifoldx.bench.micro import load, run

ROOT = Path(__file__).resolve().parent.parent
CASES = load(modules(SUITES))


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(harness, "REPEAT", 1)


def test_every_suite_has_cases():
    assert {c.suite for c in CASES} == set(SUITES)


# The web suite needs Playwright and Chromium (the browser group); without them its case
# reports this error, as the harness reports any case that cannot run.
NO_BROWSER = ("RuntimeError: playwright not installed", "RuntimeError: chromium not installed")


@pytest.mark.parametrize("c", CASES, ids=lambda c: f"{c.suite}.{c.name}")
def test_case_runs_at_smoke_size(c):
    if c.suite == "web":
        (r,) = run([c], smoke=True)
        assert (r.ms or 0) > 0 or (r.error or "").startswith(NO_BROWSER), r.error
        return
    out = c.fn(c.smoke)
    ms = out["ms"] if isinstance(out, dict) else out
    assert ms > 0


def test_run_records_a_failing_case_as_an_error():
    def broken(n):
        raise AttributeError("no such API")

    c = harness.Case("x", "broken", broken, [1], 1)
    (r,) = run([c], smoke=True)
    assert r.ms is None and "no such API" in r.error


def test_compare_reports_speedup_and_missing_cases():
    from manifoldx.bench.micro import compare

    base = {"results": [{"suite": "s", "case": "a", "n": 1, "ms": 10.0, "extra": {}}]}
    head = {
        "results": [
            {"suite": "s", "case": "a", "n": 1, "ms": 2.0, "extra": {}},
            {"suite": "s", "case": "new", "n": 1, "ms": 1.0, "extra": {}},
        ]
    }
    table = compare(base, head, "main")
    assert "| s.a | 1 | 10.0 | 2.0 | 5.0× |" in table
    assert "| s.new | 1 | n/a | 1.0 | – |" in table


def test_the_old_harness_is_the_package_module():
    assert harness is micro and harness.case is micro.case


def test_against_refuses_a_directory_without_manifoldx(tmp_path):
    with pytest.raises(SystemExit, match="src/manifoldx"):
        micro.against(["benchmarks.suites.ecs"], tmp_path, ["--case", "spawn"])


def test_against_runs_the_other_source(tmp_path):
    other = tmp_path / "other"
    shutil.copytree(ROOT / "src", other / "src", ignore=shutil.ignore_patterns("bench", "__pycache__"))
    table = micro.against(["benchmarks.suites.ecs"], other, ["--case", "spawn", "--sizes", "10"])
    assert "| ecs.spawn | 10 |" in table and "error" not in table


def test_bench_never_calls_git():
    for p in [*(ROOT / "src/manifoldx/bench").rglob("*.py"), ROOT / "src/manifoldx/counters.py"]:
        assert not re.search(r"\bgit\b", p.read_text(), re.I), p


def test_against_interleaves_rounds(tmp_path, monkeypatch):
    other = tmp_path / "other"
    shutil.copytree(ROOT / "src", other / "src", ignore=shutil.ignore_patterns("bench", "__pycache__"))
    seen = []
    real = micro._child
    monkeypatch.setattr(micro, "_child", lambda modules, src, *a, **k: seen.append(src) or real(modules, src, *a, **k))
    base, head = micro.against_results(["benchmarks.suites.ecs"], other, ["--case", "spawn", "--sizes", "10"],
                                       repeat=2)
    here = Path(micro.__file__).resolve().parents[2]
    assert seen == [other.resolve() / "src", here, here, other.resolve() / "src"]
    r = head["results"][0]
    assert r["spread"][0] <= r["ms"] <= r["spread"][1] and len(head["rounds"]) == 2


def test_compare_notes_close_ratios_and_load():
    from manifoldx.bench.micro import compare

    base = {"results": [{"suite": "s", "case": "a", "n": 1, "ms": 10.0, "spread": [9.0, 11.0], "extra": {}}],
            "load_max": 7.0, "cpus": 8}
    head = {"results": [{"suite": "s", "case": "a", "n": 1, "ms": 9.0, "spread": [8.5, 9.5], "extra": {}}],
            "load_max": 0.2, "cpus": 8}
    row = compare(base, head, "main").splitlines()[2]
    assert "10.0 (9.0–11.0)" in row and "needs a quiet re-run" in row and "taken under load" in row
