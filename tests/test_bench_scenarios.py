"""manifoldx's own scenarios run through the runner, at smoke size."""

import json
from pathlib import Path

import pytest

from manifoldx.bench.cli import load_config, main
from manifoldx.bench.folder import write_run
from manifoldx.bench.runner import run_scenario
from manifoldx.bench.scenario import SCENARIOS

ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(ROOT)
__import__(CONFIG.scenarios)
OURS = sorted(n for n, s in SCENARIOS.items() if s.fn.__module__ == CONFIG.scenarios)


def test_quad_zoom_settles_and_shoots(tmp_path):
    run = run_scenario(SCENARIOS["quad_zoom"], size=(160, 90), smoke=True)
    assert run.error is None, run.error
    assert "zoom" in run.summary["phases"] and "street" in run.shots
    folder = write_run(run, tmp_path, None)
    assert (folder / "shots" / "street.png").exists()


@pytest.mark.parametrize("loop", ["serial", "pipelined"])
def test_meshes_spin_serial_and_pipelined(loop):
    run = run_scenario(SCENARIOS["meshes_spin"], size=(64, 48), smoke=True, loop=loop, overrides={"n": "200"})
    assert run.error is None, run.error
    assert {"update", "render"} <= set(run.summary["phases"]["spin"]["split_ms"])


@pytest.mark.parametrize("name", OURS)
def test_every_declared_scenario_runs_at_smoke_size(name):
    assert run_scenario(SCENARIOS[name], size=(64, 48), smoke=True).error is None


def test_the_repo_config_lists_every_micro_suite():
    suites = {p.stem for p in (ROOT / "benchmarks" / "suites").glob("*.py") if p.stem != "__init__"}
    assert {m.rsplit(".", 1)[1] for m in CONFIG.micro} == suites


def test_bench_list_in_the_repo(monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    assert main(["list", "--json"]) == 0
    assert {"quad_zoom", "meshes_spin"} <= {s["name"] for s in json.loads(capsys.readouterr().out)["scenarios"]}
