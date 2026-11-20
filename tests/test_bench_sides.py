"""Sides and sets: a sweep runs each side in fresh processes, interleaved, into one set folder."""

import json

from manifoldx.bench.cli import main
from manifoldx.bench.sides import order
from test_bench_cli import project  # noqa: F401


def test_order_alternates_each_round():
    assert order(2, 3) == [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2)]
    assert order(3, 1) == [(0, 0), (1, 0), (2, 0)]


def test_a_sweep_runs_its_sides_interleaved_in_fresh_processes(project, capsys):  # noqa: F811
    assert main(["run", "cli_demo", "--param", "frames=2,3", "--repeat", "2", "--smoke", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    folder = project / "bench/runs" / out["set"]
    s = json.loads((folder / "set.json").read_text())
    assert [x["params"] for x in s["sides"]] == [{"frames": "2"}, {"frames": "3"}]
    assert [(r["side"], r["round"]) for r in s["runs"]] == [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert len({r["pid"] for r in s["runs"]}) == 4  # four fresh processes
    for r in s["runs"]:
        run = json.loads((folder / r["folder"] / "run.json").read_text())
        assert run["params"]["frames"] == int(s["sides"][r["side"]]["params"]["frames"])
    assert not [p for p in (project / "bench/runs").iterdir() if p != folder]  # every run moved into the set


def test_one_value_runs_in_process_as_before(project, capsys):  # noqa: F811
    assert main(["run", "cli_demo", "--param", "frames=2", "--json"]) == 0
    assert "set" not in json.loads(capsys.readouterr().out)


def test_repeat_alone_is_refused(project, capsys):  # noqa: F811
    assert main(["run", "cli_demo", "--repeat", "3"]) == 2
    assert "--repeat" in capsys.readouterr().err


def test_a_bad_swept_param_is_refused_before_running(project, capsys):  # noqa: F811
    assert main(["run", "cli_broken", "--param", "x=1,2"]) == 2
    assert "no param" in capsys.readouterr().err


def test_a_failing_side_fails_the_set_and_keeps_the_others(project, capsys):  # noqa: F811
    assert main(["run", "cli_fails_on", "--param", "bad=0,1", "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    s = json.loads((project / "bench/runs" / out["set"] / "set.json").read_text())
    ok, failed = s["runs"]
    assert ok["error"] is None and ok["exit"] == 0
    assert "broken step" in failed["error"] and failed["exit"] == 1 and failed["folder"]
