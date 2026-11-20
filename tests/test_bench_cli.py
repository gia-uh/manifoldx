"""manifoldx bench: list, run, show and micro, in a small project of its own."""

import json
import sys
import textwrap

import pytest

import manifoldx.cli
from manifoldx.bench.cli import load_config, main

SCENARIOS = textwrap.dedent('''
    import numpy as np
    from manifoldx import bench
    from manifoldx.components import Material, Mesh, Transform
    from manifoldx.resources import StandardMaterial, cube


    class Demo:
        def __init__(self, engine):
            engine.camera.fit(10)
            engine.spawn(Mesh(cube(1, 1, 1)), Material(StandardMaterial("#ff8800")),
                         Transform(pos=np.zeros((1, 3), np.float32)), n=1)


    @bench.scenario("cli_demo", params={"frames": 3}, sizes={"smoke": 2})
    def cli_demo(engine, p):
        """A cube, held a few frames."""
        return Demo(engine), [bench.mark("hold"), bench.hold(frames=p.frames), bench.shot("cube")]


    cli_demo.check(counters_draws=1000)


    @bench.scenario("cli_tight", sizes={"smoke": 3})
    def cli_tight(engine, p):
        """A budget no frame meets."""
        return Demo(engine), [bench.mark("tight"), bench.hold(frames=3)]


    cli_tight.check(p95_ms=0.001)


    @bench.step("cli_fail")
    def cli_fail(app):
        raise RuntimeError("broken step")


    @bench.scenario("cli_broken")
    def cli_broken(engine, p):
        return Demo(engine), [bench.hold(frames=1), cli_fail()]


    @bench.scenario("cli_fails_on", params={"bad": 0}, sizes={"smoke": 2})
    def cli_fails_on(engine, p):
        """Fails in a step when bad is set: one side of a sweep fails, the other runs."""
        return Demo(engine), [bench.hold(frames=1), *([cli_fail()] if p.bad else [])]
''')


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(textwrap.dedent('''
        [project]
        name = "demoapp"

        [tool.manifoldx.bench]
        scenarios = "cli_demo_scenarios"
    '''))
    (tmp_path / "cli_demo_scenarios.py").write_text(SCENARIOS)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("MANIFOLDX_BENCH_LABEL", raising=False)
    import importlib

    registry = importlib.import_module("manifoldx.bench.scenario")

    for reg in (registry.SCENARIOS, registry.STEPS):
        for k in [k for k, v in reg.items() if v.fn.__module__ in ("cli_demo_scenarios", "noisy")]:
            del reg[k]
    sys.modules.pop("cli_demo_scenarios", None)
    sys.modules.pop("noisy", None)
    return tmp_path


def test_config_reads_the_table_and_its_defaults(project):
    c = load_config(project / "sub" if (project / "sub").mkdir() is None else project)
    assert c.root == project and c.scenarios == "cli_demo_scenarios" and c.app_name == "demoapp"
    assert c.runs == project / "bench" / "runs" and c.history == project / "bench" / "history.jsonl"
    assert c.micro == []


def test_list_shows_scenarios_engine_steps_and_app_steps(project, capsys):
    assert main(["list", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    names = {s["name"] for s in out["scenarios"]}
    assert {"cli_demo", "cli_broken"} <= names
    assert {"camera", "hold", "zoom", "shot"} <= set(out["steps"]["engine"])
    assert "cli_fail" in out["steps"]["app"]


def test_run_json_prints_only_the_summary(project, capsys):
    assert main(["run", "cli_demo", "--size", "64x48", "--json"]) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["scenario"] == "cli_demo" and summary["phases"]["hold"]["frames"] == 4  # 3 held + the shot
    assert "frame" in captured.err


def test_run_writes_folder_and_history(project, capsys):
    assert main(["run", "cli_demo", "--size", "64x48", "--param", "frames=2", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    folder = project / "bench" / "runs" / summary["folder"]
    assert (folder / "run.json").exists() and (folder / "shots" / "cube.png").exists()
    lines = (project / "bench" / "history.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["params"] == {"frames": 2}


def test_run_exits_1_when_a_step_fails(project, capsys):
    assert main(["run", "cli_broken", "--size", "64x48"]) == 1
    assert "broken step" in capsys.readouterr().err


def test_run_outside_a_project_says_what_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["run", "x"]) == 2
    assert "pyproject.toml" in capsys.readouterr().err


def test_a_project_without_the_table_says_so(tmp_path, monkeypatch, capsys):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "plain"\n')
    monkeypatch.chdir(tmp_path)
    assert main(["run", "x"]) == 2
    assert "[tool.manifoldx.bench]" in capsys.readouterr().err
    assert main(["list", "--json"]) == 0  # the engine's steps still list


def test_an_unknown_scenario_is_named(project, capsys):
    assert main(["run", "nope"]) == 2
    err = capsys.readouterr().err
    assert "nope" in err and "cli_demo" in err


def test_label_comes_from_the_environment(project, monkeypatch, capsys):
    monkeypatch.setenv("MANIFOLDX_BENCH_LABEL", "ci-1234")
    main(["run", "cli_demo", "--size", "64x48", "--smoke", "--json"])
    assert json.loads(capsys.readouterr().out)["label"] == "ci-1234"


def test_size_must_be_w_x_h(project, capsys):
    with pytest.raises(SystemExit) as e:
        main(["run", "cli_demo", "--size", "abc"])
    assert e.value.code == 2 and "WxH" in capsys.readouterr().err


def test_a_json_scenario_file_runs(project, capsys):
    (project / "demo_app.py").write_text("from cli_demo_scenarios import Demo\n\ndef make(engine, p):\n    return Demo(engine)\n")
    (project / "s.json").write_text(json.dumps({"name": "from_file", "app": "demo_app:make",
                                                "steps": [{"hold": {"frames": 2}}]}))
    assert main(["run", "s.json", "--size", "64x48", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["phases"]["main"]["frames"] == 2


def test_show_reads_a_run_back(project, capsys):
    main(["run", "cli_demo", "--size", "64x48", "--smoke", "--json"])
    folder = json.loads(capsys.readouterr().out)["folder"]
    assert main(["show", folder, "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["scenario"] == "cli_demo" and shown["shots"][0].endswith("cube.png")


def test_manifoldx_cli_has_bench(project, capsys):
    assert manifoldx.cli.main(["bench", "list", "--json"]) == 0
    assert "cli_demo" in capsys.readouterr().out


def test_run_json_stdout_is_json_even_when_the_app_prints(project, capsys):
    (project / "noisy.py").write_text(
        "from manifoldx import bench\nfrom cli_demo_scenarios import Demo\n\n"
        "@bench.scenario('noisy')\ndef noisy(engine, p):\n    print('hello from the app')\n"
        "    return Demo(engine), [bench.hold(frames=1)]\n")
    (project / "pyproject.toml").write_text('[project]\nname = "demoapp"\n[tool.manifoldx.bench]\nscenarios = "noisy"\n')
    assert main(["run", "noisy", "--size", "64x48", "--json"]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["scenario"] == "noisy" and "hello from the app" in out.err


def test_list_survives_a_missing_scenarios_module(project, capsys):
    (project / "pyproject.toml").write_text('[tool.manifoldx.bench]\nscenarios = "no_such_module"\n')
    assert main(["list", "--json"]) == 0
    assert "hold" in json.loads(capsys.readouterr().out)["steps"]["engine"]


def test_run_with_a_missing_scenarios_module_exits_2(project, capsys):
    (project / "pyproject.toml").write_text('[tool.manifoldx.bench]\nscenarios = "no_such_module"\n')
    assert main(["run", "x"]) == 2
    assert "no_such_module" in capsys.readouterr().err


def test_a_missing_or_broken_json_file_exits_2(project, capsys):
    assert main(["run", "missing.json"]) == 2
    (project / "bad.json").write_text("{not json")
    assert main(["run", "bad.json"]) == 2
    assert "bad.json" in capsys.readouterr().err


def test_a_zero_step_is_a_usage_error(project):
    with pytest.raises(SystemExit) as e:
        main(["run", "cli_demo", "--step", "1/0"])
    assert e.value.code == 2


def test_a_json_scenario_can_call_app_steps_of_the_scenarios_module(project, capsys):
    (project / "other_app.py").write_text("from types import SimpleNamespace\n\ndef make(engine, p):\n"
                                          "    engine.camera.fit(10)\n    return SimpleNamespace(count=0)\n")
    (project / "uses_step.json").write_text(json.dumps({"app": "other_app:make", "steps": [{"cli_fail": []}]}))
    assert main(["run", "uses_step.json", "--size", "64x48", "--json"]) == 1  # the step ran and raised
    assert "broken step" in capsys.readouterr().err


def test_micro_against_json_is_structured(tmp_path, capsys):
    import shutil
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    other = tmp_path / "other"
    shutil.copytree(root / "src", other / "src", ignore=shutil.ignore_patterns("bench", "__pycache__"))
    import os

    cwd = os.getcwd()
    os.chdir(root)
    try:
        assert main(["micro", "benchmarks.suites.ecs", "--case", "spawn", "--sizes", "10", "--against", str(other),
                     "--json"]) == 0
    finally:
        os.chdir(cwd)
    out = json.loads(capsys.readouterr().out)
    assert {"base", "head", "table"} <= out.keys() and out["head"]["results"][0]["case"] == "spawn"


CASES = textwrap.dedent('''
    from manifoldx.bench.micro import case


    @case("t", "boom", sizes=[1], smoke=1)
    def boom(n):
        raise RuntimeError("boom")


    @case("t", "fine", sizes=[1], smoke=1)
    def fine(n):
        return 0.5
''')


@pytest.fixture
def micro_project(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    shutil.copytree(root / "src", tmp_path / "other" / "src", ignore=shutil.ignore_patterns("bench", "__pycache__"))
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "cases"\n')
    (tmp_path / "t_cases.py").write_text(CASES)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_micro_against_fails_when_a_case_errors(micro_project, capsys):
    other = str(micro_project / "other")
    assert main(["micro", "t_cases", "--against", other, "--repeat", "1"]) == 1
    assert main(["micro", "t_cases", "--case", "fine", "--against", other, "--repeat", "1"]) == 0


def test_micro_against_profiles(micro_project, capsys):
    assert main(["micro", "t_cases", "--case", "fine", "--against", str(micro_project / "other"), "--repeat", "1",
                 "--profile"]) == 0
    assert capsys.readouterr().err.count("--- profile t.fine") == 2


def test_no_doc_compares_against_a_git_ref():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for p in ["AGENTS.md", "docs/chapters/16-using-every-core.md", "benchmarks/__main__.py", "benchmarks/scenes.py",
              "know-how/benchmarking.md"]:
        assert "--against main" not in (root / p).read_text(), p


def test_folders_in_the_config_are_sized(project, capsys):
    (project / "cache").mkdir()
    (project / "cache" / "a.bin").write_bytes(b"\0" * 2048)
    with (project / "pyproject.toml").open("a") as f:
        f.write('folders = ["cache"]\n')
    assert main(["run", "cli_demo", "--size", "64x48", "--json"]) == 0
    sizes = json.loads(capsys.readouterr().out)["memory"]["folders_mb"]
    assert sizes["cache"]["start"] > 0


def test_a_folder_outside_the_project_is_refused(project, capsys):
    with (project / "pyproject.toml").open("a") as f:
        f.write('folders = ["../elsewhere"]\n')
    assert main(["run", "cli_demo", "--size", "64x48", "--json"]) == 2
    assert "folders" in capsys.readouterr().err


def test_progress_lines_count_steps(project, capsys):
    import re

    assert main(["run", "cli_demo", "--size", "64x48", "--json"]) == 0
    err = capsys.readouterr().err
    assert re.search(r"^frame \d+ phase hold step \d/3$", err, re.M), err


def _a_run(capsys):
    assert main(["run", "cli_demo", "--size", "64x48", "--json"]) == 0
    return json.loads(capsys.readouterr().out)


def test_show_picks_one_probe_of_one_phase(project, capsys):
    run = _a_run(capsys)
    assert main(["show", run["folder"], "--probe", "counters", "--phase", "hold", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == run["phases"]["hold"]["counters"]


def test_show_a_probe_without_a_phase_gives_every_phase(project, capsys):
    run = _a_run(capsys)
    assert main(["show", run["folder"], "--probe", "split", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"hold": run["phases"]["hold"]["split_ms"]}


def test_show_run_level_probes(project, capsys):
    run = _a_run(capsys)
    assert main(["show", run["folder"], "--probe", "memory", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == run["memory"]


def test_show_an_unknown_phase_lists_the_phases(project, capsys):
    run = _a_run(capsys)
    assert main(["show", run["folder"], "--probe", "counters", "--phase", "nope"]) == 2
    assert "hold" in capsys.readouterr().err


def test_the_table_has_a_line_of_counters(project, capsys):
    assert main(["run", "cli_demo", "--size", "64x48"]) == 0
    out = capsys.readouterr().out
    assert "draws" in out and "bytes_uploaded" in out


def test_show_reads_a_step_1_run(project, capsys):
    run = _a_run(capsys)
    folder = project / "bench" / "runs" / run["folder"]
    old = json.loads((folder / "run.json").read_text())
    for ph in old["phases"].values():  # a step-1 run.json: no draw time, no counters
        for k in ("draw_p50_ms", "counters", "streamer", "spans_ms"):
            ph.pop(k, None)
    old.pop("schema", None)
    (folder / "run.json").write_text(json.dumps(old))
    assert main(["show", run["folder"]]) == 0
    assert "hold" in capsys.readouterr().out


def test_the_summary_carries_its_schema(project, capsys):
    assert _a_run(capsys)["schema"] == 4


def test_check_runs_scenarios_with_checks_at_smoke_size(project, capsys):
    assert main(["check", "cli_demo", "--json"]) == 0
    (out,) = json.loads(capsys.readouterr().out)
    assert out["scenario"] == "cli_demo" and out["error"] is None
    assert [(c["phase"], c["ok"]) for c in out["checks"]] == [("hold", True)]
    assert (project / "bench" / "runs" / out["folder"] / "run.json").exists()


def test_check_exits_1_and_says_why(project, capsys):
    assert main(["check", "cli_tight"]) == 1
    captured = capsys.readouterr()
    assert "FAIL" in captured.out
    assert all(w in captured.err for w in ("check", "p95_ms", "phase tight", "frame"))


def test_check_without_names_runs_the_scenarios_with_checks(project, capsys):
    assert main(["check", "--json"]) == 1
    assert [o["scenario"] for o in json.loads(capsys.readouterr().out)] == ["cli_demo", "cli_tight"]


def test_run_exits_1_on_a_failed_check(project, capsys):
    assert main(["run", "cli_tight", "--size", "64x48"]) == 1
    captured = capsys.readouterr()
    assert "check p95_ms" in captured.out and "check p95_ms" in captured.err
    (folder,) = (project / "bench" / "runs").iterdir()
    assert json.loads((folder / "run.json").read_text())["checks"][0]["ok"] is False


def test_budgets_file_adds_checks(project, capsys):
    with open(project / "pyproject.toml", "a") as f:
        f.write('budgets = "bench/budgets.toml"\n')
    (project / "bench").mkdir()
    (project / "bench" / "budgets.toml").write_text("[cli_demo]\np95_ms = 0.001\n")
    assert main(["check", "cli_demo"]) == 1
    assert "p95_ms" in capsys.readouterr().err
    (project / "bench" / "budgets.toml").write_text("[no_such_scenario]\np95_ms = 1\n")
    assert main(["check", "cli_demo"]) == 2
    assert "no_such_scenario" in capsys.readouterr().err


def test_run_all_runs_every_scenario(project, capsys):
    assert main(["run", "--all", "--smoke", "--size", "64x48", "--json"]) == 1  # cli_broken fails
    out = json.loads(capsys.readouterr().out)
    names = [o["scenario"] for o in out]
    assert names == [n for n in ["cli_demo", "cli_tight", "cli_broken", "cli_fails_on"] if n in names] and len(names) >= 4


def test_run_needs_a_scenario_or_all(project, capsys):
    assert main(["run"]) == 2
    assert "--all" in capsys.readouterr().err


def _interrupted_after_one(monkeypatch):
    from manifoldx.bench import runner

    real, calls = runner.run_scenario, []

    def fake(*a, **kw):
        run = real(*a, **kw)
        calls.append(run.summary["scenario"])
        run.error = run.summary["error"] = "interrupted"  # as a Ctrl-C during the run leaves it
        return run

    monkeypatch.setattr(runner, "run_scenario", fake)
    return calls


@pytest.mark.parametrize("argv", [["run", "--all", "--smoke", "--size", "64x48"], ["check"]])
def test_ctrl_c_ends_a_run_of_many_scenarios(project, monkeypatch, capsys, argv):
    calls = _interrupted_after_one(monkeypatch)
    assert main(argv) == 1
    assert len(calls) == 1 and "interrupted" in capsys.readouterr().err
