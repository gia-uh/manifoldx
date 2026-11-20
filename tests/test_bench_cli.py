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


    @bench.step("cli_fail")
    def cli_fail(app):
        raise RuntimeError("broken step")


    @bench.scenario("cli_broken")
    def cli_broken(engine, p):
        return Demo(engine), [bench.hold(frames=1), cli_fail()]
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


def test_no_doc_compares_against_a_git_ref():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for p in ["AGENTS.md", "docs/chapters/16-using-every-core.md", "benchmarks/__main__.py", "benchmarks/scenes.py",
              "know-how/benchmarking.md"]:
        assert "--against main" not in (root / p).read_text(), p
