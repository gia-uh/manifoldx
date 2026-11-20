"""The pytest plugin: each scenario of the project is a test, `<scenarios file>::test_bench[<name>]`.

It is never loaded by installing manifoldx; a project turns it on in its pyproject.toml,

    [tool.pytest.ini_options]
    testpaths = ["tests", "myapp/scenarios.py"]   # the module [tool.manifoldx.bench] scenarios names
    addopts = "-p manifoldx.bench.plugin"

or with `pytest -p manifoldx.bench.plugin`. Each test runs its scenario at smoke size, 160x90, with its
checks and the budgets file's, writes the run folder under the test's temporary folder, and fails on the
run's error or a failed check, naming the check, the phase and the frame.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

SIZE = (160, 90)


class BenchFailure(Exception):
    """A scenario's run failed or one of its checks did; `lines` says which, `folder` is the run folder."""

    def __init__(self, lines: list[str], folder: Path):
        super().__init__("\n".join(lines))
        self.lines, self.folder = lines, folder


def _config(config):
    """The project's [tool.manifoldx.bench] and the file of its scenarios module, read once per session."""
    if not hasattr(config, "_manifoldx_bench"):
        from manifoldx.bench.cli import ConfigError, load_config

        found = None
        try:
            bench = load_config(Path(config.rootpath))
        except ConfigError:
            bench = None
        if bench is not None and bench.scenarios:
            if str(bench.root) not in sys.path:
                sys.path.insert(0, str(bench.root))
            try:
                spec = importlib.util.find_spec(bench.scenarios)
            except ImportError:
                spec = None
            if spec is not None and spec.origin:
                found = Path(spec.origin).resolve()
        config._manifoldx_bench = (bench, found)
    return config._manifoldx_bench


def pytest_collect_file(file_path: Path, parent):
    bench, path = _config(parent.config)
    if path is not None and Path(file_path).resolve() == path:
        return ScenarioFile.from_parent(parent, path=file_path)
    return None


class ScenarioFile(pytest.File):
    """The scenarios module: one test per scenario it defines."""

    def collect(self):
        from manifoldx.bench.scenario import SCENARIOS

        bench, _ = _config(self.config)
        importlib.import_module(bench.scenarios)
        for name, sc in SCENARIOS.items():
            if sc.fn.__module__ == bench.scenarios:
                yield ScenarioItem.from_parent(self, name=f"test_bench[{name}]", scenario=name)


class ScenarioItem(pytest.Item):
    """One scenario, run at smoke size with its checks."""

    def __init__(self, *, scenario: str, **kw):
        super().__init__(**kw)
        self.scenario = scenario

    def runtest(self):
        from manifoldx.bench.cli import _budgets
        from manifoldx.bench.folder import write_run
        from manifoldx.bench.runner import run_scenario
        from manifoldx.bench.scenario import SCENARIOS

        bench, _ = _config(self.config)
        run = run_scenario(SCENARIOS[self.scenario], size=SIZE, smoke=True, app_name=bench.app_name,
                           budgets=_budgets(bench).get(self.scenario))
        base = self.config._tmp_path_factory.mktemp(f"bench-{self.scenario}")
        folder = write_run(run, base, None)
        lines = [f"the run failed: {run.error}"] if run.error else []
        lines += [c["message"] for c in run.summary["checks"] if not c["ok"]]
        if lines:
            raise BenchFailure(lines, folder)

    def repr_failure(self, excinfo, style=None):
        if isinstance(excinfo.value, BenchFailure):
            return "\n".join([*excinfo.value.lines, f"run folder: {excinfo.value.folder}"])
        return super().repr_failure(excinfo, style)

    def reportinfo(self):
        return self.path, None, f"scenario {self.scenario}"
