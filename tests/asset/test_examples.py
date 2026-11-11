"""The two Parts examples, built and linted from their committed blueprints: no network and no image model."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "models"


def _load(name):
    path = EXAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_asset_example", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, name.capitalize())


def _codes(result):
    return [finding.code for finding in result.report.findings]


def test_the_hydrant_builds_and_lints_its_committed_blueprints():
    result = _load("hydrant")().run(stages=("build", "lint"))
    assert sorted(result.scene.meshes) == ["barrel1", "base1", "bonnet1", "pumper1", "side1", "side2"]
    assert "A-OPEN" not in _codes(result) and "A-FLIP" not in _codes(result)
    assert result.report.exit_code == 0
    for mesh in result.scene.meshes.values():
        assert np.all(np.isfinite(np.asarray(mesh.positions)))


def test_the_hydrants_report_is_deterministic():
    first = _load("hydrant")().run(stages=("build", "lint"))
    second = _load("hydrant")().run(stages=("build", "lint"))
    assert first.report.to_json() == second.report.to_json()
    assert list(first.scene.meshes) == list(second.scene.meshes)
    for name, mesh in first.scene.meshes.items():
        assert np.array_equal(np.asarray(mesh.positions), np.asarray(second.scene.meshes[name].positions))


def test_the_armchair_builds_and_lints_its_committed_blueprints():
    result = _load("armchair")().run(stages=("build", "lint"))
    assert sorted(result.scene.meshes) == ["arms1", "arms2", "back1", "legs1", "legs2", "legs3", "legs4",
                                           "seat1"]
    assert "A-OPEN" not in _codes(result) and "A-FLIP" not in _codes(result)
    # The floats the first lint found are closed by attach, which says how far it moved.
    assert "A-FLOAT" not in _codes(result)
    moves = [finding for finding in result.report.findings if finding.code == "A-ATTACH"]
    assert {finding.part for finding in moves} == {"arms", "legs"}
    assert all(finding.level == "info" for finding in moves)
