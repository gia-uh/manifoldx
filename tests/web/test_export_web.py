"""manifoldx export --web: the app runs its setup in Python, then the folder holds the
transpiled code, the state it starts from, the scene and a report, and loads in Node."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import manifoldx as mx
from manifoldx.cli import main

RIG = """
class Rig:
    def __init__(self):
        self.x = 0.0
        self.speed = 2.0
        self.notes = "setup only"

    def step(self, dt: float):
        self.x = self.x + self.speed * dt

    def describe(self):
        return self.notes
"""

APP = """
import manifoldx as mx
from manifoldx.gui import Panel, ValueDisplay
from webx_rig import Rig

engine = mx.Engine("web", width=320, height=200)
engine.state.setdefault("level", 2)
rig = Rig()


def walk(query, dt):
    rig.step(dt)


def label():
    return f"x {rig.x:.1f}"


def setup_only():
    return "never in the browser"


@engine.on("startup")
def build(_):
    engine.spawn(
        mx.components.Mesh(mx.resources.cube(1, 1, 1)),
        mx.components.Material(mx.resources.BasicMaterial("#ff0000")),
        mx.components.Transform(pos=(0, 1, 0)),
    )
    engine.state["cam"] = {"yaw": 0.5, "dist": 12.0}
    engine.state["names"] = ["a", "b"]
    rig.x = 1.0


engine.system(walk)
engine.gui.append(Panel(children=[ValueDisplay(label)]))

if __name__ == "__main__":
    engine.run()
"""

BARE = """
import manifoldx as mx

engine = mx.Engine("bare", width=64, height=64)
engine.spawn(
    mx.components.Mesh(mx.resources.cube(1, 1, 1)),
    mx.components.Material(mx.resources.BasicMaterial("#ffffff")),
    mx.components.Transform(pos=(0, 0, 0)),
)
engine.run()
"""

ASYNC = """
import asyncio
import manifoldx as mx

engine = mx.Engine("async", width=64, height=64)


@engine.on("startup")
async def load(_):
    await asyncio.sleep(0)
    engine.state["loaded"] = "yes"


engine.run()
"""

BAD = """
import manifoldx as mx

engine = mx.Engine("bad", width=64, height=64)


def walk(query, dt):
    try:
        return 1
    except ValueError:
        return 2


engine.system(walk)
engine.run()
"""

LEFT_OUT = """
import numpy as np
import manifoldx as mx

engine = mx.Engine("left", width=64, height=64)
engine.state["kept"] = [1, 2.5, "three"]
engine.state[("terrain", 7)] = 1.0
engine.state["grid"] = np.zeros((2, 2))
engine.run()
"""

# Typed arrays and Sets become arrays, a Map (a dict) an object, as JSON.stringify sees them.
EXPECTED_FILES = {
    "app/__main__.js",
    "app/__main__.js.map",
    "app/webx_rig.js",
    "app/webx_rig.js.map",
    "pyrt.js",
    "manifoldx.js",
    "replay.mjs",
    "state.json",
    "scene.glb",
    "scene.export-report.json",
    "report.json",
}


@pytest.fixture
def app_dir(tmp_path):
    """A folder for the app's files; the modules it imports are forgotten afterwards."""
    folder = tmp_path / "src"
    folder.mkdir()
    yield folder
    for name, m in list(sys.modules.items()):
        if str(getattr(m, "__file__", None) or "").startswith(str(folder)):
            del sys.modules[name]


def _export(folder: Path, out: Path, src: str, name: str = "app.py", **modules) -> int:
    for module, text in modules.items():
        (folder / f"{module}.py").write_text(text)
    (folder / name).write_text(src)
    return main(["export", "--web", str(folder / name), str(out)])


def _node(out: Path) -> dict:
    done = subprocess.run(
        ["node", str(out / "replay.mjs")], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _files(out: Path) -> set[str]:
    return {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}


def test_writes_every_file(app_dir, tmp_path):
    out = tmp_path / "out"
    assert _export(app_dir, out, APP, webx_rig=RIG) == 0
    assert _files(out) == EXPECTED_FILES
    js = (out / "app" / "__main__.js").read_text()
    assert js.rstrip().endswith("//# sourceMappingURL=__main__.js.map")
    assert json.loads((out / "app" / "__main__.js.map").read_text())["sources"] == ["app.py"]
    assert 'import * as $rt from "../pyrt.js";' in js  # one runtime, beside manifoldx.js


def test_the_report_lists_roots_transpiled_and_setup(app_dir, tmp_path):
    out = tmp_path / "out"
    assert _export(app_dir, out, APP, webx_rig=RIG) == 0
    report = json.loads((out / "report.json").read_text())
    assert [r["label"] for r in report["roots"]] == [
        "system walk",
        "ValueDisplay.getter label",
    ]
    assert {"__main__.walk", "__main__.label", "webx_rig.Rig.step"} <= set(report["transpiled"])
    assert {"__main__.setup_only", "__main__.build", "webx_rig.Rig.describe"} <= set(
        report["setup"]
    )
    assert not set(report["transpiled"]) & set(report["setup"])
    assert report["modules"] == {"__main__": "app/__main__.js", "webx_rig": "app/webx_rig.js"}
    assert report["state_left_out"] == []


def test_the_folder_loads_in_node_with_the_state_python_left(app_dir, tmp_path):
    out = tmp_path / "out"
    assert _export(app_dir, out, APP, webx_rig=RIG) == 0
    state = json.loads((out / "state.json").read_text())
    assert state["engine"] == {
        "w": 320,
        "h": 200,
        "camera": {
            "position": [0.0, 1.0, 2.0],
            "target": [0.0, 0.0, 0.0],
            "fov": 60.0,
            "up": [0.0, 1.0, 0.0],
            "near": 0.1,
            "far": 1000.0,
        },
    }
    assert state["modules"] == ["__main__", "webx_rig"]
    got = _node(out)
    assert got["roots"] == ["system walk", "ValueDisplay.getter label"]
    # engine.state as Python had it after startup, in Python's key order.
    assert got["state"] == {"level": 2, "cam": {"yaw": 0.5, "dist": 12.0}, "names": ["a", "b"]}
    assert list(got["state"]) == ["level", "cam", "names"]
    assert got["gui"] == ["x 1.0"]


def test_an_app_with_no_systems_and_no_gui(app_dir, tmp_path):
    out = tmp_path / "out"
    assert _export(app_dir, out, BARE) == 0
    assert json.loads((out / "report.json").read_text())["roots"] == []
    assert _node(out) == {"roots": [], "state": {}, "gui": []}


def test_an_async_startup_handler_runs_before_the_snapshot(app_dir, tmp_path):
    out = tmp_path / "out"
    assert _export(app_dir, out, ASYNC) == 0
    assert _node(out)["state"] == {"loaded": "yes"}


def test_state_that_cannot_cross_is_left_out_and_reported(app_dir, tmp_path):
    out = tmp_path / "out"
    assert _export(app_dir, out, LEFT_OUT) == 0
    assert _node(out)["state"] == {"kept": [1, 2.5, "three"]}
    left = json.loads((out / "report.json").read_text())["state_left_out"]
    assert [x["path"] for x in left] == ["engine.state[('terrain', 7)]", "engine.state['grid']"]


def test_a_transpile_error_exits_1_names_the_line_and_writes_nothing(app_dir, tmp_path, capsys):
    out = tmp_path / "out"
    assert _export(app_dir, out, BAD, name="bad.py") == 1
    line = BAD.splitlines().index("    try:") + 1
    assert f"bad.py:{line}: try/except is outside the subset" in capsys.readouterr().err
    assert not out.exists()


def test_the_module_form_names_the_module_file(app_dir, tmp_path, capsys, monkeypatch):
    pkg = app_dir / "webx_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(BAD)
    monkeypatch.chdir(app_dir)
    out = tmp_path / "out"
    assert main(["export", "--web", "-m", "webx_pkg.main", str(out)]) == 1
    line = BAD.splitlines().index("    try:") + 1
    assert f"webx_pkg/main.py:{line}:" in capsys.readouterr().err
    assert not out.exists()


def test_the_capture_restores_engine_run_and_main(app_dir, tmp_path):
    before_run, before_main = mx.Engine.run, sys.modules["__main__"]
    _export(app_dir, tmp_path / "out", APP, webx_rig=RIG)
    assert mx.Engine.run is before_run
    assert sys.modules["__main__"] is before_main


def test_the_scene_report_does_not_call_transpiled_code_dropped(app_dir, tmp_path, capsys):
    assert _export(app_dir, tmp_path / "out", APP, webx_rig=RIG) == 0
    err = capsys.readouterr().err
    assert "custom Python that runs every frame" not in err
    assert "ValueDisplay.getter = label" not in err  # transpiled too
