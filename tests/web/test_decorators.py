"""Handlers and systems registered with the decorator forms, @engine.on("name") and
@engine.system, export as the call forms do (#157): the decorator ran at setup and returned
the function, so the browser gets the plain function. Any other decorator still fails."""

import json
import subprocess
import sys

import pytest

from manifoldx.web.export import export_web

APP = """
import manifoldx as mx

engine = mx.Engine("decorated", width=64, height=64)
engine.state.yaw = 0.0


@engine.system
def turn(query, dt: float):
    yaw: float = engine.state.yaw
    engine.state.yaw = yaw + 1.0


@engine.on("bump")
def bump(payload: dict):
    \"\"\"Turn the camera: {"by": degrees}.\"\"\"
    by: float = payload["by"]
    yaw: float = engine.state.yaw
    engine.state.yaw = yaw + by


if __name__ == "__main__":
    engine.run()
"""

RUN = """
import { load } from %s;
import { advance } from %s;
const app = await load(new URL(%s));
const idle = { keys: [], just: [], d: [0, 0], buttons: [], clicks: [], vp: [64, 64] };
advance(app, { ...idle, events: [["bump", { by: 10 }]] }, 1 / 60);
console.log(JSON.stringify({ roots: app.roots.map((r) => r.label), yaw: app.engine.state.get("yaw") }));
"""


@pytest.fixture
def app(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    yield folder
    for name, m in list(sys.modules.items()):
        if str(getattr(m, "__file__", None) or "").startswith(str(folder)):
            del sys.modules[name]


def test_decorated_system_and_handler_export_and_run(app, tmp_path):
    script, out = app / "app.py", tmp_path / "out"
    script.write_text(APP)
    assert export_web(None, str(script), out, []) == 0
    runner = tmp_path / "run.mjs"
    uri = lambda p: json.dumps(p.as_uri())  # noqa: E731
    runner.write_text(
        RUN % (uri(out / "replay.mjs"), uri(out / "manifoldx.js"), json.dumps(out.as_uri() + "/"))
    )
    done = subprocess.run(["node", str(runner)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {"roots": ["system turn", 'on("bump") bump'], "yaw": 11.0}


OTHER = """
import manifoldx as mx

engine = mx.Engine("decorated", width=64, height=64)


def twice(f):
    return f


@twice
def helper(x: float) -> float:
    return x * 2.0


def tick(query, dt: float):
    engine.state["x"] = helper(dt)


engine.system(tick)

if __name__ == "__main__":
    engine.run()
"""


BUS = """
import manifoldx as mx

engine = mx.Engine("decorated", width=64, height=64)


class Bus:
    def on(self, name):
        return lambda f: f


bus = Bus()


@bus.on("x")
def helper(x: float) -> float:
    return x * 2.0


def tick(query, dt: float):
    engine.state["x"] = helper(dt)


engine.system(tick)

if __name__ == "__main__":
    engine.run()
"""


@pytest.mark.parametrize(
    "source, decorator",
    [(OTHER, "@twice"), (BUS, "@bus.on('x')")],
    ids=["a function", "an on() that is not the engine's"],
)
def test_any_other_decorator_still_fails(app, tmp_path, capsys, source, decorator):
    script = app / "app.py"
    script.write_text(source)
    assert export_web(None, str(script), tmp_path / "out", []) == 1
    assert f"the decorator {decorator} is outside the subset" in capsys.readouterr().err
