"""engine.state is a hot.State on the desktop: a dict whose keys also read and write as
attributes. Reached code that uses either form, and its dict methods, must run in Node as it
does in Python; a form the browser cannot run fails the export with its file and line."""

import json
import subprocess
import sys

import pytest

from manifoldx.cli import main
from manifoldx.web.export import export_web
from manifoldx.web.replay import compare, replay_node, replay_python

APP = """
import manifoldx as mx
from manifoldx.gui import Panel, ValueDisplay

engine = mx.Engine("state", width=64, height=64)
engine.state.setdefault("n", 0)
engine.state.cam = {"yaw": 0.0, "dist": 12.0}


def tick(query, dt):
    n: int = engine.state.n
    engine.state.n = n + 1
    hits: int = engine.state.setdefault("hits", 10)
    engine.state.hits = hits + 2
    if "flag" in engine.state:
        engine.state.flag = not engine.state.flag
    else:
        engine.state.flag = True
    if "never" not in engine.state:
        engine.state["seen"] = engine.state.get("seen", 0.5)
    best: int = engine.state.get("best", 3)
    engine.state.best = best + n
    yaw: float = engine.state.cam["yaw"]
    engine.state.cam["yaw"] = yaw + 0.25
    cam = engine.state.cam
    cam["dist"] = cam["dist"] - 1.0


def label():
    n: int = engine.state.n
    return f"n {n}"


engine.system(tick)
engine.gui.append(Panel(children=[ValueDisplay(label)]))

if __name__ == "__main__":
    engine.run()
"""

FRAMES = [
    {"keys": [], "just": [], "d": [0.0, 0.0], "buttons": [], "clicks": [], "vp": [64, 64]}
] * 3
PATHS = ["n", "cam.yaw", "cam.dist", "hits", "flag", "best"]


@pytest.fixture
def app(tmp_path):
    folder = tmp_path / "src"
    folder.mkdir()
    yield folder
    for name, m in list(sys.modules.items()):
        if str(getattr(m, "__file__", None) or "").startswith(str(folder)):
            del sys.modules[name]


def test_attribute_access_and_dict_methods_replay_as_in_python(app, tmp_path):
    script, out = app / "app.py", tmp_path / "out"
    script.write_text(APP)
    assert export_web(None, str(script), out, []) == 0
    python = replay_python(None, str(script), FRAMES, PATHS)
    mismatch = compare(python, replay_node(out, FRAMES, PATHS))
    assert mismatch is None, str(mismatch)
    # The script exercised both branches and every key, so agreement means something.
    assert [f["paths"]["n"] for f in python] == [1, 2, 3]
    assert [f["paths"]["flag"] for f in python] == [True, False, True]
    assert [f["paths"]["hits"] for f in python] == [12, 14, 16]
    assert [f["paths"]["cam.dist"] for f in python] == [11.0, 10.0, 9.0]


UNSUPPORTED = [
    (
        "    for engine.state.i in range(3):\n        pass",
        "`engine.state.i` as a loop or unpacking target",
    ),
    ("    engine.state.a, engine.state.b = 1, 2", "unpacking into `engine.state.a`"),
    ("    f = engine.state.keys", "`engine.state.keys`, a dict method used as a value,"),
]


@pytest.mark.parametrize(("body", "error"), UNSUPPORTED)
def test_a_form_the_browser_cannot_run_fails_the_export(app, tmp_path, capsys, body, error):
    src = (
        "import manifoldx as mx\n\n"
        'engine = mx.Engine("bad", width=64, height=64)\n\n\n'
        f"def tick(query, dt):\n{body}\n\n\n"
        "engine.system(tick)\nengine.run()\n"
    )
    (app / "bad.py").write_text(src)
    assert main(["export", "--web", str(app / "bad.py"), str(tmp_path / "out")]) == 1
    assert f"bad.py:7: {error}" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


LOG_APP = """
import manifoldx as mx

engine = mx.Engine("log", width=64, height=64)
engine.state.yaw = 10.0


def tick(query, dt):
    yaw: float = engine.state.yaw
    engine.state.yaw = yaw + 2.5
    engine.log(f"yaw {yaw:.1f}")


engine.system(tick)

if __name__ == "__main__":
    engine.run()
"""

LINES = """
import { load, replay } from %s;
const app = await load(new URL(%s));
replay(app, JSON.parse(process.argv[2]), []);
console.log(JSON.stringify(app.engine.lines));
"""


def test_engine_log_transpiles_and_collects_lines_in_node(app, tmp_path):
    script, out = app / "app.py", tmp_path / "out"
    script.write_text(LOG_APP)
    assert export_web(None, str(script), out, []) == 0
    runner = tmp_path / "lines.mjs"
    runner.write_text(
        LINES % (json.dumps((out / "replay.mjs").as_uri()), json.dumps(out.as_uri() + "/"))
    )
    done = subprocess.run(
        ["node", str(runner), json.dumps(FRAMES)], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == ["yaw 10.0", "yaw 12.5", "yaw 15.0"]
