"""The clock: time.time() in frame code reads the frame's "clock" (seconds since the epoch,
Date.now() / 1000 in the page, 0.0 when the frame has none), so both replays see the time the
page saw. Other time functions fail the export with a hint."""

import json
import subprocess
from pathlib import Path

import pytest

from manifoldx.web.export import export_web
from manifoldx.web.replay import compare, replay_node, replay_python

APP = """
import time

import manifoldx as mx

engine = mx.Engine("clock", width=64, height=64)
engine.state["t"] = -1.0
engine.state["at_frame"] = -1.0
engine.state["setup"] = time.time() > 1e9


def on_frame(p: dict):
    engine.state["at_frame"] = time.time()


def tick(q, dt: float):
    engine.state["t"] = time.time()


engine.on("frame")(on_frame)
engine.system(tick)
engine.run()
"""

IDLE = {"keys": [], "just": [], "d": [0.0, 0.0], "buttons": [], "clicks": [], "vp": [64, 64]}
CLOCKS = [1762500000.25 + i / 60 for i in range(10)]
INPUT_JS = Path(__file__).parents[2] / "src" / "manifoldx" / "web" / "js" / "input.js"


@pytest.fixture
def app(tmp_path):
    path = tmp_path / "clock_app.py"
    path.write_text(APP)
    return path


def _both(app, tmp_path, frames):
    out = tmp_path / "out"
    assert export_web(None, str(app), out, []) == 0
    return replay_python(None, str(app), frames, []), replay_node(out, frames, [])


def test_time_time_is_the_frames_clock(app, tmp_path):
    frames = [{**IDLE, "clock": c} for c in CLOCKS]
    python, node = _both(app, tmp_path, frames)
    assert compare(python, node, rel=0) is None
    assert [e["state"]["t"] for e in python] == CLOCKS
    assert [e["state"]["at_frame"] for e in node] == CLOCKS  # the handler's equals the system's
    assert python[0]["state"]["setup"] is True  # setup reads the real time


def test_no_clock_reads_zero(app, tmp_path):
    python, node = _both(app, tmp_path, [IDLE] * 3)
    assert compare(python, node) is None
    assert [e["state"]["t"] for e in python] == [0.0] * 3
    assert [e["state"]["t"] for e in node] == [0.0] * 3


@pytest.mark.parametrize(
    "source, shown",
    [
        ("import time\n", "time.perf_counter()"),
        ("from time import time\n", "from time import time"),
    ],
)
def test_other_time_functions_fail_the_export(tmp_path, capsys, source, shown):
    call = "time.perf_counter()" if source == "import time\n" else "time()"
    path = tmp_path / "bad.py"
    path.write_text(
        source + "import manifoldx as mx\n"
        "engine = mx.Engine('bad', width=64, height=64)\n"
        "engine.state['t'] = 0.0\n"
        "\n"
        "def tick(q, dt: float):\n"
        f"    engine.state['t'] = {call}\n"
        "\n"
        "engine.system(tick)\n"
        "engine.run()\n"
    )
    assert export_web(None, str(path), tmp_path / "out", []) == 1
    err = capsys.readouterr().err
    assert f"bad.py:7: `{shown}` (frame code reads" in err
    assert (
        "(frame code reads time.time(), the frame's clock, or engine.elapsed) "
        "is outside the subset (reached from system tick)"
    ) in err


def test_the_page_records_a_clock_that_does_not_go_back(tmp_path):
    script = tmp_path / "clock.mjs"
    script.write_text(
        f"import {{ FrameBuilder }} from {json.dumps(INPUT_JS.as_uri())};\n"
        "const now = [1000, 1016, 1016, 1010];  // the wall clock stepped back\n"
        "Date.now = () => now.shift();\n"
        "const b = new FrameBuilder();\n"
        "console.log(JSON.stringify([0, 1, 2, 3].map(() => b.take([8, 8]).clock)));\n"
    )
    done = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [1, 1.016, 1.016, 1.016]
