"""Under a fixed timestep the page's clock is 0 on the first frame and adds each later
frame's dt; the Python replay, which mirrors Engine._draw_frame, must agree frame for frame."""

import pytest

from manifoldx.web.export import export_web
from manifoldx.web.replay import DT, compare, replay_node, replay_python

APP = """
import manifoldx as mx

engine = mx.Engine("fixed step", width=64, height=64)
engine.set_fixed_timestep({dt!r})
engine.state["ticks"] = []
engine.state["frames"] = []


def tick(q, dt: float):
    ticks: list[float] = engine.state["ticks"]
    ticks.append(engine.elapsed)
    engine.state["ticks"] = ticks


def on_frame(p: dict):
    frames: list[float] = engine.state["frames"]
    frames.append(p["elapsed"])
    engine.state["frames"] = frames


engine.system(tick)
engine.on("frame")(on_frame)
engine.run()
"""

IDLE = {"keys": [], "just": [], "d": [0.0, 0.0], "buttons": [], "clicks": [], "vp": [64, 64]}
FRAMES = [IDLE] * 6


def test_elapsed_under_a_fixed_step(tmp_path):
    app = tmp_path / "fixed_step.py"
    app.write_text(APP.format(dt=DT))
    out = tmp_path / "out"
    assert export_web(None, str(app), out, []) == 0
    python = replay_python(None, str(app), FRAMES, [])
    node = replay_node(out, FRAMES, [])

    assert compare(python, node, rel=0) is None
    expected = [i * DT for i in range(len(FRAMES))]
    # Systems and the frame handler see the clock before the frame's step.
    assert [e["state"]["ticks"][-1] for e in python] == pytest.approx(expected)
    assert [e["state"]["frames"][-1] for e in node] == pytest.approx(expected)
