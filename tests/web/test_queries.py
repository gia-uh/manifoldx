"""Component queries in systems: a system gets its query's entities, reads their fields as the
desktop's field views do, and queues its writes for the flush after the systems, so the page
and the desktop agree frame by frame, the store included."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from manifoldx.web.export import capture_engine, export_web
from manifoldx.web.replay import compare, replay_node, replay_python

DRIFT = Path(__file__).parent / "apps" / "drift.py"
IDLE = {"keys": [], "just": [], "d": [0.0, 0.0], "buttons": [], "clicks": [], "vp": [320, 200]}
FRAMES = [IDLE] * 90


@pytest.fixture(scope="module")
def drift_export(tmp_path_factory):
    out = tmp_path_factory.mktemp("drift") / "out"
    assert export_web(None, str(DRIFT), out, []) == 0
    return out


@pytest.fixture(scope="module")
def drift_python():
    return replay_python(None, str(DRIFT), FRAMES, [])


def test_drift_replays(drift_export, drift_python):
    node = replay_node(drift_export, FRAMES, [])
    assert compare(drift_python, node) is None
    first, last = drift_python[0]["store"]["rows"], drift_python[-1]["store"]["rows"]
    assert first["Transform"] != last["Transform"]  # the boxes moved
    assert drift_python[-1]["state"]["low"] > 0  # and fell below the ground's top


def test_a_system_reads_the_rows_before_the_flush(drift_export, drift_python):
    engine, _ = capture_engine(None, str(DRIFT), [])
    store = engine.store
    boxes = np.flatnonzero(store._members["Velocity"][: store._high_water])
    exported = float(np.sum(store._components["Transform"][boxes, 0:3]))
    node = replay_node(drift_export, FRAMES[:2], [])
    assert drift_python[0]["state"]["sum"] == node[0]["state"]["sum"] == exported
    assert drift_python[1]["state"]["sum"] != exported


def test_query_writes_applied_at_once_are_a_mismatch(drift_export, drift_python, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(drift_export, out)
    js = out / "manifoldx.js"
    text = js.read_text()
    queue = "this.store.queueUpdate(this.component, this.indices, this.c0, this.c1, data);"
    assert text.count(queue) == 1
    js.write_text(text.replace(queue, queue + " this.store.flush();"))
    found = compare(drift_python, replay_node(out, FRAMES, []))
    assert found is not None and found.frame == 0


def test_ground_is_not_in_the_query(drift_export, drift_python):
    node = replay_node(drift_export, FRAMES[:3], [])
    assert drift_python[0]["state"]["n"] == node[0]["state"]["n"] == 12


HANDLER = """
import numpy as np
import manifoldx as mx
from manifoldx.components import Component, Transform
from manifoldx.types import Vector3


class Velocity(Component):
    v: Vector3


engine = mx.Engine("handler", width=64, height=64)
engine.spawn(Transform(), Velocity(v=(1.0, 0.0, 0.0)))


def bump(payload: dict, q: mx.Query[Velocity]):
    q[Velocity].v += np.array([0.0, 1.0, 0.0], np.float32)


engine.on("bump")(bump)
engine.run()
"""

WHOLE = """
import numpy as np
import manifoldx as mx
from manifoldx.components import Component, Transform
from manifoldx.types import Vector3


class Velocity(Component):
    v: Vector3


engine = mx.Engine("whole", width=64, height=64)
engine.spawn(Transform(), Velocity(v=(1.0, 0.0, 0.0)))


def push(q: mx.Query[Velocity], dt: float):
    q[Velocity] += np.array([0.0, 1.0, 0.0], np.float32)


engine.system(push)
engine.run()
"""


def _export_error(tmp_path, source, capsys) -> str:
    app = tmp_path / "app.py"
    app.write_text(source)
    assert export_web(None, str(app), tmp_path / "out", []) == 1
    return capsys.readouterr().err


def test_a_handlers_query_and_accessor_writes_fail_with_a_hint(tmp_path, capsys):
    err = _export_error(tmp_path, HANDLER, capsys)
    assert "app.py:17: `q[Velocity].v += ...`, a write through an event handler's query" in err
    assert "which is read-only" in err
    err = _export_error(tmp_path, WHOLE, capsys)
    assert (
        "app.py:17: `q[Velocity] += ...`, a write to a whole component "
        "(write a field: q[Velocity].v += ...)"
    ) in err


def test_a_field_in_a_numpy_function_names_data(tmp_path, capsys):
    source = WHOLE.replace(
        "q[Velocity] += np.array([0.0, 1.0, 0.0], np.float32)",
        'engine.state["m"] = float(np.sum(q[Velocity].v, axis=None))',
    )
    err = _export_error(tmp_path, source, capsys)
    assert "app.py:17:" in err and "q[Velocity].v.data" in err
