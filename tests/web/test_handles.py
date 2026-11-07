"""Entity handles in frame code: a handle's transform reads and writes its entity's row at
once, with the floating origin added and removed as _TransformProxy does; set_material and
destroy reach the store; engine.camera.up is writable (#153); a material's fields are not, and
the error says to swap materials with engine.set_material (#154)."""

from pathlib import Path

import pytest

from manifoldx.web.export import export_web
from manifoldx.web.replay import compare, replay_node, replay_python

HANDLES = Path(__file__).parent / "apps" / "handles.py"
IDLE = {"keys": [], "just": [], "d": [0.0, 0.0], "buttons": [], "clicks": [], "vp": [320, 200]}
FRAMES = [IDLE] * 80
PATHS = ["engine.camera.up", "clock.n"]


def _replay(tmp_path, args):
    out = tmp_path / "out"
    assert export_web(None, str(HANDLES), out, args) == 0
    python = replay_python(None, str(HANDLES), FRAMES, PATHS, app_args=args)
    return python, replay_node(out, FRAMES, PATHS)


def test_handles_replay(tmp_path):
    python, node = _replay(tmp_path, [])
    assert compare(python, node) is None
    assert 3 in python[48]["store"]["alive"] and 3 not in python[49]["store"]["alive"]
    assert [e["state"]["seen"] for e in python[48:51]] == [8, 8, 7]
    materials = {row[0] for e in python for row in e["store"]["rows"]["Material"]}
    assert len(materials) == 3  # grey, then red and blue on the timer
    assert python[10]["paths"]["engine.camera.up"] != python[20]["paths"]["engine.camera.up"]


def test_handles_replay_far_from_the_origin(tmp_path):
    python, node = _replay(tmp_path, ["--far"])
    assert compare(python, node) is None
    first = node[30]["state"]["first"]
    assert 4990.0 < first[0] < 5010.0 and -3010.0 < first[2] < -2990.0
    assert first[0] != round(first[0], 3)  # float64 world coordinates, not float32 rows
    row = python[30]["store"]["rows"]["Transform"][0]
    assert abs(row[0]) < 10.0  # the row is relative to the origin


def test_a_handle_write_is_seen_by_the_next_system_in_the_same_frame(tmp_path):
    python, node = _replay(tmp_path, [])
    for trace in (python, node):
        row = trace[5]["store"]["rows"]["Transform"][0]
        assert trace[5]["state"]["first"] == pytest.approx(row[0:3], abs=0.0)


def test_set_material_on_a_dead_entity_raises_as_the_desktop(tmp_path):
    out = tmp_path / "out"
    assert export_web(None, str(HANDLES), out, ["--dead"]) == 0
    with pytest.raises(ValueError, match="set_material: entity 3 is not alive"):
        replay_python(None, str(HANDLES), FRAMES, PATHS, app_args=["--dead"])
    with pytest.raises(RuntimeError, match="set_material: entity 3 is not alive"):
        replay_node(out, FRAMES, PATHS)


PAINT = """
import manifoldx as mx
from manifoldx.resources import PhongMaterial

engine = mx.Engine("paint", width=64, height=64)
RED = PhongMaterial(color="#ff0000")


def tick(query, dt: float):
    RED.color = "#00ff00"


engine.system(tick)
engine.run()
"""


def test_writing_a_material_field_points_at_set_material(tmp_path, capsys):
    app = tmp_path / "app.py"
    app.write_text(PAINT)
    assert export_web(None, str(app), tmp_path / "out", []) == 1
    err = capsys.readouterr().err
    assert (
        "app.py:10: writing `RED.color`, a field of a material the page does not apply "
        "(swap materials made at setup with engine.set_material)"
    ) in err
