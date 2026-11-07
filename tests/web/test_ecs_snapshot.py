"""The entity store crosses to the page: ecs.bin holds the rows and membership below the
high-water mark, state.json's "ecs" its layouts, a geometry crosses by its registry id, and
the engine dict carries the origin."""

import json

import numpy as np
import pytest

from manifoldx.web.export import build_export, capture_engine, export_web, write_version

APP = """
import numpy as np
import manifoldx as mx
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.resources import PhongMaterial, cube
from manifoldx.types import Vector3


class Velocity(Component):
    v: Vector3


engine = mx.Engine("ecs", width=64, height=64)
CUBE = cube(1.0, 1.0, 1.0)
CRATE = cube(2.0, 2.0, 2.0)
GREY = PhongMaterial(color="#888888")
engine.spawn(Mesh(CUBE), Material(GREY), Transform(pos=(0.0, -1.0, 0.0), scale=(20.0, 0.1, 20.0)))
first = engine.spawn(Mesh(CUBE), Material(GREY), Transform(pos=(0.0, 1.0, 0.0)), Velocity(v=(0.0, 1.0, 0.0)), n=3)
engine.store.destroy(np.array([first.index + 1]))


def tick(query, dt: float):
    crate = CRATE


engine.system(tick)
engine.run()
"""


@pytest.fixture
def exported(tmp_path):
    app = tmp_path / "app.py"
    app.write_text(APP)
    engine, main = capture_engine(None, str(app), [])
    built = build_export(engine, main, str(tmp_path), app_label="app.py")
    out = tmp_path / "out"
    write_version(built, out)
    return out, engine, main


def test_ecs_bin_holds_the_store_below_the_high_water_mark(exported):
    out, engine, _ = exported
    header = json.loads((out / "state.json").read_text())["ecs"]
    data = (out / "ecs.bin").read_bytes()
    store = engine.store
    hw = store._high_water
    assert header["high_water"] == hw == 4
    assert header["free"] == store._free_list
    off, n = header["alive"]
    assert np.frombuffer(data, np.uint8, n, off).astype(bool).tolist() == store._alive[:hw].tolist()
    by = {c["name"]: c for c in header["components"]}
    assert set(by) == {"Transform", "Mesh", "Material", "Velocity"}
    t = by["Transform"]
    assert (t["dtype"], t["cols"]) == ("f4", 10)
    assert t["fields"] == {
        "pos": [0, 3],
        "position": [0, 3],
        "rot": [3, 4],
        "rotation": [3, 4],
        "scale": [7, 3],
    }
    rows = np.frombuffer(data, np.float32, hw * 10, t["rows"][0]).reshape(hw, 10)
    assert np.array_equal(rows, store._components["Transform"][:hw])
    assert by["Mesh"]["dtype"] == "u4" and by["Mesh"]["fields"] == {"geometry_id": [0, 1]}
    assert by["Velocity"]["fields"] == {"v": [0, 3]}
    m = by["Velocity"]["members"]
    members = np.frombuffer(data, np.uint8, m[1], m[0]).astype(bool).tolist()
    assert members == store._members["Velocity"][:hw].tolist()
    assert all(c["rows"][0] % 4 == 0 and c["members"][0] % 4 == 0 for c in header["components"])


def test_a_geometry_reached_code_reads_crosses_by_id(exported):
    out, engine, main = exported
    state = json.loads((out / "state.json").read_text())
    crate = state["globals"]["__main__"]["CRATE"]
    assert crate == {"__geometry": engine._geometry_registry.register(main.CRATE)}


def test_the_origin_crosses(exported):
    state = json.loads((exported[0] / "state.json").read_text())
    assert state["engine"]["origin"] == [0.0, 0.0]


FLAGS = """
import numpy as np
import manifoldx as mx

engine = mx.Engine("flags", width=64, height=64)
engine.store.register_component("Flags", np.dtype("i8"), (1,))
engine.run()
"""


def test_a_component_of_another_dtype_fails_the_export(tmp_path, capsys):
    app = tmp_path / "app.py"
    app.write_text(FLAGS)
    assert export_web(None, str(app), tmp_path / "out", []) == 1
    assert not (tmp_path / "out").exists()
    err = capsys.readouterr().err
    assert "the component Flags: dtype int64; ecs.bin carries float32 and uint32" in err


def test_the_page_resolves_a_geometry_to_its_id(tmp_path):
    from manifoldx.web.replay import replay_node

    app = tmp_path / "app.py"
    app.write_text(APP)
    out = tmp_path / "out"
    assert export_web(None, str(app), out, []) == 0
    crate = json.loads((out / "state.json").read_text())["globals"]["__main__"]["CRATE"]
    frame = {"keys": [], "just": [], "d": [0.0, 0.0], "buttons": [], "clicks": [], "vp": [64, 64]}
    trace = replay_node(out, [frame], ["CRATE"])
    assert trace[0]["paths"]["CRATE"] == {"geometry_id": crate["__geometry"]}


def test_every_block_starts_on_a_multiple_of_four_when_the_store_is_odd():
    import manifoldx as mx
    from manifoldx.components import Transform
    from manifoldx.web.snapshot import ecs_bytes

    engine = mx.Engine("odd", width=64, height=64)
    engine.spawn(Transform(pos=(1.0, 2.0, 3.0)), n=3)
    header, data = ecs_bytes(engine)
    assert header["high_water"] == 3
    assert all(c["rows"][0] % 4 == 0 and c["members"][0] % 4 == 0 for c in header["components"])
    t = next(c for c in header["components"] if c["name"] == "Transform")
    rows = np.frombuffer(data, np.float32, 30, t["rows"][0]).reshape(3, 10)
    assert np.array_equal(rows, engine.store._components["Transform"][:3])
