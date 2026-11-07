"""Spawning and destroying from frame code: engine.spawn builds rows as each component's
get_data does and fills slots at once, as the desktop picks them; engine.destroy and a handle's
destroy queue for the flush; Mesh() takes a geometry the export saw, which the library glb
holds even when no entity drew it at setup."""

import json
import sys
from pathlib import Path

import pytest

from manifoldx.web.export import export_web
from manifoldx.web.replay import compare, replay_node, replay_python

sys.path.insert(0, str(Path(__file__).parent.parent))
from gltf_helpers import load  # noqa: E402

SPAWNER = Path(__file__).parent / "apps" / "spawner.py"
IDLE = {"keys": [], "just": [], "d": [0.0, 0.0], "buttons": [], "clicks": [], "vp": [320, 200]}
PRESS = {2: "Space", 3: "Space", 10: "b", 40: "x", 130: "Space"}
FRAMES = [
    {**IDLE, "keys": [PRESS[i]], "just": [PRESS[i]]} if i in PRESS else IDLE for i in range(150)
]


@pytest.fixture(scope="module")
def spawner_export(tmp_path_factory):
    out = tmp_path_factory.mktemp("spawner") / "out"
    assert export_web(None, str(SPAWNER), out, []) == 0
    return out


@pytest.fixture(scope="module")
def spawner_python():
    return replay_python(None, str(SPAWNER), FRAMES, [])


def test_spawner_replays(spawner_export, spawner_python):
    assert compare(spawner_python, replay_node(spawner_export, FRAMES, [])) is None
    alive = [len(e["store"]["alive"]) for e in spawner_python]
    assert alive[1] == 1 and alive[3] == 41 and alive[10] == 42 and alive[40] == 41
    first, second = (
        set(spawner_python[2]["store"]["alive"]),
        set(spawner_python[3]["store"]["alive"]),
    )
    assert len(second - first) == 20
    dead = sorted(spawner_python[129]["store"]["free"])
    assert len(dead) >= 40 and len(spawner_python[130]["store"]["alive"]) == 21
    # the third wave fills freed slots, in the order the desktop pops its free list
    assert set(spawner_python[130]["store"]["alive"]) - set(
        spawner_python[129]["store"]["alive"]
    ) <= set(dead)
    assert spawner_python[130]["store"]["high_water"] == spawner_python[129]["store"]["high_water"]
    lives = [e["state"]["lives"] for e in spawner_python]
    gone = next(i for i in range(4, 150) if len(spawner_python[i]["store"]["alive"]) < 41)
    assert lives[gone] == 40 and lives[gone + 1] == 20  # age's destroy lands at the flush


def test_a_spawned_geometry_no_entity_had_is_in_the_library(spawner_export, spawner_python):
    state = json.loads((spawner_export / "state.json").read_text())
    ball = state["globals"]["__main__"]["BALL"]["__geometry"]
    g = load(spawner_export / "scene.glb")
    assert str(ball) in g.scenes[0].extras["manifoldx"]["mx_geometries"]
    meshes = spawner_python[10]["store"]["rows"]["Mesh"]
    assert [ball] in meshes and [ball] not in spawner_python[9]["store"]["rows"]["Mesh"]
    tags = spawner_python[10]["store"]["rows"]["Tag"]
    assert [3.0] in tags  # Tag(), registered by this spawn, takes its class default
    assert "Tag" not in spawner_python[9]["store"]["rows"]
    wave = [row for row in spawner_python[2]["store"]["rows"]["Life"] if row[0] > 1.0]
    assert len(wave) == 20 and all(row[1] == 7.0 for row in wave)  # born, left out of Life(t=2.0)


def test_spawn_of_zero_returns_none(spawner_export, spawner_python):
    node = replay_node(spawner_export, FRAMES[:2], [])
    assert spawner_python[0]["state"]["none"] is None and node[0]["state"]["none"] is None
    assert "none" in node[0]["state"]


def _export_error(tmp_path, source, capsys) -> str:
    app = tmp_path / "app.py"
    app.write_text(source)
    assert export_web(None, str(app), tmp_path / "out", []) == 1
    return capsys.readouterr().err


BUILD = """
import manifoldx as mx
from manifoldx.components import Mesh, Transform
from manifoldx.resources import cube

engine = mx.Engine("build", width=64, height=64)


def tick(query, dt: float):
    engine.spawn(Mesh(cube(0.5, 0.5, 0.5)), Transform())


engine.system(tick)
engine.run()
"""


def test_building_a_mesh_in_frame_code_fails_the_export(tmp_path, capsys):
    err = _export_error(tmp_path, BUILD, capsys)
    assert "app.py:10:" in err and "a geometry built in frame code" in err
    assert "build it at setup" in err


def test_mesh_of_something_not_a_geometry_fails_the_export(tmp_path, capsys):
    source = BUILD.replace("Mesh(cube(0.5, 0.5, 0.5))", 'Mesh(engine.state["x"])')
    source = source.replace(
        'engine = mx.Engine("build", width=64, height=64)',
        'engine = mx.Engine("build", width=64, height=64)\nengine.state["x"] = 1.0',
    )
    err = _export_error(tmp_path, source, capsys)
    assert "Mesh() of a value that is not a geometry the export saw" in err
