"""build_export builds a version of the app in memory; write_static and write_version write it.
export_web is the two composed, so what it writes is what build_export built."""

import json
from pathlib import Path

import pytest

from manifoldx.web.export import build_export, capture_engine, export_web, write_version

WEB_WALK = Path(__file__).parents[2] / "examples" / "web_walk.py"

CROSSED = """
import numpy as np
import manifoldx as mx

engine = mx.Engine("crossed", width=64, height=64)
engine.state["yaw"] = 0.5
engine.state[("tex", 1)] = np.zeros((2, 2))
engine.run()
"""


def _build(script: Path, **kw):
    engine, main = capture_engine(None, str(script), [])
    return build_export(engine, main, str(script.parent), app_label=script.name, **kw)


@pytest.fixture(scope="module")
def walk_built():
    return _build(WEB_WALK, runtime="../../pyrt.js")


def test_build_is_what_export_writes(tmp_path):
    engine, main = capture_engine(None, str(WEB_WALK), [])
    built = build_export(engine, main, str(WEB_WALK.parent), app_label="web_walk.py")
    assert export_web(None, str(WEB_WALK), tmp_path / "out", []) == 0
    assert json.loads((tmp_path / "out" / "state.json").read_text()) == built.state
    assert (tmp_path / "out" / "scene.glb").read_bytes() == built.glb
    assert set(built.timings) == {"startup", "reach", "snapshot", "transpile", "gui", "scene"}


def test_scene_sha_follows_the_scene(tmp_path):
    # the same app twice: same sha; a colour changed in a copy: another sha
    first, second = _build(WEB_WALK), _build(WEB_WALK)
    copy = tmp_path / "web_walk.py"
    copy.write_text(WEB_WALK.read_text().replace('"#c0392b"', '"#00ff00"'))
    recoloured = _build(copy)
    assert len(first.scene_sha) == 16
    assert first.scene_sha == second.scene_sha != recoloured.scene_sha


def test_versions_share_the_runtime(walk_built):
    assert 'from "../../pyrt.js"' in walk_built.modules["__main__"][0]
    assert json.loads(walk_built.modules["__main__"][1])["sources"] == ["web_walk.py"]


def test_write_version_keeps_an_existing_scene(tmp_path, walk_built):
    (tmp_path / "scenes").mkdir()
    (tmp_path / "scenes" / "x.glb").write_bytes(b"old")
    write_version(walk_built, tmp_path / "v1", scene_path="../scenes/x.glb")
    assert (tmp_path / "scenes" / "x.glb").read_bytes() == b"old"
    assert {p.name for p in (tmp_path / "v1").iterdir()} == {"app", "state.json", "report.json"}


def test_crossed_keys_leave_out_what_cannot_cross(tmp_path):
    # an app whose engine.state holds {"yaw": 0.5, ("tex", 1): array2d}: crossed == {"yaw"}
    app = tmp_path / "crossed.py"
    app.write_text(CROSSED)
    built = _build(app)
    assert built.crossed == frozenset({"yaw"})
    assert [e.path_from_root for e in built.left_out] == ["engine.state[('tex', 1)]"]
