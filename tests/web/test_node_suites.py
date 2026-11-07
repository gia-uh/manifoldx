"""The runtime's own Node tests (tests/web/js/*.test.mjs), run from pytest so make test
covers them, and the numpy values they compare against."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from manifoldx.camera import Camera
from manifoldx.web.export import export_web

from gltf_helpers import posed_positions, skinned_figure

HERE = Path(__file__).parent / "js"
FORWARD = HERE / "forward.json"
WEB_WALK = Path(__file__).parents[2] / "examples" / "web_walk.py"
SPAWNER = Path(__file__).parent / "apps" / "spawner.py"
SKINNED = Path(__file__).parent / "apps" / "skinned.py"

# A two-module app whose system calls a helper that reads an attribute of None, for
# sourcemap.test.mjs: the error must name the helper's line, then the system's call.
HELPER = """
class Spot:
    def __init__(self):
        self.x = 1.0


def far(spot: Spot | None) -> float:
    return spot.x * 2.0
"""

FAILING = """
import manifoldx as mx
from mxsm_helper import Spot, far


class Holder:
    def __init__(self):
        self.spot: Spot | None = Spot()
        self.n = 0.0


holder = Holder()
engine = mx.Engine("sourcemap", width=64, height=64)


def tick(query, dt: float):
    holder.spot = None
    holder.n = holder.n + far(holder.spot)


engine.system(tick)
engine.run()
"""

# Position/target pairs whose float32 arithmetic differs from float64's.
PAIRS = [
    ((0.1, 1.7, 3.3), (-2.25, 0.3, 7.1)),
    ((10.0, 1.6, -4.2), (10.3, 1.55, -5.0)),
    ((-123.456, 7.89, 0.001), (0.0, 0.0, 0.0)),
    ((0.0, 2.0, 0.0), (1e-3, 2.0, 1.0)),
    ((3.14159, -2.71828, 1.41421), (1.73205, 2.23607, -0.57721)),
]


def _forward() -> list[dict]:
    out = []
    for position, target in PAIRS:
        cam = Camera(position=position, target=target)
        f = cam.get_forward()
        assert f.dtype == np.float32
        out.append(
            {
                "position": cam.position.tolist(),
                "target": cam.target.tolist(),
                "forward": f.tolist(),
            }
        )
    return out


def test_the_forward_fixture_is_numpys():
    """Writes the fixture when it is missing; otherwise it must hold numpy's values."""
    values = _forward()
    if not FORWARD.exists():
        FORWARD.write_text(json.dumps(values, indent=1) + "\n")
    assert json.loads(FORWARD.read_text()) == values


@pytest.fixture(scope="session")
def web_walk_export(tmp_path_factory) -> Path:
    """examples/web_walk.py exported once per session, for the suites that load a real folder."""
    out = tmp_path_factory.mktemp("web_walk_export") / "out"
    assert export_web(None, str(WEB_WALK), out, []) == 0
    return out


@pytest.fixture(scope="session")
def spawner_export(tmp_path_factory) -> Path:
    """tests/web/apps/spawner.py exported, for scene.test.mjs: a library geometry no entity draws."""
    out = tmp_path_factory.mktemp("spawner_export") / "out"
    assert export_web(None, str(SPAWNER), out, []) == 0
    return out


@pytest.fixture(scope="session")
def skinned_export(tmp_path_factory) -> Path:
    """tests/web/apps/skinned.py exported, for scene.test.mjs: three figures and a box."""
    out = tmp_path_factory.mktemp("skinned_export") / "out"
    assert export_web(None, str(SKINNED), out, []) == 0
    return out


@pytest.fixture(scope="session")
def skinned_glb_path(tmp_path_factory) -> Path:
    """A posed skinned figure through Engine.export_gltf, for gltf_skin.test.mjs, with
    expected.json beside it: each vertex where the desktop draws it, in float64."""
    base = tmp_path_factory.mktemp("skinned_glb")
    e, h, geo, P = skinned_figure()
    e.export_gltf(base / "skinned.glb")
    (base / "expected.json").write_text(json.dumps(posed_positions(e, h, geo, P).tolist()))
    return base / "skinned.glb"


@pytest.fixture(scope="session")
def sourcemap_export(tmp_path_factory) -> Path:
    """FAILING exported, with expect.json beside it: the Python lines the error must name."""
    import sys

    base = tmp_path_factory.mktemp("sourcemap_export")
    src = base / "src"
    src.mkdir()
    (src / "mxsm_helper.py").write_text(HELPER)
    (src / "failing.py").write_text(FAILING)
    out = base / "out"
    try:
        assert export_web(None, str(src / "failing.py"), out, []) == 0
    finally:
        sys.modules.pop("mxsm_helper", None)
    lines = {
        "helper": HELPER.splitlines().index("    return spot.x * 2.0") + 1,
        "call": FAILING.splitlines().index("    holder.n = holder.n + far(holder.spot)") + 1,
    }
    (base / "expect.json").write_text(json.dumps(lines))
    return out


def _run(suites: list[str], env: dict[str, str]) -> None:
    # TAP, which prints "# skipped N": newer Node defaults to the spec reporter even off a TTY.
    done = subprocess.run(
        ["node", "--test", "--test-reporter=tap", *suites],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    # A suite that skips for want of the export would pass without checking anything.
    assert "# skipped 0" in done.stdout, done.stdout


def test_node_suites_pass(
    web_walk_export, spawner_export, skinned_export, sourcemap_export, skinned_glb_path
):
    # Node 22's --test takes files or globs, not a folder.
    suites = sorted(str(f) for f in HERE.glob("*.test.mjs"))
    assert suites
    env = {
        **os.environ,
        "MX_WEB_WALK_EXPORT": str(web_walk_export),
        "MX_SPAWNER_EXPORT": str(spawner_export),
        "MX_SKINNED_EXPORT": str(skinned_export),
        "MX_SOURCEMAP_EXPORT": str(sourcemap_export),
        "MX_SKIN_GLB": str(skinned_glb_path),
    }
    _run(suites, env)


def test_sourcemap_names_the_python_lines(sourcemap_export):
    _run(
        [str(HERE / "sourcemap.test.mjs")],
        {**os.environ, "MX_SOURCEMAP_EXPORT": str(sourcemap_export)},
    )
