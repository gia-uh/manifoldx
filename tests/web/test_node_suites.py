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

HERE = Path(__file__).parent / "js"
FORWARD = HERE / "forward.json"
WEB_WALK = Path(__file__).parents[2] / "examples" / "web_walk.py"

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


def test_node_suites_pass(web_walk_export):
    # Node 22's --test takes files or globs, not a folder.
    suites = sorted(str(f) for f in HERE.glob("*.test.mjs"))
    assert suites
    env = {**os.environ, "MX_WEB_WALK_EXPORT": str(web_walk_export)}
    done = subprocess.run(
        ["node", "--test", *suites], capture_output=True, text=True, timeout=120, env=env
    )
    assert done.returncode == 0, done.stdout + done.stderr
    # A suite that skips for want of the export would pass without checking anything.
    assert "# skipped 0" in done.stdout, done.stdout
