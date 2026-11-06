"""The runtime's own Node tests (tests/web/js/*.test.mjs), run from pytest so make test
covers them, and the numpy values they compare against."""

import json
import subprocess
from pathlib import Path

import numpy as np

from manifoldx.camera import Camera

HERE = Path(__file__).parent / "js"
FORWARD = HERE / "forward.json"

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


def test_node_suites_pass():
    # Node 22's --test takes files or globs, not a folder.
    suites = sorted(str(f) for f in HERE.glob("*.test.mjs"))
    assert suites
    done = subprocess.run(["node", "--test", *suites], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
