"""The show-off demos export too: their frame code stays inside the subset and runs in Node."""

import json
import subprocess
from pathlib import Path

from manifoldx.web.export import export_web

LABYRINTH = Path(__file__).parents[2] / "demos" / "labyrinth_fps.py"


def _frame(keys):
    return {
        "keys": list(keys),
        "just": [],
        "d": [0.0, 0.0],
        "buttons": [],
        "clicks": [],
        "vp": [1024, 768],
    }


def test_labyrinth_fps_exports_and_walks_in_the_page(tmp_path):
    out = tmp_path / "out"
    assert export_web(None, str(LABYRINTH), out, []) == 0
    report = json.loads((out / "report.json").read_text())
    assert [r["label"] for r in report["roots"]] == ["system fps_controls"]
    assert "__main__.fps_controls" in report["transpiled"]

    # A frame with W down must move the camera: the system's globals and the flat wall
    # grid crossed, and the transpiled controller runs.
    done = subprocess.run(
        ["node", str(out / "replay.mjs"), "--replay"],
        input=json.dumps({"frames": [_frame(()), _frame("w")], "paths": []}),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    idle, walked = (frame["camera"]["position"] for frame in json.loads(done.stdout))
    assert walked != idle
