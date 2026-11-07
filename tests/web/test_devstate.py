"""The tab's engine.state back to Python: decoded against what Python had, merged into State."""

import json
import math
import shutil
import subprocess

import numpy as np
import pytest

from manifoldx.hot import State
from manifoldx.web.devstate import decode, merge_tab_state
from manifoldx.web.export import capture_engine, export_web

# Every value form the state takes on its way to the tab, and a key that cannot cross.
ORBIT = """
import numpy as np
import manifoldx as mx

engine = mx.Engine("orbit", width=64, height=64)
engine.state.setdefault("cam", {"yaw": 0.5, "pitch": -0.25, "dist": 6.0})
engine.state["pos"] = (1.0, 2.0, 3.0)
engine.state["path"] = [(0.0, 1.0), (2.0, 3.0)]
engine.state["tags"] = {"a", "b"}
engine.state["v"] = np.array([0.1, 0.2], dtype=np.float32)
engine.state["far"] = float("inf")
engine.state[("cache", 1)] = np.zeros((2, 2))
engine.run()
"""


def test_decode_restores_what_the_previous_value_was():
    assert decode([1, 2], like=(0, 0)) == (1, 2)
    assert decode([1, 2], like=[0]) == [1, 2]
    assert type(decode(6, like=0.5)) is float and type(decode(6)) is int
    v = decode({"__f32": [1.5, 2.0]}, like=None)
    assert v.dtype == np.float32 and v.tolist() == [1.5, 2.0]
    assert decode({"__set": ["a"]}) == {"a"}
    assert math.isnan(decode({"__float": "nan"}))
    assert decode({"__dict": [["cam", {"__dict": [["yaw", 0.5]]}]]}) == {"cam": {"yaw": 0.5}}


def test_merge_replaces_crossed_keys_and_keeps_the_rest():
    s = State(yaw=0.0, gone=1, cache={("tex", 1): "big"}, rig=0)
    tab = {"__dict": [["yaw", 0.75], ["rig", {"__skip": "an object of Rig"}], ["new", 3]]}
    kept = merge_tab_state(s, tab, crossed=frozenset({"yaw", "gone", "rig"}))
    assert s == {"yaw": 0.75, "cache": {("tex", 1): "big"}, "rig": 0, "new": 3}
    assert sorted(kept) == [
        "cache: left out of the snapshot",
        "rig: an object of Rig stays as Python had it",
    ]


def _same(a, b) -> bool:
    """Equal values of the same types all the way down; numpy arrays by dtype and items."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return type(a) is type(b) and a.dtype == b.dtype and a.tolist() == b.tolist()
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.skipif(shutil.which("node") is None, reason="node not found")
def test_round_trip_of_the_snapshot_state(tmp_path):
    # export the app, run its state.json through Node's loadApp then encodeState,
    # decode in Python: equal to the State the export started from
    app = tmp_path / "orbit.py"
    app.write_text(ORBIT)
    out = tmp_path / "out"
    assert export_web(None, str(app), out, []) == 0
    engine, _ = capture_engine(None, str(app), [])
    script = (
        f'import {{ load }} from "{(out / "replay.mjs").as_uri()}";\n'
        f'import {{ encodeState }} from "{(out / "manifoldx.js").as_uri()}";\n'
        f'const app = await load(new URL("{out.as_uri()}/"));\n'
        "console.log(JSON.stringify(encodeState(app.engine.state)));\n"
    )
    done = subprocess.run(
        ["node", "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    tab = json.loads(done.stdout)
    started = {k: v for k, v in engine.state.items() if isinstance(k, str)}
    assert _same(decode(tab, like=started), started)
