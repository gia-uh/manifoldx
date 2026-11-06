"""Replay: one scripted input through examples/web_walk.py in Python and through its web export
in Node, compared frame by frame. The script is a fixture the seeded generator below writes
(`pytest tests/web/test_replay.py --regen` rewrites it). Each breaking test damages the export
one way and must see compare report a mismatch."""

import json
import random
import shutil
from pathlib import Path

import pytest

from manifoldx.web.export import export_web
from manifoldx.web.replay import compare, replay_node, replay_python

EXAMPLE = Path(__file__).parents[2] / "examples" / "web_walk.py"
FIXTURE = Path(__file__).parent / "fixtures" / "web_walk_script.json"
PATHS = ["rig.pos", "rig.yaw", "rig.mode", "selected"]

SEED, FRAMES, VIEW = 10, 600, [800, 500]
LOOK = 0.004  # the example's radians per pixel, so the script can turn toward what it aims at


def script(seed: int = SEED, n: int = FRAMES) -> list[dict]:
    """Input frames: walk, look while captured, right-drag, a click every 12 frames (the first
    captures the pointer, the next ones pick), Escape twice, Tab into and out of fly mode, and
    a resize."""
    rng = random.Random(seed)
    frames, held, yaw, pitch = [], [], 0.0, 0.0
    captured, dragging, view = False, False, list(VIEW)
    for i in range(n):
        if i % 30 == 0:  # a new move: hold one or two walk keys, or none
            held = rng.sample(["w", "a", "s", "d", "w"], rng.choice([0, 1, 1, 2]))
            # The view the next moves turn toward: ahead, where the boxes are, and a bit down.
            aim_yaw, aim_pitch = rng.uniform(-1.3, 1.3), rng.uniform(-0.3, 0.0)
            dragging = not captured and rng.random() < 0.6
        just, clicks = [], []
        if i in (200, 330):
            just.append("Tab")
        if i in (150, 420):
            just.append("Escape")
        if i == 450:
            view = [640, 480]
        if i % 12 == 5:
            clicks.append(1)
        keys = list(held)
        if 200 <= i < 330:  # flying: rise a while, then sink
            keys.append("Space" if i < 260 else "Shift")
        d = [0.0, 0.0]
        if captured or dragging:
            dx = max(-20.0, min(20.0, round((aim_yaw - yaw) / LOOK * 0.1, 1)))
            d = [dx, round((pitch - aim_pitch) / LOOK * 0.1, 1)]
            yaw += d[0] * LOOK
            pitch = max(-1.4, min(1.4, pitch - d[1] * LOOK))
        if "Escape" in just:
            captured = False
        if clicks and not captured:
            captured = True
            clicks = [1]
        frames.append(
            {
                "keys": keys + just,
                "just": just,
                "d": d,
                "buttons": ([2] if dragging and not captured else []) + clicks,
                "clicks": clicks,
                "vp": list(view),
            }
        )
    return frames


@pytest.fixture(scope="module")
def frames(request):
    if request.config.getoption("--regen", default=False) or not FIXTURE.exists():
        FIXTURE.parent.mkdir(exist_ok=True)
        FIXTURE.write_text("[\n" + ",\n".join(json.dumps(f) for f in script()) + "\n]\n")
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("web_walk") / "out"
    assert export_web(None, str(EXAMPLE), out, []) == 0
    return out


@pytest.fixture(scope="module")
def python_trace(frames):
    return replay_python(None, str(EXAMPLE), frames, PATHS)


def test_the_export_replays_the_python_app(exported, python_trace, frames):
    node_trace = replay_node(exported, frames, PATHS)
    mismatch = compare(python_trace, node_trace)
    assert mismatch is None, str(mismatch)
    selected = {f["paths"]["selected"] for f in python_trace} - {None}
    materials = [e for f in python_trace for e in f["log"] if e[0] == "material"]
    print(
        f"\nreplayed {len(python_trace)} frames: {len(selected)} distinct selections "
        f"({', '.join(sorted(selected))}), {len(materials)} material entries"
    )
    assert len(python_trace) == len(frames) == FRAMES
    assert len(selected) >= 3
    assert materials


def _copy(exported: Path, tmp_path: Path) -> Path:
    out = tmp_path / "broken"
    shutil.copytree(exported, out)
    return out


def _patch(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, f"{path.name} no longer holds {old!r}"
    path.write_text(text.replace(old, new))


def _mismatch(python_trace, out, frames):
    mismatch = compare(python_trace, replay_node(out, frames, PATHS))
    assert mismatch is not None
    print(f"\n{mismatch}")


def test_a_pick_that_returns_null_is_a_mismatch(exported, python_trace, frames, tmp_path):
    out = _copy(exported, tmp_path)
    head = "export function pick(cam) {"
    _patch(out / "app" / "__main__.js", head, head + "\n  return null;")
    _mismatch(python_trace, out, frames)


def test_a_camera_held_as_a_plain_array_is_a_mismatch(exported, python_trace, frames, tmp_path):
    out = _copy(exported, tmp_path)
    _patch(out / "manifoldx.js", "Float32Array.from(position)", "Array.from(position)")
    _patch(out / "manifoldx.js", "Float32Array.from(target)", "Array.from(target)")
    _mismatch(python_trace, out, frames)


def test_boxes_dropped_from_the_snapshot_are_a_mismatch(exported, python_trace, frames, tmp_path):
    out = _copy(exported, tmp_path)
    state = json.loads((out / "state.json").read_text())
    assert "boxes" in state["globals"]["__main__"]
    state["globals"]["__main__"]["boxes"] = []
    (out / "state.json").write_text(json.dumps(state))
    _mismatch(python_trace, out, frames)
