"""examples/web_crowd.py end to end: one scripted input through the app in Python and through
its web export in Node, compared frame by frame, store included. The script is a fixture the
seeded generator below writes (`pytest tests/web/test_web_crowd.py --regen` rewrites it).
Each breaking test damages the copied runtime one way and must see compare report a
mismatch. tests/web/browser reuses script, PATHS and the two export fixtures."""

import json
import random
import shutil
from pathlib import Path

import pytest

from manifoldx.web.export import export_web
from manifoldx.web.replay import compare, replay_node, replay_python

EXAMPLE = Path(__file__).parents[2] / "examples" / "web_crowd.py"
FIXTURE = Path(__file__).parent / "fixtures" / "web_crowd_script.json"
PATHS = ["waves", "picked"]

SEED, FRAMES, VIEW = 5, 600, [800, 500]
SPACE = (5, 6, 200, 301)  # two waves in a row, one after the first fell, one after the resize
# Left clicks at CSS pixels: over the front box of the first ring (twice, the same box), over
# its neighbour, then on the sky. Measured by projecting the Python trace's rows through the
# camera at (0, 12, 22).
CLICKS = {40: (400.0, 168.6), 41: (400.0, 168.6), 120: (449.0, 317.2), 150: (400.0, 20.0)}
RESIZE = 300


def script(seed: int = SEED, n: int = FRAMES) -> list[dict]:
    """Input frames: Space at SPACE, the pointer wandering over the view and clicking at
    CLICKS, a resize to 640x400 at RESIZE."""
    rng = random.Random(seed)
    frames, view = [], list(VIEW)
    for i in range(n):
        if i == RESIZE:
            view = [640, 400]
        just = ["Space"] if i in SPACE else []
        clicks = [1] if i in CLICKS else []
        if i in CLICKS:
            pos = list(CLICKS[i])
        else:
            pos = [round(rng.uniform(0, view[0]), 1), round(rng.uniform(0, view[1]), 1)]
        frames.append(
            {
                "keys": list(just),
                "just": just,
                "d": [0.0, 0.0],
                "buttons": list(clicks),
                "clicks": clicks,
                "vp": list(view),
                "pos": pos,
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
def crowd_export(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("web_crowd") / "out"
    assert export_web(None, str(EXAMPLE), out, []) == 0
    return out


@pytest.fixture(scope="module")
def crowd_python_trace(frames) -> list[dict]:
    return replay_python(None, str(EXAMPLE), frames, PATHS)


def test_the_fixture_is_the_generators(frames):
    assert frames == script()


def test_web_crowd_replays_in_node(crowd_export, crowd_python_trace, frames):
    mismatch = compare(crowd_python_trace, replay_node(crowd_export, frames, PATHS))
    assert mismatch is None, str(mismatch)


def _materials(entry: dict) -> list[list]:
    return [c for c in entry["log"] if c[0] == "material"]


def test_the_script_does_what_it_says(crowd_python_trace):
    t = crowd_python_trace
    alive = [len(e["store"]["alive"]) for e in t]
    assert alive[7] == 48 + 1  # two waves and the ground
    assert t[7]["state"]["waves"] == 2

    # The first rings fall below the floor and are destroyed before the third wave...
    drops = [i for i in range(8, 200) if alive[i] < alive[i - 1]]
    assert drops, "no box was destroyed before frame 200"
    assert alive[199] == 1
    # ...which spawns into the freed slots, the free list from its end.
    freed = t[199]["store"]["free"]
    assert len(freed) == 48
    assert t[200]["store"]["high_water"] == t[199]["store"]["high_water"] == 49
    assert t[200]["store"]["alive"] == [0, *sorted(freed[-24:])]

    # Frames 40 and 41 pick the same box and paint it gold; 120 picks its neighbour, so the
    # first gets its colour back; the sky at 150 clears the pick.
    first = t[40]["paths"]["picked"]
    assert first is not None
    [[_, entity, gold]] = _materials(t[40])
    assert entity == first
    assert t[41]["paths"]["picked"] == first
    [restore, again] = _materials(t[41])
    assert restore[1] == first and restore[2] != gold
    assert again == ["material", first, gold]
    second = t[120]["paths"]["picked"]
    assert second not in (None, first)
    back, now = _materials(t[120])
    assert back == restore
    assert now == ["material", second, gold]
    assert t[149]["paths"]["picked"] == second
    assert t[150]["paths"]["picked"] is None
    [[_, entity, colour]] = _materials(t[150])
    assert entity == second and colour != gold

    # The resize reaches the engine, and the wave after it is the fourth.
    assert t[301]["state"]["waves"] == 4
    assert len(t[301]["store"]["alive"]) == 1 + 24 + 24
    assert t[400]["state"]["fall"] > 0.0


# Each damages the copied manifoldx.js or pyrt.js one way: [file, old text, new text].
PATCHES = {
    # Store.spawn reuses the free list from its start instead of popping from its end.
    "free list taken from its start": [
        "manifoldx.js",
        "const reused = this.free.slice(this.free.length - k).reverse();",
        "const reused = this.free.slice(0, k);",
    ],
    # FieldView.set and the in-place forms write at once instead of at the flush.
    "query writes applied at once": [
        "manifoldx.js",
        'this._queue.push(["update", name, Array.from(indices), colStart, colEnd, data]);',
        'this._queue.push(["update", name, Array.from(indices), colStart, colEnd, data]);'
        " this.flush();",
    ],
    # $rt.nd.mul computes float32 products in float64 and keeps them unrounded.
    "float32 product left unrounded": [
        "pyrt.js",
        "  mul: binary(times, null, Math.imul),",
        '  mul: (x, y, d) => binary(times, null, Math.imul)(x, y, d === "f32" ? "f64" : d),',
    ],
    # Store.query returns every alive slot, as the desktop did before task 1.
    "queries not filtered": [
        "manifoldx.js",
        "if (this.alive[i] && bits.every((b) => b[i])) out.push(i);",
        "if (this.alive[i]) out.push(i);",
    ],
}


def _copy(export: Path, tmp_path: Path) -> Path:
    out = tmp_path / "out"
    shutil.copytree(export, out)
    return out


def _patch(out: Path, name: str, old: str, new: str) -> None:
    path = out / name
    text = path.read_text()
    assert text.count(old) == 1, f"{name} no longer holds the text the patch replaces: {old}"
    path.write_text(text.replace(old, new))


@pytest.mark.parametrize("patch", list(PATCHES))
def test_each_break_is_a_mismatch(crowd_export, crowd_python_trace, frames, tmp_path, patch):
    out = _copy(crowd_export, tmp_path)
    _patch(out, *PATCHES[patch])
    assert compare(crowd_python_trace, replay_node(out, frames, PATHS)) is not None
