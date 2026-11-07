"""examples/web_crowd.py end to end: one scripted input through the app in Python and through
its web export in Node, compared frame by frame, store included. The script is a fixture the
seeded generator below writes (`pytest tests/web/test_web_crowd.py --regen` rewrites it).
Each breaking test damages the copied runtime one way and must see compare report a
mismatch. tests/web/browser reuses script, PATHS and the two export fixtures."""

import json
import math
import random
import shutil
from pathlib import Path

import pytest

from manifoldx.web.export import export_web
from manifoldx.web.replay import compare, replay_node, replay_python

EXAMPLE = Path(__file__).parents[2] / "examples" / "web_crowd.py"
FIXTURE = Path(__file__).parent / "fixtures" / "web_crowd_script.json"
PATHS = ["waves", "picked", "throws", "g", "fallen", "walked"]

SEED, FRAMES, VIEW = 5, 600, [800, 500]
SPACE = (5, 6, 200, 301)  # two waves in a row, one after the first fell, one after the resize
# Left clicks at CSS pixels: over the front box of the first ring (twice, the same box), over
# its neighbour, then on the sky. Measured by projecting the Python trace's rows through the
# camera at (0, 12, 22).
CLICKS = {40: (400.0, 168.6), 41: (400.0, 168.6), 120: (449.0, 317.2), 150: (400.0, 20.0)}
RESIZE = 300
CLOCK = 1762500000.0  # the first frame's clock; each frame adds 1/60
# The controls' indices in state.json's gui (test_the_widget_indices_are_the_exports), and
# where each one sits on the page, (left, top, width, height) in CSS pixels, for the x and y of
# the pointer events routed to it.
WIDGETS = {"wave": 7, "gravity": 8, "spin": 9, "wind": 11}
BOXES = {"wave": (24, 152, 120, 24), "gravity": (24, 182, 160, 16), "spin": (24, 204, 60, 16), "wind": (24, 226, 96, 96)}  # fmt: skip
# Pointer events routed to a control, by frame: (phase, control, fx, fy, pointer_id). Two taps
# on the Button; a drag on the Slider from fx 0.2 to 0.9; a tap on the Toggle; the Joystick
# held at (0.8, 0.5) while a second finger (pointer 4) taps the Button; a tap on the Slider at
# its middle, so the ArrowLeft after it steps the Slider.
TAP = lambda name, fx, fy, pid=0: [("down", name, fx, fy, pid), ("up", name, fx, fy, pid)]  # noqa: E731
ROUTED = {
    60: TAP("wave", 0.5, 0.5),
    61: TAP("wave", 0.5, 0.5),
    80: [("down", "gravity", 0.2, 0.5, 0)],
    **{i: [("move", "gravity", round(0.2 + 0.07 * (i - 80), 3), 0.5, 0)] for i in range(81, 90)},
    90: [("up", "gravity", 0.9, 0.5, 0)],
    100: TAP("spin", 0.5, 0.5),
    220: [("down", "wind", 0.8, 0.5, 0)],
    230: [("move", "wind", 0.8, 0.5, 0)],
    240: TAP("wave", 0.5, 0.5, 4),
    260: [("up", "wind", 0.8, 0.5, 0)],
    275: TAP("gravity", 0.5, 0.5),
}
HELD = [(80, 90), (220, 260)]  # the pointer stays on its control: no wandering moves
WHEEL = {270: 120.0}  # dy, CSS pixels
KEYS = {
    280: "ArrowLeft",
    400: "f",
    460: "f",
}  # pressed, released the next frame; f pauses the figure


class _Page:
    """The page's FrameBuilder (src/manifoldx/web/js/input.js), as far as the script needs it:
    keys, the left button shared by every pointer, wheel pixels, the events in order, a resize
    event when the view changes."""

    def __init__(self):
        self.held, self.buttons, self.pos, self.last, self.view = (
            set(),
            set(),
            [0.0, 0.0],
            None,
            None,
        )
        self._reset()

    def _reset(self):
        self.just, self.up, self.clicks, self.released = set(), set(), set(), set()
        self.d, self.wheel, self.events = [0.0, 0.0], [0.0, 0.0], []

    def key(self, name: str, down: bool):
        if down:
            self.held.add(name)
            self.just.add(name)
        else:
            self.held.discard(name)
            self.up.add(name)
        event = {"key": name, "modifiers": [], "is_down": down}
        self.events.append(["key_down" if down else "key_up", event])

    def pointer(self, phase: str, x: float, y: float, gui=None, pointer_id: int = 0):
        if phase == "down":
            if 1 not in self.buttons:
                self.clicks.add(1)
            self.buttons.add(1)
        elif phase == "up" and 1 in self.buttons:
            self.buttons.discard(1)
            self.released.add(1)
        moved = phase == "move" and self.last is not None
        dx, dy = (x - self.last[0], y - self.last[1]) if moved else (0.0, 0.0)
        if phase == "move":
            self.d = [self.d[0] + dx, self.d[1] + dy]
        self.pos = [x, y]
        self.last = [x, y]
        event = {"x": x, "y": y, "dx": dx, "dy": dy, "button": 0 if phase == "move" else 1,
                 "buttons": sorted(self.buttons), "modifiers": [], "phase": phase, "pointer_id": pointer_id}  # fmt: skip
        if gui is not None:
            event["gui"] = gui
        self.events.append([f"pointer_{phase}", event])

    def scroll(self, dy: float):
        self.wheel[1] += dy
        x, y = self.pos
        self.events.append(["wheel", {"dx": 0.0, "dy": dy, "x": x, "y": y, "buttons": sorted(self.buttons), "modifiers": []}])  # fmt: skip

    def take(self, view: list[int], clock: float) -> dict:
        if self.view is not None and view != self.view:
            self.events.append(["resize", {"width": view[0], "height": view[1], "pixel_ratio": 1}])
        self.view = list(view)
        frame = {
            "keys": sorted(self.held), "just": sorted(self.just), "d": list(self.d),
            "buttons": sorted(self.buttons), "clicks": sorted(self.clicks), "vp": list(view),
            "up": sorted(self.up), "released": sorted(self.released), "pos": list(self.pos),
            "wheel": list(self.wheel), "mods": [], "events": self.events, "clock": clock,
        }  # fmt: skip
        self._reset()
        return frame


def _at(name: str, fx: float, fy: float) -> tuple[float, float]:
    left, top, w, h = BOXES[name]
    return round(left + fx * w, 1), round(top + fy * h, 1)


def script(seed: int = SEED, n: int = FRAMES) -> list[dict]:
    """Input frames as the page records them: Space at SPACE, the pointer wandering over the
    view (a pointer_move each frame) and clicking at CLICKS, a resize to 640x400 at RESIZE,
    the controls' pointer events at ROUTED, routed to them, a wheel at WHEEL, KEYS, and a
    clock from CLOCK. A press is released the frame after it, as a person's is."""
    rng = random.Random(seed)
    page, frames, view = _Page(), [], list(VIEW)
    for i in range(n):
        if i == RESIZE:
            view = [640, 400]
        if i - 1 in SPACE:
            page.key("Space", False)
        if i - 1 in KEYS:
            page.key(KEYS[i - 1], False)
        if i - 1 in CLICKS:
            page.pointer("up", *CLICKS[i - 1])
        if i in SPACE:
            page.key("Space", True)
        if i in KEYS:
            page.key(KEYS[i], True)
        if i in CLICKS:
            page.pointer("move", *CLICKS[i])
            page.pointer("down", *CLICKS[i])
        else:
            wander = [round(rng.uniform(0, view[0]), 1), round(rng.uniform(0, view[1]), 1)]
            if i in ROUTED:
                for phase, name, fx, fy, pid in ROUTED[i]:
                    gui = [WIDGETS[name], fx, fy]
                    page.pointer(phase, *_at(name, fx, fy), gui=gui, pointer_id=pid)
            elif not any(a < i < b for a, b in HELD):
                page.pointer("move", *wander)
        if i in WHEEL:
            page.scroll(WHEEL[i])
        frames.append(page.take(view, CLOCK + i / 60))
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


def test_the_widget_indices_are_the_exports(crowd_export):
    state = json.loads((crowd_export / "state.json").read_text())
    found = {}

    def walk(node):
        if "name" in node:
            found[node["name"]] = node["widget"]
        for c in node.get("children", ()):
            walk(c)

    for p in state["gui"]["panels"]:
        walk(p)
    assert found == WIDGETS


def _events(entry: dict) -> list[str]:
    return [e for e in entry["events"] if e not in ("frame", "pointer_move")]


def test_the_script_does_what_it_says(crowd_python_trace):
    t = crowd_python_trace
    alive = [len(e["store"]["alive"]) for e in t]
    assert alive[7] == 48 + 2  # two waves, the ground and the figure
    assert t[7]["state"]["waves"] == 2

    # The Button's two clicks at 60 and 61: each click event the frame after its pointer, each
    # throw the frame after that, each wave in the frame of its throw.
    assert _events(t[60]) == ["pointer_down", "pointer_up"]
    assert t[60]["state"]["waves"] == 2
    assert "gui:button:wave:click" in t[61]["events"] and "gui:button:wave:click" in t[62]["events"]
    assert "throw" in t[62]["events"] and "throw" in t[63]["events"]
    assert [t[i]["state"]["waves"] for i in (61, 62, 63, 65)] == [2, 3, 4, 4]
    assert alive[63] == 4 * 24 + 2

    # The first rings fall below the floor and are destroyed before the fifth wave, the thrown
    # ones still up; it spawns into the freed slots, the free list from its end.
    drops = [i for i in range(8, 200) if alive[i] < alive[i - 1]]
    assert drops, "no box was destroyed before frame 200"
    assert alive[199] == 2 + 48
    freed = t[199]["store"]["free"]
    assert len(freed) == 48
    assert t[200]["store"]["high_water"] == t[199]["store"]["high_water"] == 98
    assert t[200]["store"]["alive"] == sorted([*t[199]["store"]["alive"], *freed[-24:]])
    assert t[199]["state"]["fallen"] == 48 and t[230]["state"]["fallen"] == 96
    assert t[599]["state"]["fallen"] == 7 * 24 and alive[599] == 2

    # Frames 40 and 41 pick the same box and paint it gold; the Button's click at 60 clears
    # the pick, giving its colour back; 120 picks its neighbour; the sky at 150 clears it.
    first = t[40]["paths"]["picked"]
    assert first is not None
    [[_, entity, gold]] = _materials(t[40])
    assert entity == first
    assert t[41]["paths"]["picked"] == first
    [restore, again] = _materials(t[41])
    assert restore[1] == first and restore[2] != gold
    assert again == ["material", first, gold]
    assert _materials(t[60]) == [restore] and t[60]["paths"]["picked"] is None
    second = t[120]["paths"]["picked"]
    assert second not in (None, first)
    assert _materials(t[120]) == [["material", second, gold]]
    assert t[149]["paths"]["picked"] == second
    assert t[150]["paths"]["picked"] is None
    [[_, entity, colour]] = _materials(t[150])
    assert entity == second and colour != gold

    # The Slider: the drag's values from frame 81, its release's commit at 91 (0.9 * 12);
    # the tap at 275 and the ArrowLeft at 280 step it to 5.4, change and commit.
    assert t[80]["paths"]["g"] == 6.0
    assert t[81]["paths"]["g"] == 0.2 * 12.0
    assert t[90]["state"]["commits"] == []
    assert t[91]["state"]["commits"] == [0.9 * 12.0]
    assert t[276]["paths"]["g"] == 6.0
    assert t[281]["paths"]["g"] == 6.0 - 12.0 / 20
    assert t[281]["state"]["commits"] == [0.9 * 12.0, 6.0, 6.0 - 12.0 / 20]
    # The Toggle stops the spin: its change at 101, the rotation frozen after it.
    assert t[100]["state"]["spins"] == 0 and t[101]["state"]["spins"] == 1
    # The Joystick at (0.8, 0.5) of its box: (0.6, 0) while held, kept through the tap of
    # pointer 4 on the Button at 240 (which throws the sixth wave at 242), (0, 0) after 260.
    assert t[230]["state"]["wind"][0] == pytest.approx(0.6)
    assert t[241]["state"]["wind"][0] == pytest.approx(0.6)
    assert t[240]["state"]["pointer_down"][2] == 4
    assert t[242]["state"]["waves"] == 6
    assert t[261]["state"]["wind"] == [0.0, 0.0]
    # The wheel at 270 moves the camera out.
    assert t[270]["state"]["zoom"] == pytest.approx(1.24)
    assert t[271]["camera"]["position"][2] == pytest.approx(22.0 * 1.24)
    # The resize reaches the engine and its handler, and the wave after it is the seventh.
    assert t[300]["state"]["resize"] == [640, 400]
    assert t[301]["state"]["waves"] == 7
    assert alive[301] == 2 + 3 * 24
    assert t[400]["state"]["fall"] > 0.0
    # The clock and the frame event.
    assert t[60]["state"]["sec"] == pytest.approx(1.0, abs=1e-5)
    assert t[60]["state"]["elapsed"] == pytest.approx(1.0)
    # Every handler ran.
    names = {e for entry in t for e in entry["events"]}
    assert names >= {
        "frame", "key_down", "key_up", "pointer_down", "pointer_up", "pointer_move", "wheel",
        "resize", "gui:button:wave:click", "gui:slider:gravity:change", "gui:slider:gravity:commit",
        "gui:toggle:spin:change", "gui:joystick:wind:change", "throw", "fell",
    }  # fmt: skip
    last = t[-1]["state"]
    assert last["key_down"] == last["key_up"] == 7  # Space four times, ArrowLeft and f twice


FIGURE = 1  # the walking figure's slot, spawned after the ground


def _figure(entry: dict) -> tuple[list[float], list[list[float]]]:
    """The figure's Transform row and its joints in a trace entry."""
    alive = entry["store"]["alive"]
    return entry["store"]["rows"]["Transform"][alive.index(FIGURE)], entry["joints"][str(FIGURE)]


def test_the_figure_walks_and_swings(crowd_python_trace):
    t = crowd_python_trace
    row, joints = _figure(t[599])
    assert abs(math.hypot(row[0], row[2]) - 6.0) < 1e-4
    assert row[1] == 0.0
    assert len(joints) == 6
    assert _figure(t[100])[1] != _figure(t[200])[1]
    # f at 400 pauses it: pose, place and walked hold until f at 460 resumes it
    held = [_figure(t[i]) for i in range(399, 460)]
    assert all(h == held[0] for h in held)
    assert {t[i]["paths"]["walked"] for i in range(399, 460)} == {t[399]["paths"]["walked"]}
    assert _figure(t[460]) != held[0] and t[460]["paths"]["walked"] > t[399]["paths"]["walked"]
    # 1.2 m/s over the frames it walked
    assert t[599]["paths"]["walked"] == pytest.approx(1.2 * (600 - 60) / 60)


def test_a_figure_drawn_at_rest_is_a_mismatch(crowd_export, crowd_python_trace, frames, tmp_path):
    out = _copy(crowd_export, tmp_path)
    _patch(
        out,
        "manifoldx.js",
        "  set_joints(matrices) {\n",
        "  set_joints(matrices) {\n    return null;\n",
    )
    mismatch = compare(crowd_python_trace, replay_node(out, frames, PATHS))
    assert mismatch is not None and mismatch.frame == 0, str(mismatch)


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
    # EventBus.drain keeps one queue, so an emit during delivery is delivered at once.
    "emits delivered in the same frame": [
        "manifoldx.js",
        "    const drained = this.pending;\n    this.pending = [];\n"
        "    for (const [name, payload] of drained) {\n"
        "      this.delivered.push(name);\n      deliver(name, payload);\n    }\n",
        "    const drained = this.pending;\n"
        "    for (const [name, payload] of drained) {\n"
        "      this.delivered.push(name);\n      deliver(name, payload);\n    }\n"
        "    this.pending = [];\n",
    ],
    # The Slider's move hook queues its change before it sets the value.
    "a widget's value after its event": [
        "manifoldx.js",
        "    move(w, ev, engine) {\n      w.value = sliderValue(w, ev.x);\n"
        '      engine.emit(`gui:slider:${w.name}:change`, new Map([["value", w.value]]));\n',
        "    move(w, ev, engine) {\n"
        '      engine.emit(`gui:slider:${w.name}:change`, new Map([["value", w.value]]));\n'
        "      w.value = sliderValue(w, ev.x);\n",
    ],
    # The export's generator is not loaded: a generator of an all-zero state stands in.
    "the random state not loaded": [
        "manifoldx.js",
        'if ("__random" in e) return Random.fromState(e.__random);',
        'if ("__random" in e) return Random.fromState({ key: new Array(624).fill(0), pos: 624, gauss: null });',
    ],
    # The frame's input events go before what the previous frame emitted.
    "events after the frame's input": [
        "manifoldx.js",
        'engine.bus.pending.unshift(["frame", payload]);',
        "engine.bus.pending.sort((a, b) => (EVENT_FIELDS[b[0]] !== undefined) - (EVENT_FIELDS[a[0]] !== undefined));"
        ' engine.bus.pending.unshift(["frame", payload]);',
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
