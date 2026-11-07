"""examples/web_crowd.py in headless Chromium: tests/web/test_web_crowd.py's script driven
through Playwright's keyboard and mouse runs as Python runs it on the frames the page built,
the page draws one call per non-empty (geometry, material) batch and one for the skinned
figure, whose bones are the pose Python gave it, and a picked box is drawn by the batch of its
material. Each control works under a real pointer, a Joystick and a Button
under two fingers at once, and the frames the page records say what the pointer did.
tests/web/browser/test_visual.py checks the batched scene against the desktop's: lit_yard
exports as a library and draws through batches."""

import json
import re
import shutil

from manifoldx.web.replay import compare, replay_python

from ..test_web_crowd import FIGURE, PATHS, SPACE, WIDGETS, script
from .conftest import BATCHES_OF, WEB_CROWD, touch


def widget_box(page, name: str) -> dict:
    """The bounding box ({x, y, width, height}, CSS pixels) of the control of that name."""
    return page.locator(f'[data-mx-widget="{WIDGETS[name]}"]').bounding_box()


def _on(page, name: str, fx: float, fy: float) -> tuple[float, float]:
    b = widget_box(page, name)
    return b["x"] + fx * b["width"], b["y"] + fy * b["height"]


def tick(page) -> dict:
    return page.evaluate("__mx.tick(1 / 60)")


def drive(page, n: int | None = None, each=None) -> list[dict]:
    """The script's first n frames (all by default), each frame's events made with Playwright's
    keyboard, mouse and wheel in their order, then one __mx.tick. A pointer event routed to a
    control goes to its place (fx, fy) in that control's box on the page; the others go to their
    x and y. The tap of a second finger (a pointer_id other than 0) is left out: the mouse is one
    pointer, and test_a_joystick_and_a_button_at_once taps with two. Returns what each tick
    returned, or what each(page) returned after it."""
    page.evaluate(f"() => {{ __mx.paths = {json.dumps(PATHS)}; }}")
    ran, view, at = [], None, None
    for f in script()[:n]:
        if f["vp"] != view:
            view = f["vp"]
            page.set_viewport_size({"width": view[0], "height": view[1]})
        for name, ev in f["events"]:
            if name in ("key_down", "key_up"):
                (page.keyboard.down if ev["is_down"] else page.keyboard.up)(ev["key"])
            elif name.startswith("pointer_"):
                if ev["pointer_id"] != 0:
                    continue
                if "gui" in ev:
                    index, fx, fy = ev["gui"]
                    [control] = [k for k, v in WIDGETS.items() if v == index]
                    xy = _on(page, control, fx, fy)
                else:
                    xy = (ev["x"], ev["y"])
                if xy != at:
                    page.mouse.move(*xy)
                    at = xy
                if name == "pointer_down":
                    page.mouse.down()
                elif name == "pointer_up":
                    page.mouse.up()
            elif name == "wheel":
                page.mouse.wheel(ev["dx"], ev["dy"])
        ran.append(tick(page) if each is None else (tick(page), each(page))[1])
    return ran


def test_chromium_matches_python(open_page, served, crowd_export):
    page = open_page(served(crowd_export))
    ran = drive(page)
    frames = page.evaluate("__mx.frames")
    trace = page.evaluate("__mx.trace")
    assert len(frames) == len(script())
    # the frames the page built hold the script's input
    assert [i for i, f in enumerate(frames) if f["just"] == ["Space"]] == list(SPACE)
    clicks = [i for i, f in enumerate(script()) if f["clicks"]]
    assert [i for i, f in enumerate(frames) if f["clicks"] == [1]] == clicks
    assert frames[300]["vp"] == [640, 400]

    python = replay_python(None, str(WEB_CROWD), frames, PATHS)
    mismatch = compare(python, trace)
    assert mismatch is None, str(mismatch)

    # what the script does happened in the page: waves (four by Space, two by the Button),
    # destroys, a pick in gold, the Slider dragged and stepped, the Joystick held
    assert trace[-1]["state"]["waves"] == 6
    alive = [len(e["store"]["alive"]) for e in trace]
    assert alive[7] == 50 and alive[63] == 98
    # boxes fell and were destroyed before the fifth Space wave (when exactly depends on the
    # gravity the drag left, and Playwright's mouse moves to the Slider's end before its up)
    assert any(alive[i] < alive[i - 1] for i in range(64, 200))
    gold = page.evaluate("__mx.app.modules.__main__.GOLD.id")
    assert ran[40]["entry"]["paths"]["picked"] is not None
    assert ["material", ran[40]["entry"]["paths"]["picked"], gold] in ran[40]["entry"]["log"]
    assert trace[91]["state"]["commits"] == [0.9 * 12.0]
    assert trace[281]["paths"]["g"] == 6.0 - 12.0 / 20
    assert trace[230]["state"]["wind"][0] > 0.5 and trace[261]["state"]["wind"] == [0.0, 0.0]


DRAWN = "[...__mx.scene.batches.values()].filter((b) => b.count > 0).length"
# Each bone's matrixWorld (column-major) of the figure's SkinnedMesh.
BONES = f"__mx.scene.skinned.get({FIGURE}).skeleton.bones.map((b) => Array.from(b.matrixWorld.elements))"


def test_one_draw_call_per_batch(open_page, served, crowd_export):
    page = open_page(served(crowd_export))
    drive(page, 42)  # two waves, and frame 41 painted one box gold
    assert page.evaluate(DRAWN) == 4  # the ground, two colours of 23 and 24 boxes, the gold one
    assert page.evaluate("[...__mx.scene.skinned.keys()]") == [FIGURE]


def test_draw_calls_are_the_batches_plus_the_figure(open_page, served, crowd_export):
    page = open_page(served(crowd_export))
    assert page.evaluate("__mx.renderer.info.autoReset") is True
    counts = drive(page, 42, lambda p: p.evaluate(f"[__mx.renderer.info.render.calls, {DRAWN}]"))
    assert all(calls == drawn + 1 for calls, drawn in counts), counts
    assert counts[-1] == [5, 4]


def test_the_figure_is_posed_as_python_poses_it(open_page, served, crowd_export):
    page = open_page(served(crowd_export))
    bones = drive(page, 30, lambda p: p.evaluate(BONES))
    python = replay_python(None, str(WEB_CROWD), page.evaluate("__mx.frames"), PATHS)
    for i, (entry, drawn) in enumerate(zip(python, bones)):
        pose = entry["joints"][str(FIGURE)]
        assert len(drawn) == len(pose) == 6
        for j, (m, b) in enumerate(zip(pose, drawn)):
            want = [m[4 * r + c] for c in range(4) for r in range(4)]  # column-major
            assert max(abs(x - y) for x, y in zip(want, b)) <= 1e-6, f"frame {i} joint {j}"
    assert python[0]["joints"] != python[29]["joints"]


def test_an_entitys_material_is_read_from_its_batch(open_page, served, crowd_export):
    page = open_page(served(crowd_export))
    ran = drive(page, 41)  # frame 40 picks the front box of the first ring
    picked = ran[40]["entry"]["paths"]["picked"]
    assert picked is not None
    gold = page.evaluate("__mx.app.modules.__main__.GOLD.id")
    keys = page.evaluate(BATCHES_OF, picked)
    assert len(keys) == 1, f"batches drawing entity {picked}: {keys}"
    assert page.evaluate(
        "([key, id]) => __mx.scene.batches.get(key).material === __mx.scene.materials.get(id)",
        [keys[0], gold],
    ), f"entity {picked} is drawn by batch {keys[0]}, not with GOLD"
    # its neighbour keeps its wave's colour
    red = page.evaluate("__mx.app.modules.__main__.COLORS[0].id")
    others = [e for e in ran[40]["entry"]["store"]["alive"] if e not in (0, FIGURE, picked)]
    [key] = page.evaluate(BATCHES_OF, others[0])
    assert page.evaluate(
        "([key, id]) => __mx.scene.batches.get(key).material === __mx.scene.materials.get(id)",
        [key, red],
    )


def use_each_widget(page) -> dict:
    """Each control under the mouse, one tick after each step: Space; a click on the Button; a
    drag on the Slider from 20% to 90% in five moves and past its end; a click on the Toggle;
    the Joystick dragged to its right edge and back; a click on the Slider and ArrowLeft twice;
    a wheel over the scene; a resize to 640x400; three more ticks. Returns the frame of each
    step by name, a list for the Slider's moves and the arrows."""
    page.evaluate(f"() => {{ __mx.paths = {json.dumps(PATHS)}; }}")
    marks: dict = {"drag": [], "arrows": []}
    frame = 0

    def step(mark=None):
        nonlocal frame
        tick(page)
        if mark in ("drag", "arrows"):
            marks[mark].append(frame)
        elif mark is not None:
            marks[mark] = frame
        frame += 1

    page.keyboard.down("Space")
    step("space")
    page.keyboard.up("Space")
    step()
    page.mouse.click(*_on(page, "wave", 0.5, 0.5))
    step("button")
    page.mouse.move(*_on(page, "gravity", 0.2, 0.5))
    page.mouse.down()
    step("slider_down")
    for fx in (0.34, 0.48, 0.62, 0.76, 0.9, 1.2):
        page.mouse.move(*_on(page, "gravity", fx, 0.5))
        step("drag")
    page.mouse.up()
    step("slider_up")
    page.mouse.click(*_on(page, "spin", 0.1, 0.5))
    step("toggle")
    page.mouse.move(*_on(page, "wind", 0.5, 0.5))
    page.mouse.down()
    step("joystick_down")
    page.mouse.move(*_on(page, "wind", 1.0, 0.5))
    step("joystick_edge")
    page.mouse.move(*_on(page, "wind", 0.5, 0.5))
    step("joystick_back")
    page.mouse.up()
    step("joystick_up")
    page.mouse.click(*_on(page, "gravity", 0.5, 0.5))
    step("slider_click")
    for _ in range(2):
        page.keyboard.press("ArrowLeft")
        step("arrows")
    page.mouse.move(600, 400)
    page.mouse.wheel(0, 120)
    step("wheel")
    page.set_viewport_size({"width": 640, "height": 400})
    step("resize")
    for _ in range(3):
        step()
    return marks


def test_each_widget_under_the_mouse_replays(open_page, served, crowd_export):
    page = open_page(served(crowd_export))
    marks = use_each_widget(page)
    frames = page.evaluate("__mx.frames")
    trace = page.evaluate("__mx.trace")
    mismatch = compare(replay_python(None, str(WEB_CROWD), frames, PATHS), trace)
    assert mismatch is None, str(mismatch)
    # each control did what it does: two waves, the Slider at its end and stepped back twice,
    # the spin off, the wind back to rest, the camera moved out, the view resized
    last = trace[-1]["state"]
    assert last["waves"] == 2
    assert last["g"] == 6.0 - 12.0 / 20 - 12.0 / 20
    assert last["commits"][0] == 12.0
    assert last["spins"] == 1
    assert trace[marks["joystick_edge"] + 1]["state"]["wind"] == [1.0, 0.0]
    assert last["wind"] == [0.0, 0.0]
    assert last["zoom"] > 1.0
    assert last["resize"] == [640, 400]


def _routed(frame: dict, name: str) -> list:
    return [ev["gui"] for n, ev in frame["events"] if n == name and "gui" in ev]


def test_the_frame_checks(open_page, served, crowd_export):
    page = open_page(served(crowd_export))
    marks = use_each_widget(page)
    frames = page.evaluate("__mx.frames")
    trace = page.evaluate("__mx.trace")
    # The Button's press: routed to it, a click in the frame; its click event the frame after,
    # the throw it emits the frame after that.
    k = marks["button"]
    assert [g[0] for g in _routed(frames[k], "pointer_down")] == [WIDGETS["wave"]]
    assert frames[k]["clicks"] == [1]
    assert "gui:button:wave:click" in trace[k + 1]["events"]
    assert "throw" in trace[k + 2]["events"]
    # The Slider's drag: every move routed to it, its fx rising, the last past its end.
    moves = [g for i in marks["drag"] for g in _routed(frames[i], "pointer_move")]
    assert len(moves) == len(marks["drag"])
    assert all(g[0] == WIDGETS["gravity"] for g in moves)
    fx = [g[1] for g in moves]
    assert fx == sorted(fx) and len(set(fx)) == len(fx)
    assert fx[-1] > 1.0
    # The clock never runs backwards.
    clocks = [f["clock"] for f in frames]
    assert all(b >= a for a, b in zip(clocks, clocks[1:]))
    # Each arrow is delivered with its frame and steps the Slider, change and commit, the frame
    # after.
    for a in marks["arrows"]:
        assert "key_down" in trace[a]["events"]
        assert {"gui:slider:gravity:change", "gui:slider:gravity:commit"} <= set(trace[a + 1]["events"])  # fmt: skip


def two_fingers(page) -> dict:
    """Finger A down on the Joystick and moved right; finger B down and up on the Button while
    A holds; A moved up; A lifted. One tick after each step; returns the frame of each."""
    page.evaluate(f"() => {{ __mx.paths = {json.dumps(PATHS)}; }}")
    a = _on(page, "wind", 0.5, 0.5)
    right, up = _on(page, "wind", 0.8, 0.5), _on(page, "wind", 0.8, 0.2)
    b = _on(page, "wave", 0.5, 0.5)
    steps = [
        ("a_down", {1: a}),
        ("a_right", {1: right}),
        ("b_down", {1: right, 2: b}),
        ("b_up", {1: right}),
        ("a_up_move", {1: up}),
        ("a_lift", {}),
    ]
    marks = {}
    for frame, (mark, points) in enumerate(steps):
        touch(page, points)
        tick(page)
        marks[mark] = frame
    for _ in range(3):
        tick(page)
    return marks


def joystick_follows_finger_a(trace: list[dict], marks: dict) -> str | None:
    """None when the wind followed finger A through finger B's tap, else what went wrong."""
    wind = lambda m: trace[marks[m] + 1]["state"]["wind"]  # noqa: E731
    if not wind("a_right")[0] > 0.5:
        return f"A's move right gave {wind('a_right')}"
    for m in ("b_down", "b_up"):
        if wind(m) != wind("a_right"):
            return f"B's tap ({m}) moved the wind to {wind(m)}"
    if not wind("a_up_move")[1] > 0.5:
        return f"A's move up after B's tap gave {wind('a_up_move')}"
    if wind("a_lift") != [0.0, 0.0]:
        return f"A's lift gave {wind('a_lift')}"
    return None


def test_a_joystick_and_a_button_at_once(open_page, served, crowd_export):
    page = open_page(served(crowd_export), touch=True)
    marks = two_fingers(page)
    frames = page.evaluate("__mx.frames")
    trace = page.evaluate("__mx.trace")
    # B is a pointer of its own: not 0, routed to the Button, not a new click of the left button
    [(_, b_down)] = [e for e in frames[marks["b_down"]]["events"] if e[0] == "pointer_down"]
    assert b_down["pointer_id"] != 0 and b_down["gui"][0] == WIDGETS["wave"]
    assert joystick_follows_finger_a(trace, marks) is None
    clicks = [i for i, e in enumerate(trace) if "gui:button:wave:click" in e["events"]]
    assert clicks == [marks["b_down"] + 1]
    mismatch = compare(replay_python(None, str(WEB_CROWD), frames, PATHS), trace)
    assert mismatch is None, str(mismatch)


def test_one_capture_for_all_pointers_is_a_mismatch(open_page, served, crowd_export, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(crowd_export, out)
    js = out / "manifoldx.js"
    text, n = re.subn(
        r"this\.captured\.(get|set|delete)\(ev\.pointer_id", r"this.captured.\1(0", js.read_text()
    )
    assert n == 4, f"GuiBridge no longer keys its capture as the patch expects ({n} matches)"
    js.write_text(text)
    page = open_page(served(out), touch=True)
    marks = two_fingers(page)
    frames = page.evaluate("__mx.frames")
    trace = page.evaluate("__mx.trace")
    assert joystick_follows_finger_a(trace, marks) is not None
    assert compare(replay_python(None, str(WEB_CROWD), frames, PATHS), trace) is not None


def test_the_widgets_have_their_roles(open_page, served, crowd_export):
    page = open_page(served(crowd_export))
    slider = page.locator('[role="slider"]')
    assert slider.get_attribute("aria-valuenow") == "6"
    assert slider.get_attribute("aria-valuemin") == "0"
    assert slider.get_attribute("aria-valuemax") == "12"
    assert slider.get_attribute("aria-label") == "gravity"
    page.mouse.move(*_on(page, "gravity", 0.2, 0.5))
    page.mouse.down()
    tick(page)
    page.mouse.move(*_on(page, "gravity", 0.9, 0.5))
    tick(page)
    page.mouse.up()
    tick(page)
    value = page.evaluate(f"__mx.app.engine.gui.widgets[{WIDGETS['gravity']}].value")
    assert value != 6.0
    assert slider.get_attribute("aria-valuenow") == page.evaluate(f"String({value})")
    switch = page.locator('[role="switch"]')
    assert switch.get_attribute("aria-checked") == "true"
    button = page.locator(f'[data-mx-widget="{WIDGETS["wave"]}"]')
    assert button.evaluate("(e) => e.tagName") == "BUTTON"
    assert button.text_content() == "Throw a wave"
    image = page.locator("img[data-mx-widget]")
    assert image.get_attribute("src").startswith("data:image/svg+xml")
    assert image.evaluate("(e) => e.complete && e.naturalWidth > 0")
    for el in page.locator("[data-mx-widget]").all():
        if el.evaluate("(e) => e.tagName") != "IMG":
            assert el.evaluate("(e) => getComputedStyle(e).touchAction") == "none"
