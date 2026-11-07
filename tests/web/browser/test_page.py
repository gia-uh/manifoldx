"""The export's page in headless Chromium: web_walk driven through Playwright's real keyboard
and mouse runs as Python runs it on the frames the page built, and the page's failure modes
(a file:// URL, an error in the app, a high-density screen, the progress line, a lost lock)
behave as plan 2 says."""

import json
import math
import re
from pathlib import Path

import pytest

from manifoldx.web.replay import compare, replay_python

from .conftest import APPS, BATCHES_OF, WEB_WALK, _export

PATHS = ["rig.pos", "rig.yaw", "rig.pitch", "rig.mode", "selected"]
LOOK = 0.004  # web_walk's radians per pixel
CENTRES = {  # web_walk's boxes the script picks, in this order
    "red crate": (-3.0, 0.5, -4.0),
    "green tower": (5.0, 1.5, -1.0),
    "blue crate": (2.0, 0.5, -5.0),
}
RAISES_LINE = next(
    i
    for i, line in enumerate((APPS / "raises.py").read_text().splitlines(), 1)
    if "counter.spot.x" in line
)


def tick(page) -> dict | None:
    return page.evaluate("__mx.tick(1 / 60)")


def ticks(page, n: int) -> None:
    for _ in range(n):
        tick(page)


def last_paths(page) -> dict:
    return page.evaluate("__mx.trace[__mx.trace.length - 1].paths")


def wait_lock(page, on: bool) -> None:
    page.wait_for_function(f"(document.pointerLockElement !== null) === {json.dumps(on)}")


def drive(page) -> dict:
    """The script, about 200 ticks: walk with W and D, a click that captures, mouse look, a
    pick at each of CENTRES, Tab, Space and Shift in fly mode (with Shift+W), the wheel, a
    release by exitPointerLock, a right drag, a resize to 640x480. Returns the frames of the
    ticks the checks name."""
    page.evaluate(
        """() => {
      __mx.paths = %s;
      window.__keys = [];
      addEventListener("keydown", (e) => __keys.push(e.key));
    }"""
        % json.dumps(PATHS)
    )
    marks = {}
    mouse = [400, 250]

    def move(x, y):
        mouse[:] = [x, y]
        page.mouse.move(x, y)

    move(400, 250)
    ticks(page, 5)
    page.keyboard.down("w")
    marks["w"] = tick(page)["frame"]
    ticks(page, 20)
    page.keyboard.down("d")
    ticks(page, 10)
    page.keyboard.up("w")
    page.keyboard.up("d")
    ticks(page, 3)

    page.mouse.click(*mouse)  # not captured: this click captures
    marks["capture"] = tick(page)["frame"]
    wait_lock(page, True)
    for x, y in [(440, 240), (380, 262), (420, 250), (400, 250)]:  # mouse look
        move(x, y)
        tick(page)
    ticks(page, 2)

    for name, centre in CENTRES.items():  # aim at the box, then click: a pick
        p = last_paths(page)
        eye = p["rig.pos"]
        v = [centre[i] - eye[i] for i in range(3)]
        yaw = math.atan2(v[0], -v[2])
        pitch = math.atan2(v[1], math.hypot(v[0], v[2]))
        dx = round((yaw - p["rig.yaw"]) / LOOK)
        dy = round(-(pitch - p["rig.pitch"]) / LOOK)
        move(mouse[0] + dx, mouse[1] + dy)
        tick(page)
        page.mouse.click(*mouse)
        marks[name] = tick(page)["entry"]["paths"]["selected"]
        ticks(page, 3)
    move(400, 250)
    tick(page)

    page.keyboard.press("Tab")  # fly
    marks["tab"] = tick(page)["frame"]
    page.keyboard.down("Space")
    ticks(page, 15)
    page.keyboard.up("Space")
    tick(page)
    page.keyboard.down("Shift")
    ticks(page, 5)
    page.keyboard.down("KeyW")  # with Shift held, the DOM's key is "W"
    marks["shift_w"] = tick(page)["frame"]
    ticks(page, 5)
    page.keyboard.up("KeyW")
    page.keyboard.up("Shift")
    tick(page)
    page.mouse.wheel(0, 120)
    marks["wheel"] = tick(page)["frame"]
    ticks(page, 3)

    page.evaluate("""() => new Promise((r) => {
      document.addEventListener("pointerlockchange", r, { once: true });
      document.exitPointerLock();
    })""")  # the browser ends the lock, as on Escape (Q7)
    marks["release"] = tick(page)["frame"]
    wait_lock(page, False)
    ticks(page, 3)

    page.mouse.down(button="right")  # a right drag looks without capturing
    tick(page)
    for x in (430, 460, 490, 470):
        move(x, 250)
        tick(page)
    page.mouse.up(button="right")
    tick(page)

    page.set_viewport_size({"width": 640, "height": 480})
    marks["resize"] = tick(page)["frame"]
    page.keyboard.press("Tab")  # walk again
    page.keyboard.down("s")
    ticks(page, 40)
    page.keyboard.up("s")
    page.keyboard.down("a")
    ticks(page, 30)
    page.keyboard.up("a")
    ticks(page, 25)
    return marks


def check(page, marks: dict) -> None:
    """What the trace, the frames and the page must show after drive(). Each assertion names
    what it checks, so a breaking test can say which one failed."""
    frames = page.evaluate("__mx.frames")
    browser = page.evaluate("__mx.trace")
    assert len(frames) >= 190, f"script length: {len(frames)} ticks"

    # the frames the page built hold what the script did
    assert "w" in marks["w"]["just"] and "w" in marks["w"]["keys"], f"frame after w: {marks['w']}"
    assert page.evaluate("__keys").count("W") == 1, "the DOM sent Shift+W as W"
    assert marks["shift_w"]["just"] == ["w"] and marks["shift_w"]["mods"] == ["Shift"], (
        f"frame after Shift+W: {marks['shift_w']}"
    )
    assert marks["tab"]["just"] == ["Tab"], f"frame after Tab: {marks['tab']}"
    assert marks["capture"]["clicks"] == [1], f"frame after the capturing click: {marks['capture']}"
    assert marks["wheel"]["wheel"][1] > 0, f"frame after the wheel: {marks['wheel']}"
    assert "Escape" in marks["release"]["just"], f"frame after the release: {marks['release']}"
    assert marks["resize"]["vp"] == [640, 480], f"frame after the resize: {marks['resize']}"

    python = replay_python(None, str(WEB_WALK), frames, PATHS)
    mismatch = compare(python, browser)
    assert mismatch is None, f"replay: {mismatch}"

    picks = [marks[name] for name in CENTRES]
    assert picks == list(CENTRES), f"picks: {picks}"
    assert len({e["paths"]["selected"] for e in browser} - {None}) >= 3
    assert any(x[0] == "material" for e in browser for x in e["log"])
    assert ["capture", True] in [x for e in browser for x in e["log"]]
    assert {e["paths"]["rig.mode"] for e in browser} == {"walk", "fly"}

    # the HTML shows what the trace shows
    assert page.locator("#mx-gui").inner_text().splitlines() == browser[-1]["gui"]
    # the picked box is drawn with the material the log names: by the batch whose instance
    # sits at its Transform row, not the store's Material column
    last = [x for e in browser for x in e["log"] if x[0] == "material"][-1]
    assert page.evaluate(f"__mx.scene.materialOf({last[1]})") == last[2]
    keys = page.evaluate(BATCHES_OF, last[1])
    drawn = len(keys) == 1 and page.evaluate(
        "([key, id]) => __mx.scene.batches.get(key).material === __mx.scene.materials.get(id)",
        [keys[0], last[2]],
    )
    assert drawn, f"material: entity {last[1]}'s batch {keys} does not draw material {last[2]}"
    # the three.js camera follows engine.camera
    assert page.evaluate("__mx.scene.camera.position.toArray()") == pytest.approx(
        browser[-1]["camera"]["position"], abs=1e-6
    )
    # every lock request came with a user activation (Q2), when the fake lock recorded them
    requests = page.evaluate("window.__lockRequests ?? null")
    assert requests is None or (requests and all(requests)), f"lock requests: {requests}"


def test_chromium_matches_python(open_page, served, web_walk_export):
    page = open_page(served(web_walk_export))
    marks = drive(page)
    check(page, marks)


def test_break_batches_ignore_materials(open_page, served, web_walk_export):
    """scene.js patched so every batch takes the white of no Material: the trace still matches
    and materialOf still reads the column, but the material check sees the wrong batch."""

    def patch(page):
        def route(r):
            body = r.fetch().text()
            patched = body.replace("batch(gids[i], mat?.[i] ? mids[i] : 0)", "batch(gids[i], 0)")
            assert patched != body, "scene.js's batch keys changed: update this patch"
            r.fulfill(body=patched, content_type="text/javascript")

        page.route(re.compile(r".*/scene\.js$"), route)

    page = open_page(served(web_walk_export), before=patch)
    marks = drive(page)
    with pytest.raises(AssertionError, match=r"^material: entity \d+'s batch"):
        check(page, marks)


def test_break_key_case(open_page, served, web_walk_export):
    """The page's FrameBuilder patched so key names keep their case: Shift+W's frame holds "W".
    The trace cannot show it (setFrame and Python's ScriptInput both lowercase a key again, and
    Python reads the page's own frames), so the frame check is what fails."""

    def patch(page):
        def route(r):
            body = r.fetch().text()
            patched = body.replace("k.length === 1 ? k.toLowerCase() : k", "k")
            assert patched != body, "input.js's key naming changed: update this patch"
            r.fulfill(body=patched, content_type="text/javascript")

        page.route(re.compile(r".*/input\.js$"), route)

    page = open_page(served(web_walk_export), before=patch)
    marks = drive(page)
    with pytest.raises(AssertionError, match=r"^frame after Shift\+W"):
        check(page, marks)


def test_file_url_says_to_serve(open_page, web_walk_export):
    page = open_page((Path(web_walk_export) / "index.html").as_uri(), ready=False)
    expect = page.get_by_text("serve this folder over HTTP", exact=False).first
    expect.wait_for(state="visible", timeout=5000)
    assert expect.is_visible()


@pytest.fixture(scope="module")
def raises_export(tmp_path_factory):
    return _export(APPS / "raises.py", tmp_path_factory.mktemp("raises"))


def test_error_names_the_python_line(open_page, served, raises_export):
    page = open_page(served(raises_export))
    ran = [tick(page) for _ in range(6)]
    assert all(r is not None for r in ran[:5]) and ran[5] is None
    error = page.evaluate("__mx.error")
    assert error.startswith(f"raises.py:{RAISES_LINE} in "), error
    assert page.locator("#mx-error").is_visible()
    assert page.locator("#mx-error").inner_text().startswith(f"raises.py:{RAISES_LINE} in ")
    assert tick(page) is None and page.evaluate("__mx.trace.length") == 5


def test_high_density_and_resize(open_page, served, web_walk_export):
    page = open_page(served(web_walk_export), width=800, height=500, scale=2)
    for w, h in [(800, 500), (640, 480)]:
        page.set_viewport_size({"width": w, "height": h})
        tick(page)
        got = page.evaluate("""() => ({
          w: __mx.app.engine.w, h: __mx.app.engine.h,
          vp: __mx.app.engine.input.viewport_size,
          cw: document.getElementById("mx-canvas").width,
          ch: document.getElementById("mx-canvas").height,
          aspect: __mx.scene.camera.aspect,
        })""")
        assert (got["w"], got["h"]) == (w, h)
        assert list(got["vp"]) == [w, h]
        assert (got["cw"], got["ch"]) == (2 * w, 2 * h)
        assert got["aspect"] == pytest.approx(w / h)


def test_progress_counts_the_scene(open_page, served, web_walk_export):
    page = open_page(served(web_walk_export))
    meta = page.evaluate("Number(document.querySelector('meta[name=scene-bytes]').content)")
    assert meta == (Path(web_walk_export) / "scene.glb").stat().st_size
    assert page.evaluate("__mx.loadStats.bytes") == meta
    assert page.locator("#mx-progress").is_hidden()


def test_lock_lost_releases(open_page, served, web_walk_export):
    page = open_page(served(web_walk_export))
    page.mouse.click(400, 250)
    tick(page)
    wait_lock(page, True)
    assert page.evaluate("__mx.app.engine.input.pointer_captured") is True
    page.evaluate("""() => new Promise((r) => {
      document.addEventListener("pointerlockchange", r, { once: true });
      document.exitPointerLock();
    })""")
    ran = tick(page)
    assert "Escape" in ran["frame"]["just"]
    assert page.evaluate("__mx.app.engine.input.pointer_captured") is False


@pytest.fixture(scope="module")
def moving_export(tmp_path_factory):
    return _export(APPS / "moving_panel.py", tmp_path_factory.mktemp("moving"))


def test_a_system_moves_a_panel_and_the_dom_follows(open_page, served, moving_export):
    """gia-uh/manifoldx#147: the offset, text and width a system writes each frame reach the
    panel's div, as the desktop paints them."""
    page = open_page(served(moving_export), width=800, height=500)
    box = """() => {
      const el = document.getElementById("mx-gui").children[0];
      const r = el.getBoundingClientRect();
      return { left: r.left, top: r.top, width: r.width, text: el.innerText };
    }"""
    first = page.evaluate(box)
    assert (first["left"], first["top"], first["text"]) == (10, 20, "frame 0")
    ticks(page, 10)
    assert page.evaluate(box) == {"left": 60, "top": 20, "width": 110, "text": "frame 10"}
    ticks(page, 1)
    assert page.evaluate(box) == {"left": 65, "top": 20, "width": 111, "text": "frame 11"}


@pytest.fixture(scope="module")
def camera_export(tmp_path_factory):
    return _export(APPS / "camera_writes.py", tmp_path_factory.mktemp("camera"))


def test_a_system_moves_the_camera_and_three_follows(open_page, served, camera_export):
    """gia-uh/manifoldx#151: position, target, fov, near and far a system writes reach the
    three.js camera the page renders with."""
    page = open_page(served(camera_export), width=800, height=500)
    seen = """() => {
      const c = __mx.scene.camera, e = __mx.app.engine.camera;
      const d = c.getWorldDirection(new c.position.constructor());
      return {
        position: c.position.toArray(), forward: d.toArray(),
        fov: c.fov, near: c.near, far: c.far,
        want: { position: [...e.position], forward: [...e.get_forward()] },
      };
    }"""
    ticks(page, 5)
    got = page.evaluate(seen)
    assert got["position"] == pytest.approx([0.5, 2.0, 7.0], abs=1e-6)
    assert got["position"] == pytest.approx(got["want"]["position"])
    assert got["forward"] == pytest.approx(got["want"]["forward"], abs=1e-6)
    assert got["forward"][2] == pytest.approx(-3 / (0.3**2 + 0.5**2 + 3.0**2) ** 0.5, abs=1e-6)
    assert (got["fov"], got["far"]) == (55, 1050.0)
    assert got["near"] == pytest.approx(0.15)
    ticks(page, 1)
    assert page.evaluate(seen)["fov"] == 56


@pytest.fixture(scope="module")
def handles_export(tmp_path_factory):
    return _export(APPS / "handles.py", tmp_path_factory.mktemp("handles"))


def test_a_system_writes_camera_up_and_three_follows(open_page, served, handles_export):
    """gia-uh/manifoldx#153: the up vector frame code writes each frame reaches the three.js
    camera the page renders with, as the Python replay computes it."""
    page = open_page(served(handles_export), width=320, height=200)
    page.evaluate('() => { __mx.paths = ["engine.camera.up"]; }')
    seen = []
    for _ in range(20):
        tick(page)
        seen.append(page.evaluate("__mx.scene.camera.up.toArray()"))
    python = replay_python(
        None, str(APPS / "handles.py"), page.evaluate("__mx.frames"), ["engine.camera.up"]
    )
    want = [e["paths"]["engine.camera.up"] for e in python]
    assert seen == [pytest.approx(w, abs=1e-7) for w in want]
    assert seen[4] != seen[14]  # it moves
