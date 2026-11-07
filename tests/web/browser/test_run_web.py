"""manifoldx run --web end to end: the real CLI serves tests/web/apps/dev_orbit.py, headless
Chromium opens the printed URL, and each test edits a fresh copy of the app. A save swaps the
new version into the open tab with engine.state kept and no page reload; a broken edit or a
frame error shows the overlay with the Python line; the agent commands answer from the tab."""

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import APPS

ORBIT = APPS / "dev_orbit.py"
YAW = "__mx.app.engine.state.get('cam').get('yaw')"
STATUS = "document.getElementById('mx-dev').textContent"
PANEL = "document.getElementById('mx-error')"

# The tab's answer to reload_state with an empty engine.state: what a tab that lost its state
# would send. Every reply goes through fetch, so the init script rewrites the reply's state.
EMPTY_RELOAD_STATE = """
(() => {
  const real = window.fetch;
  window.fetch = (url, init) => {
    if (String(url).includes("api/web/reply/") && init?.body && typeof init.body === "string") {
      const body = JSON.parse(init.body);
      if (body.state) init = { ...init, body: JSON.stringify({ ...body, state: { __dict: [] } }) };
    }
    return real(url, init);
  };
})();
"""


def line_of(app: Path, text: str) -> int:
    return next(i for i, line in enumerate(app.read_text().splitlines(), 1) if text in line)


@pytest.fixture
def app(tmp_path) -> Path:
    folder = tmp_path / "orbit"
    folder.mkdir()
    return Path(shutil.copy(ORBIT, folder / ORBIT.name))


def tick(page, n: int = 1):
    for _ in range(n):
        page.evaluate("__mx.tick(1 / 60)")


def hold(page, key: str, ticks: int):
    page.keyboard.down(key)
    tick(page, ticks)
    page.keyboard.up(key)
    tick(page)


def edit(app: Path, old: str, new: str):
    text = app.read_text()
    assert old in text
    app.write_text(text.replace(old, new, 1))


def wait_version(page, n: int):
    page.wait_for_function(f"__mx.version === {n}", timeout=15000)


def manifoldx(app: Path, *args: str) -> str:
    done = subprocess.run(
        [str(Path(sys.executable).parent / "manifoldx"), *args],
        cwd=app.parent,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def save_keeps_state(page, app: Path):
    hold(page, "d", ticks=30)  # turn the camera
    yaw = page.evaluate(YAW)
    assert yaw > 0.4
    scene = page.evaluate("__mx.scene.scene.uuid")
    page.evaluate("window.__marker = 1")
    page.keyboard.down("w")  # a key held across the save
    edit(app, "SPEED = 1.0", "SPEED = 2.0")
    wait_version(page, 2)
    assert page.evaluate(YAW) == pytest.approx(yaw), "yaw"
    tick(page)
    assert page.evaluate("__mx.scene.scene.uuid") == scene  # code-only: scene kept
    assert page.evaluate("__mx.swaps.map((s) => s.scene)") == ["kept"]
    assert "SPEED = 2.0" in page.locator("#mx-gui").inner_text()
    assert page.evaluate("__mx.app.engine.input.is_pressed('w')")
    assert page.evaluate("window.__marker") == 1  # set before the save: no page reload
    assert page.evaluate(STATUS) == "v2 · live"


def test_a_save_swaps_and_keeps_the_state(run_web, open_page, app):
    save_keeps_state(open_page(run_web(app)), app)


def test_break_reload_state_empty(run_web, open_page, app):
    url = run_web(app)
    page = open_page(url, before=lambda p: p.add_init_script(EMPTY_RELOAD_STATE))
    with pytest.raises(AssertionError, match="yaw"):
        save_keeps_state(page, app)


def box_colour(page) -> str:
    return page.evaluate("""() => {
      let found = null;
      __mx.scene.scene.traverse((o) => { if (o.isMesh && !found) found = o; });
      return found.material.color.getHexString();
    }""")


def test_a_colour_edit_rebuilds_the_scene(run_web, open_page, app):
    page = open_page(run_web(app))
    tick(page)
    before = page.evaluate("__mx.scene.scene.uuid")
    assert box_colour(page) == "4080c0"
    edit(app, 'BOX = "#4080c0"', 'BOX = "#c04020"')
    wait_version(page, 2)
    tick(page)
    assert page.evaluate("__mx.scene.scene.uuid") != before
    assert page.evaluate("__mx.swaps.map((s) => s.scene)") == ["new"]
    assert box_colour(page) == "c04020"


def runtime_material_reset(page, app: Path):
    cube = page.evaluate("__mx.app.modules.__main__.CUBE.index")
    original = page.evaluate(f"__mx.scene.materialOf({cube})")
    lit = page.evaluate("__mx.app.modules.__main__.LIT.id")
    assert lit != original
    page.keyboard.press("h")
    tick(page)
    assert page.evaluate(f"__mx.scene.materialOf({cube})") == lit
    edit(app, "SPEED = 1.0", "SPEED = 2.0")
    wait_version(page, 2)
    tick(page)
    assert page.evaluate("__mx.swaps.map((s) => s.scene)") == ["kept"]
    assert page.evaluate(f"__mx.scene.materialOf({cube})") == original, "material"
    # The meshes themselves, not only the page's map: each holds the glTF's material again.
    assert page.evaluate(
        f"__mx.scene.entities.get({cube}).every((m) => m.material !== __mx.scene.materials.get({lit}))"
    ), "meshes"


def test_a_kept_scene_drops_runtime_materials(run_web, open_page, app):
    runtime_material_reset(open_page(run_web(app)), app)


def test_break_reset_materials_noop(run_web, open_page, app):
    page = open_page(run_web(app))
    page.evaluate("__mx.scene.resetMaterials = () => {}")
    with pytest.raises(AssertionError, match="material"):
        runtime_material_reset(page, app)


def test_broken_edit_keeps_running_with_the_line(run_web, open_page, app):
    page = open_page(run_web(app))
    tick(page)
    n = line_of(app, "SPEED = 1.0")
    edit(app, "SPEED = 1.0", "SPEED = (1.0")
    page.wait_for_function(f"{STATUS}.includes('reload failed')", timeout=15000)
    assert page.evaluate(STATUS) == "v1 · reload failed, running v1"
    panel = page.locator("#mx-error")
    assert panel.is_visible()
    text = panel.inner_text()
    assert text.startswith(f"dev_orbit.py:{n}: SyntaxError: "), text
    assert text.endswith("Fix it and save: the tab keeps running v1."), text
    hold(page, "d", ticks=10)  # still running v1
    assert page.evaluate(YAW) > 0.1
    edit(app, "SPEED = (1.0", "SPEED = 1.0")
    wait_version(page, 2)
    assert page.evaluate(STATUS) == "v2 · live"
    assert not panel.is_visible()


def test_frame_error_pauses_and_a_save_resumes(run_web, open_page, app):
    url = run_web(app)
    page = open_page(url)
    tick(page)
    old = 'yaw: float = engine.state.cam["yaw"]\n    engine.state.cam["yaw"] = yaw + turn'
    edit(app, old, old.replace('cam["yaw"]\n', 'cam["yawn"]\n', 1))
    n = line_of(app, 'cam["yawn"]')
    wait_version(page, 2)
    assert page.evaluate("__mx.tick(1 / 60)") is None  # the frame threw: paused
    assert page.evaluate(STATUS) == "v2 · paused until the next edit"
    text = page.locator("#mx-error").inner_text()
    assert text.startswith(f"dev_orbit.py:{n} in orbit: "), text
    assert page.evaluate("__mx.tick(1 / 60)") is None  # stays paused
    log = manifoldx(app, "log", "--match", "paused until the next edit", "--wait", "5")
    assert f"paused until the next edit: dev_orbit.py:{n} in orbit" in log
    edit(app, 'cam["yawn"]', 'cam["yaw"]')
    wait_version(page, 3)
    assert page.evaluate("__mx.tick(1 / 60)") is not None  # resumed
    assert page.evaluate(STATUS) == "v3 · live"
    assert not page.locator("#mx-error").is_visible()


def pixel(img, x: int, y: int) -> tuple[int, int, int]:
    return img.getpixel((x, y))[:3]


def near(a, b, tol: int = 8) -> bool:
    return all(abs(p - q) <= tol for p, q in zip(a, b))


def test_agent_commands(run_web, open_page, app):
    from PIL import Image

    page = open_page(run_web(app))
    hold(page, "d", ticks=20)
    yaw = page.evaluate(YAW)
    assert json.loads(manifoldx(app, "state", "cam"))["yaw"] == pytest.approx(yaw)

    manifoldx(app, "set", '{"cam": {"yaw": 0}}')
    tick(page)
    assert page.evaluate(YAW) == 0
    assert page.evaluate("__mx.app.engine.camera.position[0]") == pytest.approx(0, abs=1e-6)

    manifoldx(app, "emit", "bump", '{"by": 10}')
    assert page.evaluate(YAW) == 0  # an event runs at the start of the next frame
    tick(page)
    assert page.evaluate(YAW) == pytest.approx(0.17453292519943295)

    events = manifoldx(app, "events")
    assert 'bump\n    Turn the camera: {"by": degrees}.' in events

    path = app.parent / manifoldx(app, "look").strip()
    img = Image.open(path).convert("RGB")
    size = page.evaluate(
        "[document.getElementById('mx-canvas').width, document.getElementById('mx-canvas').height]"
    )
    assert list(img.size) == size == [800, 500]
    box = page.locator("#mx-gui > div").first.bounding_box()
    cx, cy = int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2)
    assert near(pixel(img, cx, cy), (0x20, 0x28, 0x30)), pixel(img, cx, cy)
    # The box at the centre, as the tab shows it: the page's own screenshot of the same frame.
    shot = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
    centre = pixel(img, 400, 250)
    assert near(centre, pixel(shot, 400, 250)), (centre, pixel(shot, 400, 250))
    assert near(pixel(img, 5, 5), pixel(shot, 5, 5))
    assert not near(centre, pixel(img, 5, 5), 24)  # the box, not the background

    edit(app, "SPEED = 1.0", "SPEED = 2.0")
    wait_version(page, 2)
    out = manifoldx(app, "log", "--match", "tab: v2", "--wait", "5")
    assert "tab: v2 in " in out and "(scene kept)" in out


def test_lock_survives_a_save(run_web, open_page, app):
    page = open_page(run_web(app))
    tick(page)
    page.mouse.click(400, 250)
    tick(page)  # the click captures
    page.wait_for_function("document.pointerLockElement !== null")
    tick(page)
    assert page.evaluate("__mx.app.engine.input.pointer_captured")
    edit(app, "SPEED = 1.0", "SPEED = 2.0")
    wait_version(page, 2)
    tick(page)
    assert page.evaluate("__mx.app.engine.input.pointer_captured")
    assert page.evaluate("document.pointerLockElement !== null")


def test_a_reopened_tab_keeps_the_camera(run_web, open_page, app):
    page = open_page(run_web(app))
    hold(page, "d", ticks=30)
    yaw = page.evaluate(YAW)
    assert yaw > 0.4
    page.reload()
    page.evaluate("__mx.ready.then(() => true)")
    wait_version(page, 2)  # the beacon's state differs from v1's: a rebuild for the new tab
    assert page.evaluate(YAW) == pytest.approx(yaw)
