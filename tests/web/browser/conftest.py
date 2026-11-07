"""The browser tests open an export in headless Chromium through Playwright. Without the
browser dependency group they skip with "playwright not installed", and without a Chromium
Playwright can launch with "chromium not installed".

Pointer lock (Q2 of plan 2): the session probes it once. Headless Chromium grants the lock,
but under it each synthetic mouse move arrives as two cancelling movements, so a probe passes
only when the lock is granted and a move under it nets the motion it made. When it fails, the
pages get FAKE_LOCK, an init script that keeps the lock in a field of the document and records
whether each request came with a user activation. The result is printed at the end of the run.
"""

import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
WEB_WALK = ROOT / "examples" / "web_walk.py"
WEB_CROWD = ROOT / "examples" / "web_crowd.py"
APPS = HERE.parent / "apps"
ARGS = ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"]

try:
    import playwright.sync_api  # noqa: F401

    HAVE_PLAYWRIGHT = True
except ImportError:
    HAVE_PLAYWRIGHT = False

FAKE_LOCK = """
(() => {
  let locked = null;
  window.__lockRequests = [];
  const change = () => setTimeout(() => document.dispatchEvent(new Event("pointerlockchange")), 0);
  Object.defineProperty(Document.prototype, "pointerLockElement", { configurable: true, get: () => locked });
  Element.prototype.requestPointerLock = function () {
    window.__lockRequests.push(navigator.userActivation.isActive);
    locked = this;
    change();
    return Promise.resolve();
  };
  Document.prototype.exitPointerLock = function () {
    if (locked === null) return;
    locked = null;
    change();
  };
})();
"""

# Stops the page's animation loop as __mx.ready resolves, before any animation frame runs the
# app, so a test's first __mx.tick is the app's first frame, as in Python.
MANUAL = """
Object.defineProperty(window, "__mx", {
  configurable: true,
  set(mx) {
    Object.defineProperty(window, "__mx", { value: mx, writable: true, configurable: true });
    mx.ready.then(() => mx.manual(true), () => {});
  },
});
"""

# The keys of the batches whose drawn instances sit at the entity's Transform row: the
# instance matrices three.js draws, not the page's own map from slots to instances.
BATCHES_OF = """(entity) => {
  const rows = __mx.app.engine.store.column("Transform"), o = entity * 10, keys = [];
  const m = __mx.scene.camera.matrix.clone();
  for (const [key, b] of __mx.scene.batches) {
    if (!b.visible) continue;
    for (let k = 0; k < b.count; k++) {
      const e = b.getMatrixAt(k, m).elements;
      if (Math.abs(e[12] - rows[o]) + Math.abs(e[13] - rows[o + 1]) + Math.abs(e[14] - rows[o + 2]) < 1e-5) keys.push(key);
    }
  }
  return keys;
}"""

_probe: dict = {}


def pytest_collection_modifyitems(config, items):
    if HAVE_PLAYWRIGHT:
        return
    skip = pytest.mark.skip(reason="playwright not installed")
    for item in items:
        if item.path.is_relative_to(HERE):
            item.add_marker(skip)


def pytest_terminal_summary(terminalreporter):
    if _probe:
        terminalreporter.write_line(f"pointer lock probe: {_probe['line']}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def served():
    """served(folder) -> the URL of a python -m http.server on a free port of 127.0.0.1; each
    server is stopped by its PID when the session ends."""
    procs = []

    def serve(folder) -> str:
        port = _free_port()
        cmd = [sys.executable, "-m", "http.server", "-b", "127.0.0.1", "-d", str(folder), str(port)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(proc)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
                return f"http://127.0.0.1:{port}/"
            except OSError:
                if proc.poll() is not None:
                    raise RuntimeError(f"http.server on port {port} exited with {proc.returncode}")
                time.sleep(0.05)
        raise RuntimeError(f"http.server on port {port} did not start")

    yield serve
    for proc in procs:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def chromium():
    """Headless Chromium, one per test module: Playwright's sync API leaves its asyncio loop
    running in this thread while it is open, and an engine adopts a running loop for its async
    startup handlers, so the loop must be gone before the tests outside this folder run."""
    from playwright.sync_api import Error, sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True, args=ARGS)
        except Error as e:
            pytest.skip(f"chromium not installed: {str(e).splitlines()[0]}")
        yield browser
        browser.close()


def _export(script: Path, out: Path) -> Path:
    from manifoldx.web.export import export_web

    assert export_web(None, str(script), out, []) == 0
    return out


@pytest.fixture(scope="session")
def web_walk_export(tmp_path_factory):
    return _export(WEB_WALK, tmp_path_factory.mktemp("web_walk"))


@pytest.fixture(scope="session")
def crowd_export(tmp_path_factory):
    return _export(WEB_CROWD, tmp_path_factory.mktemp("web_crowd"))


@pytest.fixture(scope="module")
def fake_lock(chromium, served, web_walk_export) -> bool:
    """Whether pages need FAKE_LOCK: the probe, once a session, clicks the canvas of a served
    export, waits for pointerlockchange (or 1 s), then moves the mouse 10 px under the lock."""
    if _probe:
        return _probe["fake"]
    page = chromium.new_page(viewport={"width": 800, "height": 500})
    try:
        page.goto(served(web_walk_export))
        page.evaluate("__mx.ready.then(() => __mx.manual(true))")
        page.evaluate("""() => {
          window.__probe = { change: new Promise((r) => document.addEventListener("pointerlockchange", r, { once: true })), moved: 0 };
          addEventListener("pointermove", (e) => { window.__probe.moved += e.movementX; });
          document.getElementById("mx-canvas").addEventListener("pointerdown", (e) => e.target.requestPointerLock(), { once: true });
        }""")
        page.mouse.click(400, 250)
        granted = page.evaluate("""() => Promise.race([__probe.change, new Promise((r) => setTimeout(r, 1000))])
          .then(() => document.pointerLockElement?.id === "mx-canvas")""")
        moved = None
        if granted:
            page.evaluate("__probe.moved = 0")
            page.mouse.move(410, 250)
            moved = page.evaluate(
                "new Promise((r) => requestAnimationFrame(() => r(__probe.moved)))"
            )
    finally:
        page.close()
    works = bool(granted) and moved == 10
    _probe["line"] = f"granted={bool(granted)}, a 10 px move under it nets {moved}; " + (
        "the real lock is used" if works else "the pages use the fake lock (FAKE_LOCK)"
    )
    _probe["fake"] = not works
    print(f"\npointer lock probe: {_probe['line']}")
    return not works


@pytest.fixture
def open_page(chromium, fake_lock):
    """open_page(url, width=800, height=500, scale=1) -> a page of its own context, past
    __mx.ready unless ready=False, its loop stopped (MANUAL) so the test ticks it, with the fake
    lock when the probe asked for it. before(page) runs before the page loads. touch=True makes
    a context with a touch screen, for touch(page, points)."""
    contexts = []

    def open_(url, width=800, height=500, scale=1, ready=True, before=None, touch=False):
        ctx = chromium.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=scale,
            has_touch=touch,
        )
        contexts.append(ctx)
        ctx.add_init_script(MANUAL)
        if fake_lock:
            ctx.add_init_script(FAKE_LOCK)
        page = ctx.new_page()
        if before is not None:
            before(page)
        page.goto(url)
        if ready:
            page.evaluate("__mx.ready.then(() => true)")
        return page

    yield open_
    for ctx in contexts:
        ctx.close()


@pytest.fixture
def run_web():
    """run_web(app) -> the URL of `manifoldx run --web` of the script app (a path in a tmp_path
    folder), run in its folder on a free port. The URL is read from .manifoldx/server.json; the
    server's stderr goes to .manifoldx/run.err beside it. Each server is stopped by its PID."""
    procs = []

    def start(app: Path) -> str:
        cli = Path(sys.executable).parent / "manifoldx"
        folder = app.parent / ".manifoldx"
        folder.mkdir(exist_ok=True)
        err = (folder / "run.err").open("w")
        proc = subprocess.Popen(
            [str(cli), "run", "--web", "--port", "0", app.name],
            cwd=app.parent,
            stdout=subprocess.DEVNULL,
            stderr=err,
        )
        procs.append(proc)
        server = folder / "server.json"
        for _ in range(600):
            if server.exists():
                return json.loads(server.read_text())["url"] + "/"
            if proc.poll() is not None:
                raise RuntimeError(
                    f"run --web exited with {proc.returncode}:\n{(folder / 'run.err').read_text()}"
                )
            time.sleep(0.05)
        raise RuntimeError(f"run --web did not start:\n{(folder / 'run.err').read_text()}")

    yield start
    for proc in procs:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


_touches: dict = {}  # page -> [its CDP session, the touch points down]


def touch(page, points: dict) -> None:
    """Put the touch screen in the state points says, {finger: (x, y)} in CSS pixels, through
    Input.dispatchTouchEvent: a finger new to points goes down, one gone from it lifts, and the
    others move to their place. Chromium sends one pointer event per finger that changed. The
    page needs a context made with touch=True."""
    if page not in _touches:
        _touches[page] = [page.context.new_cdp_session(page), {}]
    cdp, before = _touches[page]
    # Chromium's touchEnd lifts the fingers it lists; touchStart and touchMove list them all.
    lifted = {f: xy for f, xy in before.items() if f not in points}
    if lifted:
        kind, sent = "touchEnd", lifted
    else:
        kind, sent = ("touchStart" if points.keys() - before.keys() else "touchMove"), points
    active = [{"x": x, "y": y, "id": finger} for finger, (x, y) in sent.items()]
    cdp.send("Input.dispatchTouchEvent", {"type": kind, "touchPoints": active})
    _touches[page][1] = dict(points)
    # Chromium hands pointermove to the page at its next animation frame.
    page.evaluate("new Promise((r) => requestAnimationFrame(() => r()))")
