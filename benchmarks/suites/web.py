"""The exported page's work per frame in headless Chromium: the transpiled systems over the
entity store, the flush, and the instanced batches' sync. It reports and does not gate.

Needs the browser dependency group and a Chromium Playwright can launch
(`uv sync --group browser && uv run playwright install chromium`); without them the case
raises, which the harness reports as its error."""

import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from benchmarks.harness import case

# SwiftShader for WebGL on a machine without a GPU, as tests/web/browser does.
ARGS = ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"]
TICKS = 60

# Stops the page's loop as __mx.ready resolves, so each __mx.tick is one frame the bench runs.
MANUAL = """
Object.defineProperty(window, "__mx", {
  configurable: true,
  set(mx) {
    Object.defineProperty(window, "__mx", { value: mx, writable: true, configurable: true });
    mx.ready.then(() => mx.manual(true), () => {});
  },
});
"""

APP = """
import numpy as np

import manifoldx as mx
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.types import Vector3

N = {n}


class Velocity(Component):
    v: Vector3


engine = mx.Engine("web-crowd bench", width=320, height=200, max_entities=N + 16)
a = np.arange(N) * (2.0 * np.pi / N)
r = 5.0 + np.arange(N) * (20.0 / N)
engine.spawn(
    Mesh(mx.geometry.cube(0.2, 0.2, 0.2)),
    Material(mx.material.phong("#2e86c1")),
    Transform(pos=np.column_stack([np.cos(a) * r, np.sin(a * 7.0), np.sin(a) * r])),
    Velocity(v=np.column_stack([-np.sin(a), np.cos(a * 3.0) * 0.1, np.cos(a)])),
    n=N,
)
for i, x in enumerate((0.0, 30.0, 40.0)):
    engine.camera.position[i] = x
    engine.camera.target[i] = 0.0


def move(q: mx.Query[Transform, Velocity], dt: float):
    q[Transform].pos += q[Velocity].v * dt


engine.system(move)

if __name__ == "__main__":
    engine.cli()
"""


def _serve(folder: Path) -> tuple[subprocess.Popen, str]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [sys.executable, "-m", "http.server", "-b", "127.0.0.1", "-d", str(folder), str(port)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            return proc, f"http://127.0.0.1:{port}/"
        except OSError:
            time.sleep(0.05)
    proc.terminate()
    raise RuntimeError(f"http.server on port {port} did not start")


@case("web", "web-crowd", sizes=[1_000, 10_000], smoke=100)
def web_crowd(n):
    """n boxes in one batch, a query system moving all of them: page work per frame."""
    try:
        from playwright.sync_api import Error, sync_playwright
    except ImportError:
        raise RuntimeError("playwright not installed") from None
    from manifoldx.web.export import export_web

    with tempfile.TemporaryDirectory(prefix="mx-web-bench-") as tmp:
        app = Path(tmp) / "crowd.py"
        app.write_text(APP.format(n=n))
        out = Path(tmp) / "out"
        if export_web(None, str(app), out, []) != 0:
            raise RuntimeError("the bench app did not export")
        server, url = _serve(out)
        try:
            with sync_playwright() as p:
                try:
                    browser = p.chromium.launch(headless=True, args=ARGS)
                except Error as e:
                    raise RuntimeError(f"chromium not installed: {str(e).splitlines()[0]}")
                try:
                    page = browser.new_page(viewport={"width": 320, "height": 200})
                    page.add_init_script(MANUAL)
                    page.goto(url)
                    page.evaluate("__mx.ready.then(() => true)")
                    work = []
                    for _ in range(TICKS):
                        if page.evaluate("__mx.tick(1 / 60) !== null") is not True:
                            raise RuntimeError(f"the page stopped: {page.evaluate('__mx.error')}")
                        work.append(page.evaluate("__mx.frameMs.work"))
                    calls = page.evaluate("__mx.renderer.info.render.calls")
                finally:
                    browser.close()
        finally:
            server.terminate()
            server.wait(timeout=10)
    return {"ms": float(np.median(work)), "calls": calls}
