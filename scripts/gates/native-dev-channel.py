#!/usr/bin/env python3
"""Exercise the installed dev channel of `manifoldx run` under the Rust core.

Run this from a disposable cwd with `-I`, after a native wheel is installed, so
`manifoldx` and `manifoldx._core_native` come from site-packages. It starts a
private Xvfb display (the physical DISPLAY and WAYLAND_DISPLAY are cleared), runs
an app through the installed console script with `--port`, and drives the real
`state`, `set`, `events`, `emit`, `look` and `log` subprocesses against it.

Checks: a state read, a merge patch, the patched value seen by an event handler,
an event's registered handler and its log line, an unknown event rejected with
404, `look` writing the actual presented frame as a PNG (window dimensions,
post-processed scene, GUI painted over it), a second identical look as the
no-change control, a look after a controlled colour change, four concurrent
looks, the normal shutdown after a `quit` event, a look refused because no
frame is presented any more, and the port released with
`.manifoldx/server.json` removed. Every invocation is written to
`invocations.json`.

Usage: native-dev-channel.py [--manifoldx PATH] [--xvfb PATH] [--keep DIR]
Prints `ok N  label` per check and a final `checks: N`; exits 1 on the first
failure. The run directory (app, log and server state) is printed and kept.
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

APP = '''\
from manifoldx import Engine
from manifoldx.gui import Panel, Text
from manifoldx.post import Grade

engine = Engine("e8 look", width=320, height=200, max_entities=8)
engine.state.setdefault("color", [0.85, 0.1, 0.1])
engine.background_color = tuple(engine.state["color"])
# A visible post pass, drawn before the GUI: the corners darken.
engine.set_post(grade=Grade(vignette=0.8), fxaa=True)
# A solid panel in a colour the scene never has: GUI after post.
engine.gui.append(Panel(
    [Text("look")],
    anchor="top-left",
    offset=(24, 24),
    style_overrides={"width": 96, "height": 64, "padding": 0, "bg": "#1e44d8", "radius": 0},
))


@engine.on("paint")
def paint(payload):
    """Set the clear colour the frame presents: {"rgb": [r, g, b]}."""
    color = [float(value) for value in payload["rgb"]]
    engine.state["color"] = color
    engine.background_color = tuple(color)
    engine.log("paint %.2f %.2f %.2f" % tuple(color))


@engine.on("quit")
def quit_app(payload):
    """Close the window and the dev server."""
    engine.log("quitting")
    engine.quit()


engine.run()
'''

WIDTH, HEIGHT = 320, 200
CENTER = (100, 160)  # row, column, outside the panel
CORNER = (5, 300)
PANEL = (78, 105)  # row, column, inside the solid panel below its label
SCREEN = (640, 400)  # the private Xvfb root, never the PNG

checks = 0
invocations = []


def check(condition, label):
    global checks
    if not condition:
        raise AssertionError(label)
    checks += 1
    print(f"ok {checks}  {label}", flush=True)


def free_port():
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
    finally:
        probe.close()


def free_display():
    for number in range(90, 130):
        if not Path(f"/tmp/.X11-unix/X{number}").exists():
            return number
    raise RuntimeError("no free X display number between :90 and :129")


def wait_port(port, timeout, closed):
    """Whether 127.0.0.1:port can (closed) or cannot (held) be bound."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = socket.socket()
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
            bound = True
        except OSError:
            bound = False
        finally:
            probe.close()
        if bound == closed:
            return True
        time.sleep(0.1)
    return False


def png_rgb(data):
    """(h, w, 3) uint8 pixels of a filter-0, 8-bit RGB PNG, without a decoder dependency."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("not a PNG")
    pos, chunks = 8, {}
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        chunks[kind] = chunks.get(kind, b"") + data[pos + 8 : pos + 8 + length]
        pos += 12 + length
    width, height, depth, color = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (depth, color) == (8, 2), f"unexpected PNG layout {depth}/{color}"
    rows = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(height, 1 + width * 3)
    assert not rows[:, 0].any(), "unexpected PNG row filters"
    return width, height, rows[:, 1:].reshape(height, width, 3).copy()


def channel(port):
    return f"http://127.0.0.1:{port}"


def get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def record(argv, code, stdout, stderr):
    invocations.append({"argv": argv, "exit": code, "stdout": stdout, "stderr": stderr})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifoldx", help="the installed console script (default: beside this interpreter)"
    )
    parser.add_argument("--xvfb", default="Xvfb", help="the Xvfb program (default: Xvfb on PATH)")
    parser.add_argument(
        "--keep", help="the run directory to use (default: a new one under /var/tmp)"
    )
    args = parser.parse_args()

    console = (
        Path(args.manifoldx) if args.manifoldx else Path(sys.executable).with_name("manifoldx")
    )
    if not console.is_file():
        found = shutil.which("manifoldx")
        if found is None:
            raise SystemExit(f"FAIL: no manifoldx console script: {console}")
        console = Path(found)
    console = console.resolve()
    xvfb = shutil.which(args.xvfb)
    if xvfb is None:
        raise SystemExit(f"FAIL: no {args.xvfb} on PATH")

    root = (
        Path(args.keep)
        if args.keep
        else Path(tempfile.mkdtemp(prefix="manifoldx-dev-channel.", dir="/var/tmp"))
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "app.py").write_text(APP)

    # The physical display is never used; the private one is chosen here.
    clean = {
        key: value
        for key, value in os.environ.items()
        if key not in ("DISPLAY", "WAYLAND_DISPLAY", "PYTHONPATH", "PYTHONHOME")
    }
    display = free_display()
    screen = f":{display}"
    server = subprocess.Popen(
        [xvfb, screen, "-screen", "0", f"{SCREEN[0]}x{SCREEN[1]}x24", "-nolisten", "tcp"],
        cwd=root,
        env=clean,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    env = clean | {"DISPLAY": screen, "MANIFOLDX_CORE": "rust"}
    port = free_port()
    app = None
    try:
        deadline = time.monotonic() + 20
        while not Path(f"/tmp/.X11-unix/X{display}").exists():
            if server.poll() is not None or time.monotonic() > deadline:
                raise SystemExit(f"FAIL: Xvfb did not start on {screen}")
            time.sleep(0.1)
        check(True, f"private Xvfb on {screen}; physical DISPLAY/WAYLAND_DISPLAY cleared")

        app = subprocess.Popen(
            [str(console), "run", "app.py", "--port", str(port)],
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        manifest = root / ".manifoldx" / "server.json"
        deadline = time.monotonic() + 60
        while not manifest.exists():
            if app.poll() is not None:
                raise SystemExit(f"FAIL: the app exited: {app.stderr.read()}")
            if time.monotonic() > deadline:
                raise SystemExit("FAIL: no .manifoldx/server.json after 60 s")
            time.sleep(0.1)
        url = json.loads(manifest.read_text())["url"]
        check(url == f"http://127.0.0.1:{port}", f"--port {port} served at {url}")
        check(
            wait_port(port, 5, closed=False),
            "the app holds its port while it runs (release control)",
        )

        def cli(*argv, expect=0):
            done = subprocess.run(
                [str(console), *argv],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
            )
            record([str(console), *argv], done.returncode, done.stdout, done.stderr)
            if done.returncode != expect:
                raise AssertionError(
                    f"manifoldx {' '.join(argv)} exited {done.returncode}, expected {expect}\n"
                    f"stdout:\n{done.stdout}\nstderr:\n{done.stderr}"
                )
            return done

        def look(name, expect=0):
            done = cli("look", expect=expect)
            if expect:
                return None
            data = (root / ".manifoldx" / "look.png").read_bytes()
            (root / f"{name}.png").write_bytes(data)
            width, height, rgb = png_rgb(data)
            return {
                "name": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "width": width,
                "height": height,
                "rgb": rgb,
            }

        state = json.loads(cli("state").stdout)
        check(state["color"] == [0.85, 0.1, 0.1], "state reads engine.state")

        patched = json.loads(cli("set", '{"color": [0.85, 0.1, 0.1], "mark": 1}').stdout)
        check(patched["mark"] == 1, "set applies a merge patch")
        check(json.loads(cli("state", "mark").stdout) == 1, "state rereads the patched value")

        events = cli("events").stdout
        check(
            "paint\n    Set the clear colour" in events and "quit\n    Close the window" in events,
            "events lists the app's handlers and docstrings",
        )

        fired = json.loads(cli("emit", "paint", '{"rgb": [0.85, 0.1, 0.1]}').stdout)
        check(fired == {"emitted": "paint", "handlers": 1}, "emit reaches the registered handler")
        log = cli("log", "--match", "paint 0.85 0.10 0.10", "--wait", "10").stdout
        check(log.endswith("paint 0.85 0.10 0.10\n"), "the handler's log line is observed")

        rejected = cli("emit", "nothing", "null", expect=1)
        check(
            "404" in rejected.stderr and "paint" in rejected.stderr,
            "an unknown event is rejected with 404",
        )

        red = look("look-red")
        check(
            (red["width"], red["height"]) == (WIDTH, HEIGHT),
            f"look writes the presented window size {WIDTH}x{HEIGHT}",
        )
        check(
            (red["width"], red["height"]) != SCREEN,
            f"look is the window, not the {SCREEN[0]}x{SCREEN[1]} private desktop",
        )
        row, col = CENTER
        center = red["rgb"][row, col].astype(int)
        check(
            center[0] > center[1] + 40 and center[0] > center[2] + 40,
            f"the presented scene carries the controlled red state ({center.tolist()})",
        )
        row, col = CORNER
        corner = red["rgb"][row, col].astype(int)
        check(
            int(corner.sum()) + 40 < int(center.sum()),
            f"the post vignette darkens the corner ({corner.tolist()} vs {center.tolist()})",
        )
        row, col = PANEL
        panel = red["rgb"][row, col].astype(int)
        check(
            panel[2] > panel[0] + 40 and panel[2] > panel[1] + 20,
            f"the GUI panel is painted over the post-processed scene ({panel.tolist()})",
        )
        check(
            not (panel[0] > panel[2] + 40),
            "the GUI pixel is not the scene colour, so look is not a pre-GUI capture",
        )

        steady = look("look-steady")
        check(
            np.abs(steady["rgb"].astype(int) - red["rgb"].astype(int)).max() <= 1,
            "two looks with no visible change show the same frame (sensitivity control)",
        )

        cli("emit", "paint", '{"rgb": [0.1, 0.85, 0.1]}')
        cli("log", "--match", "paint 0.10 0.85 0.10", "--wait", "10")
        green = look("look-green")
        row, col = CENTER
        center = green["rgb"][row, col].astype(int)
        check(
            center[1] > center[0] + 40 and center[1] > center[2] + 40,
            f"look follows the next presented state ({center.tolist()})",
        )
        check(
            green["sha256"] != red["sha256"],
            "the changed presented frame writes different PNG bytes",
        )
        row, col = CORNER
        corner = green["rgb"][row, col].astype(int)
        check(
            int(corner.sum()) + 40 < int(center.sum()),
            "post and GUI still compose after the change",
        )
        row, col = PANEL
        panel = green["rgb"][row, col].astype(int)
        check(panel[2] > panel[0] + 40, "the GUI keeps its colour across the scene change")

        cli("emit", "paint", '{"rgb": [0.1, 0.1, 0.85]}')
        cli("log", "--match", "paint 0.10 0.10 0.85", "--wait", "10")

        def burst(_):
            return png_rgb(get(channel(port) + "/api/look"))

        with ThreadPoolExecutor(max_workers=4) as pool:
            bursts = list(pool.map(burst, range(4)))
        invocations.append(
            {"request": f"GET {channel(port)}/api/look", "count": 4, "concurrency": 4}
        )
        check(
            all((w, h) == (WIDTH, HEIGHT) for w, h, _ in bursts),
            "four concurrent looks all answer with the presented size",
        )
        row, col = CENTER
        check(
            all(rgb[row, col, 2] > rgb[row, col, 0] + 40 for _, _, rgb in bursts),
            "all four concurrent looks see the same new blue state",
        )
        (root / "look-blue.png").write_bytes(get(channel(port) + "/api/look"))

        (root / ".manifoldx" / "look.png").unlink()
        cli("emit", "quit", "null")
        code = app.wait(timeout=60)
        check(code == 0, f"the app exits normally after the quit event (code {code})")
        record([str(console), "run", "app.py", "--port", str(port)], code, "", "")
        refused = cli("look", expect=1)
        check(
            ("no app runs" in refused.stderr or "does not answer" in refused.stderr)
            and not (root / ".manifoldx" / "look.png").exists(),
            "a look with no presented frame fails and writes no PNG (negative control)",
        )
        check(not manifest.exists(), ".manifoldx/server.json is removed on close")
        check(wait_port(port, 10, closed=True), "the port is released on close")
        log_text = (root / ".manifoldx" / "log.txt").read_text()
        check(
            "paint 0.10 0.10 0.85" in log_text and "quitting" in log_text and "closed" in log_text,
            "the log keeps the colour changes, the quit and the close",
        )
        stderr = app.stderr.read()
        check("dev page and API at" in stderr, "the runner announced the dev page")
        (root / "invocations.json").write_text(json.dumps(invocations, indent=1) + "\n")
        check(len(invocations) >= 12, f"retained {len(invocations)} exact invocations")
    finally:
        if app is not None and app.poll() is None:
            app.kill()
            app.wait()
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        print(f"run directory: {root}", flush=True)

    print(f"checks: {checks}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
