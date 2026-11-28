#!/usr/bin/env python3
"""Exercise the installed `--debug` metrics and HUD of `manifoldx run` under the Rust core.

Run this from a disposable cwd after a native wheel is installed, so
`manifoldx`, `manifoldx._core_native` and the console script come from
site-packages. It starts a private Xvfb display (the physical DISPLAY and
WAYLAND_DISPLAY are cleared) and runs the normal presented desktop loop,
not a diagnostic or offscreen path.

The fixture app has one synchronous callback with two controlled phases: fast
(no extra work) and slow (`time.sleep(0.002)`). The gate reads the raw JSON from
`/api/metrics`, runs `manifoldx metrics`, and captures the actual presented
window with `manifoldx look` in each phase. It checks:

- metrics exist with the existing fields, including the native backend split;
- the fast phase reports no callback over 1 ms, so the detector is sensitive;
- the slow phase reports the named callback over 1 ms in both metrics and the
  console, and its HUD border turns red while the fast border is green;
- the HUD is painted from the presented frame after post, is dark at its
  background and absent without `--debug`;
- `/api/metrics` and `manifoldx metrics` are absent without `--debug`.

Every subprocess is written to `invocations.json`; the raw metrics, PNGs,
hashes and installed identities go in the same run directory. The script does
not read or write the checkout. Prints `ok N  label` per check and a final
`checks: N`; exits 1 on the first failure.

Usage: native-metrics-hud.py [--manifoldx PATH] [--xvfb PATH] [--wheel PATH]
                             [--keep DIR]
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import numpy as np

APP = '''\
import time

import manifoldx as mx

engine = mx.Engine("e8 metrics", width=320, height=200, max_entities=8)
engine.state.setdefault("slow", False)
engine.background_color = (0.05, 0.2, 0.5)


@engine.system
def step(query: mx.Query[mx.Transform], dt: float):
    if engine.state["slow"]:
        time.sleep(0.002)


@engine.on("phase")
def phase(payload):
    """Set the callback phase: {"slow": bool}."""
    engine.state["slow"] = bool(payload["slow"])
    engine.log("phase slow" if engine.state["slow"] else "phase fast")


@engine.on("quit")
def quit_app(payload):
    """Close the window and the dev server."""
    engine.log("quitting")
    engine.quit()


engine.run()
'''

WIDTH, HEIGHT = 320, 200
SCREEN = (640, 400)  # the private Xvfb root, never the PNG
HUD_BACKGROUND = (12, 100)  # row, column: padding inside the HUD panel
HUD_BAND = (slice(8, 12), slice(40, 300))  # row, column crop of the panel's top border

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
    rows = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(
        height, 1 + width * 3
    )
    assert not rows[:, 0].any(), "unexpected PNG row filters"
    return width, height, rows[:, 1:].reshape(height, width, 3).copy()


def channel(port):
    return f"http://127.0.0.1:{port}"


def get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def get_json(url):
    return json.loads(get(url))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def record(argv, code, stdout, stderr):
    invocations.append({"argv": argv, "exit": code, "stdout": stdout, "stderr": stderr})


def installed_identity(console, wheel):
    """What the console runs, read from the same interpreter the script runs in."""
    import importlib.metadata

    def module_path(name):
        try:
            module = __import__(name, fromlist=["__file__"])
        except Exception as error:  # noqa: BLE001  recorded, not fatal
            return f"unavailable: {error}"
        return str(Path(module.__file__).resolve())

    try:
        version = importlib.metadata.version("manifold-gfx")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "console": str(console),
        "console_sha256": sha256(console.read_bytes()),
        "manifoldx_module": module_path("manifoldx"),
        "core_module": module_path("manifoldx._core_native"),
        "version": version,
        "wheel": None if wheel is None else str(wheel.resolve()),
        "wheel_sha256": None if wheel is None else sha256(wheel.read_bytes()),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifoldx", help="the installed console script (default: beside this interpreter)"
    )
    parser.add_argument("--xvfb", default="Xvfb", help="the Xvfb program (default: Xvfb on PATH)")
    parser.add_argument("--wheel", help="the exact wheel installed, to hash it")
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
    wheel = None if args.wheel is None else Path(args.wheel)

    root = (
        Path(args.keep)
        if args.keep
        else Path(tempfile.mkdtemp(prefix="manifoldx-e8-metrics.", dir="/var/tmp"))
    )
    root.mkdir(parents=True, exist_ok=True)
    app = root / "app.py"
    app.write_text(APP)

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
    debug_app = None
    plain_app = None
    try:
        deadline = time.monotonic() + 20
        while not Path(f"/tmp/.X11-unix/X{display}").exists():
            if server.poll() is not None or time.monotonic() > deadline:
                raise SystemExit(f"FAIL: Xvfb did not start on {screen}")
            time.sleep(0.1)
        check(True, f"private Xvfb on {screen}; physical DISPLAY/WAYLAND_DISPLAY cleared")

        def wait_manifest(process, timeout=60):
            manifest = root / ".manifoldx" / "server.json"
            deadline = time.monotonic() + timeout
            while not manifest.exists():
                if process.poll() is not None:
                    raise SystemExit(f"FAIL: the app exited: {process.stderr.read()}")
                if time.monotonic() > deadline:
                    raise SystemExit("FAIL: no .manifoldx/server.json")
                time.sleep(0.1)
            return manifest

        def start(extra):
            port = free_port()
            process = subprocess.Popen(
                [str(console), "run", "app.py", "--port", str(port), *extra],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            manifest = wait_manifest(process)
            url = json.loads(manifest.read_text())["url"]
            check(url == f"http://127.0.0.1:{port}", f"--port {port} served at {url}")
            return process, port

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

        def look(name):
            cli("look")
            data = (root / ".manifoldx" / "look.png").read_bytes()
            (root / f"{name}.png").write_bytes(data)
            width, height, rgb = png_rgb(data)
            return {
                "name": name,
                "sha256": sha256(data),
                "width": width,
                "height": height,
                "rgb": rgb,
            }

        def hud_state(shot):
            row, col = HUD_BACKGROUND
            background = shot["rgb"][row, col].astype(int)
            band = shot["rgb"][HUD_BAND].astype(int)
            red = int(((band[:, :, 0] > band[:, :, 2] + 40) & (band[:, :, 0] > band[:, :, 1] + 20)).sum())
            green = int(((band[:, :, 1] > band[:, :, 0] + 20) & (band[:, :, 1] > band[:, :, 2] + 20)).sum())
            return {
                "background": background.tolist(),
                "dark": int(background.sum()) < 150,
                "red": red,
                "green": green,
            }

        def phase(slow):
            payload = "true" if slow else "false"
            cli("emit", "phase", f'{{"slow": {payload}}}')
            cli("log", "--match", "phase slow" if slow else "phase fast", "--wait", "10")
            time.sleep(1.6)  # let a whole metrics second land after the change

        # ---- fast phase ----
        debug_app, port = start(["--debug"])
        print(f"  debug app pid {debug_app.pid}", flush=True)
        phase(False)
        fast = get_json(channel(port) + "/api/metrics")
        (root / "metrics-fast.json").write_text(json.dumps(fast, indent=1) + "\n")
        check(bool(fast), "/api/metrics answers under --debug")
        latest = fast[-1]
        check(
            {"time", "fps", "frame_ms", "worst_ms", "rss_mb", "gc", "split_ms"} <= set(latest),
            "the metrics entry keeps the existing schema",
        )
        split = latest["split_ms"]
        check(
            {"update", "render", "acquire", "submit", "present", "idle"} <= set(split),
            "the split carries the native backend's update/render/acquire/submit/present/idle",
        )
        check(
            latest["fps"] > 0 and latest["frame_ms"] > 0 and latest["worst_ms"] >= latest["frame_ms"],
            f"the measured frame sample is positive (fps {latest['fps']}, frame {latest['frame_ms']} ms)",
        )
        expected_ms = 1000.0 / latest["fps"]
        check(
            abs(latest["frame_ms"] - expected_ms) <= 0.2 * expected_ms,
            "fps and frame_ms describe the same measured presented interval",
        )
        check("step" in latest.get("systems_ms", {}), "the scheduler names the synchronous system")
        fast_step = latest["systems_ms"]["step"]
        fast_over = latest.get("over_budget_ms") or {}
        check(not fast_over, f"the fast phase reports no callback over 1 ms ({fast_over})")
        check(all(v < 1.0 for v in latest["systems_ms"].values()), "the fast callback stays under 1 ms")

        fast_hud = look("hud-fast")
        check(
            (fast_hud["width"], fast_hud["height"]) == (WIDTH, HEIGHT),
            f"look writes the presented window size {WIDTH}x{HEIGHT}",
        )
        fast_state = hud_state(fast_hud)
        check(fast_state["dark"], f"the HUD panel is painted over the scene ({fast_state['background']})")
        check(fast_state["green"] > 20, "the fast HUD border is green")
        check(fast_state["red"] < 20, "the fast HUD border is not red")

        fast_cli = cli("metrics", "-n", "3")
        check(
            "fps" in fast_cli.stdout and "frame" in fast_cli.stdout,
            "the installed `manifoldx metrics` prints the measured samples",
        )
        check("sync >1 ms" not in fast_cli.stdout, "the fast phase prints no over-budget callback")

        # ---- slow phase ----
        phase(True)
        slow = get_json(channel(port) + "/api/metrics")
        (root / "metrics-slow.json").write_text(json.dumps(slow, indent=1) + "\n")
        latest = slow[-1]
        over = latest.get("over_budget_ms") or {}
        check("step" in over, f"the slow phase names the callback over 1 ms ({over})")
        check(over["step"] >= 1.0, f"the measured callback is over 1 ms ({over.get('step')} ms)")
        check(
            latest["systems_ms"]["step"] > fast_step * 1.5,
            f"the slow callback costs more than the fast one ({latest['systems_ms']['step']} vs {fast_step} ms)",
        )

        slow_hud = look("hud-slow")
        slow_state = hud_state(slow_hud)
        check(slow_state["dark"], "the slow HUD panel is still painted over the scene")
        check(slow_state["red"] > 20, "the slow HUD border is red")
        check(slow_state["green"] < 20, "the slow HUD border is not green")
        check(
            slow_hud["sha256"] != fast_hud["sha256"],
            "the presented HUD frame changes between the controlled phases",
        )

        slow_cli = cli("metrics", "-n", "3")
        check("sync >1 ms: step" in slow_cli.stdout, "the console names the over-budget callback")

        # ---- fast again: the detector clears ----
        phase(False)
        clear = get_json(channel(port) + "/api/metrics")
        (root / "metrics-clear.json").write_text(json.dumps(clear, indent=1) + "\n")
        check(
            not (clear[-1].get("over_budget_ms") or {}),
            "returning to the fast phase clears the over-budget sample",
        )
        clear_hud = look("hud-clear")
        clear_state = hud_state(clear_hud)
        check(clear_state["green"] > 20 and clear_state["red"] < 20, "the HUD border returns to green")

        cli("emit", "quit", "null")
        code = debug_app.wait(timeout=60)
        check(code == 0, f"the debug app exits normally (code {code})")
        record([str(console), "run", "app.py", "--port", str(port), "--debug"], code, "", "")
        check(not (root / ".manifoldx" / "server.json").exists(), "the debug server closes")
        debug_app = None

        # ---- no --debug: metrics and HUD are absent ----
        plain_app, plain_port = start([])
        print(f"  plain app pid {plain_app.pid}", flush=True)
        refused = cli("metrics", "-n", "1", expect=1)
        check(
            "metrics need manifoldx run --debug" in refused.stderr,
            "the console refuses metrics without --debug",
        )
        try:
            get_json(channel(plain_port) + "/api/metrics")
        except urllib.error.HTTPError as error:
            check(error.code == 404, "the API refuses metrics without --debug (404)")
        else:
            raise AssertionError("/api/metrics answered without --debug")
        plain_hud = look("no-hud")
        plain_state = hud_state(plain_hud)
        check(
            not plain_state["dark"] and plain_hud["rgb"][HUD_BACKGROUND[0], HUD_BACKGROUND[1], 2]
            > plain_hud["rgb"][HUD_BACKGROUND[0], HUD_BACKGROUND[1], 0] + 80,
            "the scene reaches the top-left without a debug HUD",
        )
        check(
            plain_state["red"] < 20 and plain_state["green"] < 20,
            "the private desktop shows no HUD border without --debug",
        )
        cli("emit", "quit", "null")
        code = plain_app.wait(timeout=60)
        check(code == 0, f"the plain app exits normally (code {code})")
        record([str(console), "run", "app.py", "--port", str(plain_port)], code, "", "")
        plain_app = None

        (root / "invocations.json").write_text(json.dumps(invocations, indent=1) + "\n")
        check(len(invocations) >= 15, f"retained {len(invocations)} exact invocations")
        manifest = {
            "checks": checks,
            "app_sha256": sha256(APP.encode()),
            "script_sha256": sha256(Path(__file__).read_bytes()),
            "xvfb": xvfb,
            "screen": f"{SCREEN[0]}x{SCREEN[1]}x24",
            "identity": installed_identity(console, wheel),
            "pictures": {
                "hud-fast": fast_hud["sha256"],
                "hud-slow": slow_hud["sha256"],
                "hud-clear": clear_hud["sha256"],
                "no-hud": plain_hud["sha256"],
            },
            "source_overlays": [],
        }
        (root / "identities.json").write_text(json.dumps(manifest, indent=1) + "\n")
    finally:
        for process in (debug_app, plain_app):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
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
