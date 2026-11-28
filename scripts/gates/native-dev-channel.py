#!/usr/bin/env python3
"""Exercise the installed dev channel of `manifoldx run` under the Rust core.

Run this from a disposable cwd with `-I`, after a native wheel is installed, so
`manifoldx` and `manifoldx._core_native` come from site-packages. It starts a
private Xvfb display (the physical DISPLAY and WAYLAND_DISPLAY are cleared), runs
an app through the installed console script with `--port`, and drives the real
`state`, `set`, `events`, `emit` and `log` subprocesses against it.

Checks: a state read, a merge patch, the patched value seen by an event handler,
an event's registered handler and its log line, an unknown event rejected with
404, `look` answering 501 without writing a PNG, the normal shutdown after a `quit` event, and the
port released with `.manifoldx/server.json` removed. The same port is held while
the app runs, which is the control for the release check.

Usage: native-dev-channel.py [--manifoldx PATH] [--xvfb PATH] [--keep DIR]
Prints `ok N  label` per check and a final `checks: N`; exits 1 on the first
failure. The run directory (app, log and server state) is printed and kept.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

APP = '''\
from manifoldx import Engine

engine = Engine("e8 dev channel", width=320, height=200)
engine.state.setdefault("where", {"place": "yard", "x": 2.5})
engine.state.setdefault("hits", [])


@engine.on("hit")
def hit(payload):
    """Hit something: payload {"what": name}."""
    engine.state["hits"].append(payload["what"])
    engine.log(f"hit {payload['what']}")


@engine.on("probe")
def probe(payload):
    """Log the x the app reads, for the frame barrier."""
    engine.log(f"probe {engine.state['where']['x']}")


@engine.on("quit")
def quit_app(payload):
    """Close the window and the dev server."""
    engine.log("quitting")
    engine.quit()


engine.run()
'''

checks = 0


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
        [xvfb, screen, "-screen", "0", "640x400x24", "-nolisten", "tcp"],
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
            if done.returncode != expect:
                raise AssertionError(
                    f"manifoldx {' '.join(argv)} exited {done.returncode}, expected {expect}\n"
                    f"stdout:\n{done.stdout}\nstderr:\n{done.stderr}"
                )
            return done

        state = json.loads(cli("state").stdout)
        check(state["where"] == {"place": "yard", "x": 2.5}, "state reads engine.state")

        patched = json.loads(cli("set", '{"where": {"x": 7, "place": null}, "new": 1}').stdout)
        check(patched["where"] == {"x": 7} and patched["new"] == 1, "set applies a merge patch")
        check(json.loads(cli("state", "where.x").stdout) == 7, "state rereads the patched value")

        events = cli("events").stdout
        check(
            "hit\n    Hit something" in events and "quit\n    Close the window" in events,
            "events lists the app's handlers and docstrings",
        )

        fired = json.loads(cli("emit", "hit", '{"what": "the wall"}').stdout)
        check(fired == {"emitted": "hit", "handlers": 1}, "emit reaches the registered handler")
        log = cli("log", "--match", "hit the wall", "--wait", "10").stdout
        check(log.endswith("hit the wall\n"), "the handler's log line is observed")
        check(
            json.loads(cli("state", "hits").stdout) == ["the wall"],
            "state shows the handler's effect",
        )

        cli("emit", "probe", "null")
        probe = cli("log", "--match", "probe 7", "--wait", "10").stdout
        check(
            probe.endswith("probe 7\n"),
            "the frame barrier applies set before the next handler runs",
        )

        rejected = cli("emit", "nothing", "null", expect=1)
        check(
            "404" in rejected.stderr and "hit" in rejected.stderr,
            "an unknown event is rejected with 404",
        )

        unavailable = cli("look", expect=1)
        check(
            "501" in unavailable.stderr and "Rust core" in unavailable.stderr,
            "look answers 501 with the reason",
        )
        check(not (root / ".manifoldx" / "look.png").exists(), "a refused look writes no PNG")

        cli("emit", "quit", "null")
        code = app.wait(timeout=60)
        check(code == 0, f"the app exits normally after the quit event (code {code})")
        check(not manifest.exists(), ".manifoldx/server.json is removed on close")
        check(wait_port(port, 10, closed=True), "the port is released on close")
        log_text = (root / ".manifoldx" / "log.txt").read_text()
        check(
            "hit the wall" in log_text and "quitting" in log_text and "closed" in log_text,
            "the log keeps the run, the event and the close",
        )
        stderr = app.stderr.read()
        check("dev page and API at" in stderr, "the runner announced the dev page")
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
