#!/usr/bin/env python3
"""Prove the installed native wheel's bounded desktop GPU timestamps.

Run this after a native wheel is installed, so `manifoldx`, its extension and
the console script come from site-packages. It starts a private Xvfb display
(the physical DISPLAY and WAYLAND_DISPLAY are cleared) and runs the installed
`manifoldx run` on a copied fixture, never the checkout.

The fixture draws one lit cube. The gate drives three sessions and reads the
run report each one leaves behind:

- a bounded diagnostic (`MANIFOLDX_DESKTOP_FRAMES=N`) on the capable adapter:
  every presented frame owns one positive `gpu_render_ns`, in nanoseconds, and
  `gpu_timings.state` is `measured`;
- the same bounded session with `MANIFOLDX_GPU_TIMINGS=0`: no timestamps are
  written, every value is null and the state is `disabled`;
- a normal uncompressed run that quits itself: no frame is sampled, the report
  holds no per-frame GPU value and the state is `not_bounded`, so a normal
  series never pays for a readback.

`Engine.gpu_timings()` is called after each run and must agree with that
report: `main` and `total` in milliseconds for the newest measured frame, only
the passes that wrote queries, and None in the disabled and normal states. A
software adapter is a negative control: the native desktop refuses it, so it
cannot silently fall back and fabricate fast timing numbers.

If a Mesa lavapipe ICD is installed, the gate also records that refusal; an
adapter without timestamp queries cannot be produced on the test host, so the
`unavailable` state is recorded as a limit, not exercised.

Usage: native-gpu-timings.py [--manifoldx PATH] [--xvfb PATH] [--wheel PATH]
                             [--keep DIR]
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FRAMES = 24
SCREEN = (640, 400)  # the private Xvfb root, never the report
LAVAPIPE = "/usr/share/vulkan/icd.d/lvp_icd.json"

APP = """\
import json
import os

import manifoldx as mx

engine = mx.Engine("e8 gpu timings", width=320, height=200, max_entities=8)
mesh = mx.geometry.cube(1, 1, 1)
material = mx.material.phong(mx.colors.RED)
quit_after = int(os.environ.get("E8_QUIT_AFTER", "0"))
frames = 0


@engine.on("startup")
def build(_payload):
    engine.spawn(
        mx.components.Mesh(mesh),
        mx.components.Material(material),
        mx.components.Transform(pos=(0, 0, 0)),
    )


@engine.system
def spin(query: mx.Query[mx.Transform], dt: float):
    global frames
    frames += 1
    if quit_after and frames >= quit_after:
        engine.quit()


engine.run()
if path := os.environ.get("E8_API_OUT"):
    with open(path, "w") as out:
        json.dump(engine.gpu_timings(), out)
"""

checks = 0
invocations = []


def check(condition, label):
    global checks
    if not condition:
        raise AssertionError(label)
    checks += 1
    print(f"ok {checks}  {label}", flush=True)


def free_display():
    for number in range(90, 130):
        if not Path(f"/tmp/.X11-unix/X{number}").exists():
            return number
    raise RuntimeError("no free X display number between :90 and :129")


def installed_identity(console, wheel):
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
        "console_sha256": hashlib.sha256(console.read_bytes()).hexdigest(),
        "manifoldx_module": module_path("manifoldx"),
        "core_module": module_path("manifoldx._core_native"),
        "version": version,
        "wheel": None if wheel is None else str(wheel.resolve()),
        "wheel_sha256": None if wheel is None else hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def read_json(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        else Path(tempfile.mkdtemp(prefix="manifoldx-e8-gpu.", dir="/var/tmp"))
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
    try:
        deadline = time.monotonic() + 20
        while not Path(f"/tmp/.X11-unix/X{display}").exists():
            if server.poll() is not None or time.monotonic() > deadline:
                raise SystemExit(f"FAIL: Xvfb did not start on {screen}")
            time.sleep(0.1)
        check(True, f"private Xvfb on {screen}; physical DISPLAY/WAYLAND_DISPLAY cleared")

        def process_port():
            import socket

            probe = socket.socket()
            try:
                probe.bind(("127.0.0.1", 0))
                return probe.getsockname()[1]
            finally:
                probe.close()

        def run(name, extra, timeout=90):
            report = root / f"{name}-report.json"
            api = root / f"{name}-api.json"
            argv = [str(console), "run", "app.py", "--port", str(process_port())]
            session = env | {
                "MANIFOLDX_DESKTOP_REPORT": str(report),
                "E8_API_OUT": str(api),
                **extra,
            }
            done = subprocess.run(
                argv,
                cwd=root,
                env=session,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            invocations.append(
                {
                    "name": name,
                    "argv": argv,
                    "env": {key: session[key] for key in extra},
                    "exit": done.returncode,
                    "stdout": done.stdout,
                    "stderr": done.stderr,
                }
            )
            (root / f"{name}-worker.log").write_text(done.stdout + done.stderr)
            return done, report, api

        def report_of(name, report, api):
            if not report.is_file():
                raise AssertionError(f"{name}: the run wrote no report")
            if not api.is_file():
                raise AssertionError(f"{name}: the app did not read Engine.gpu_timings()")
            return read_json(report), read_json(api)

        # ---- bounded diagnostic on the capable adapter ----
        done, report_path, api_path = run("bounded", {"MANIFOLDX_DESKTOP_FRAMES": str(FRAMES)})
        check(done.returncode == 0, f"bounded run exits 0 ({done.returncode})")
        report, api = report_of("bounded", report_path, api_path)
        (root / "bounded-report.json").write_text(json.dumps(report, indent=1) + "\n")
        adapter = json.loads(report["adapter"])
        (root / "adapter.json").write_text(json.dumps(adapter, indent=1) + "\n")
        check(
            adapter["device_type"] != "Cpu",
            f"bounded diagnostic runs on a hardware adapter ({adapter['name']}, {adapter['backend']})",
        )
        check(report["presented"] == FRAMES, f"the session presents {FRAMES} frames")
        samples = report["samples"]
        check(
            len(samples) == FRAMES, "one sample per presented frame, not a shared or repeated one"
        )
        check(
            all(sample["sequence"] == index + 1 for index, sample in enumerate(samples)),
            "every sample is tied to its own frame sequence",
        )
        state = report.get("gpu_timings") or {}
        check(state.get("state") == "measured", "the report says the timestamps were measured")
        check(state.get("pass") == "main", "the report names the one pass that wrote the queries")
        check(state.get("unit") == "ns", "the report names the raw unit as nanoseconds")
        check(
            state.get("frames") == FRAMES, "the report counts one measured frame per presentation"
        )
        values = [sample.get("gpu_render_ns") for sample in samples]
        check(
            all(isinstance(value, int) and value > 0 for value in values),
            "every presented frame owns a positive integer GPU time",
        )
        check(
            all(value < 5_000_000_000 for value in values),
            "the values are a frame's GPU work, not raw ticks or seconds",
        )
        check(
            len(set(values)) > 1 and max(values) - min(values) > 0,
            "the samples vary frame to frame instead of repeating one constant",
        )
        check(
            isinstance(api, dict) and set(api) == {"main", "total"},
            "Engine.gpu_timings() reports only the measured pass and its total",
        )
        newest = values[-1]
        check(
            api["main"] == api["total"] and abs(api["main"] * 1e6 - newest) <= 1e-3,
            "Engine.gpu_timings() is the newest frame's measured interval in milliseconds",
        )
        observations = {
            "bounded_adapter": adapter,
            "bounded_state": state,
            "bounded_gpu_render_ns": values,
            "bounded_reported_ms": api,
            "bounded_newest_ns": newest,
        }

        # ---- the documented off switch ----
        done, report_path, api_path = run(
            "disabled",
            {"MANIFOLDX_DESKTOP_FRAMES": str(FRAMES), "MANIFOLDX_GPU_TIMINGS": "0"},
        )
        check(done.returncode == 0, f"disabled run exits 0 ({done.returncode})")
        report, api = report_of("disabled", report_path, api_path)
        (root / "disabled-report.json").write_text(json.dumps(report, indent=1) + "\n")
        check(report["presented"] == FRAMES, "the disabled run still presents every frame")
        state = report.get("gpu_timings") or {}
        check(state.get("state") == "disabled", "MANIFOLDX_GPU_TIMINGS=0 is explicit in the report")
        check(state.get("frames") == 0, "the disabled run measured no frame")
        check(
            all(sample.get("gpu_render_ns") is None for sample in report["samples"]),
            "the disabled run carries no GPU value for any frame",
        )
        check(api is None, "Engine.gpu_timings() is None when the switch is off")
        observations["disabled_state"] = state

        # ---- a normal series takes no measurement ----
        done, report_path, api_path = run("normal", {"E8_QUIT_AFTER": str(FRAMES)}, timeout=90)
        check(done.returncode == 0, f"normal run exits 0 ({done.returncode})")
        report, api = report_of("normal", report_path, api_path)
        (root / "normal-report.json").write_text(json.dumps(report, indent=1) + "\n")
        check(report["presented"] >= FRAMES - 1, "the normal run presents its frames")
        state = report.get("gpu_timings") or {}
        check(state.get("state") == "not_bounded", "a normal run states that it did not measure")
        check(report["samples"] == [], "a normal run keeps no per-frame GPU sample")
        check(api is None, "Engine.gpu_timings() is None outside a bounded diagnostic")
        observations["normal_state"] = state

        # ---- no software fallback ----
        if Path(LAVAPIPE).is_file():
            done, report_path, api_path = run(
                "software",
                {
                    "MANIFOLDX_DESKTOP_FRAMES": str(FRAMES),
                    "VK_DRIVER_FILES": LAVAPIPE,
                },
                timeout=90,
            )
            check(
                done.returncode != 0 and "hardware adapter required" in done.stderr,
                "the software adapter is refused instead of timed on a fallback",
            )
            observations["software_adapter"] = "refused: hardware adapter required"
        else:
            observations["software_adapter"] = f"not exercised: no {LAVAPIPE}"

        observations["unavailable_state"] = (
            "not exercised: the host has no hardware adapter without timestamp queries"
        )
        (root / "observations.json").write_text(json.dumps(observations, indent=1) + "\n")
        (root / "identity.json").write_text(
            json.dumps(installed_identity(console, wheel), indent=1) + "\n"
        )
        (root / "invocations.json").write_text(json.dumps(invocations, indent=1) + "\n")
        print(f"run directory: {root}")
        print(f"checks: {checks}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
