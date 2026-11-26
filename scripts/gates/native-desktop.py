"""Verify the installed native desktop wheel from an X11 or Xwayland desktop."""

import argparse
import ctypes as c
import json
import os
import re
import runpy
import subprocess
import sys
import time
from pathlib import Path

import manifoldx as mx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "examples/kernel_drive.py"


def recycle(world, car_type, directory):
    old = world.query(car_type).read().entities[1]
    payload = {}
    for name in ("Transform", "Material", "Mesh"):
        rows = world.query(car_type).read(name)
        payload[name] = rows.numpy()[1:2].copy()

    @world.system(structural=True)
    def replace(snapshot, commands):
        if snapshot.clock.frame == 3:
            commands.destroy((old,))
            commands.spawn(car_type(pos=(2.5, 0, -1.6), forward=(1, 0, 0)), payload=payload)

    (directory / "old-entity.json").write_text(json.dumps([old.slot, old.generation]))


def worker(mode, directory):
    assert sys.flags.isolated
    assert Path(mx.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    if mode in {"move", "no-dispatch", "recycle"}:
        app = runpy.run_path(str(APP))
        engine = app["desktop_scene"]()
        if mode == "recycle":
            recycle(engine.world, app["Car"], directory)
        engine._desktop_probe = (mode == "no-dispatch", False, mode == "recycle", None)
    else:
        engine = runpy.run_path(str(ROOT / "examples/cube.py"))["engine"]
        engine.title = "Desktop verification"
        engine._desktop_probe = (
            False,
            mode == "no-present",
            False,
            (704, 512) if mode == "resize" else None,
        )
    engine.run()
    report = engine.desktop_report
    expected = int(os.environ["MANIFOLDX_DESKTOP_FRAMES"])
    assert report["presented"] == expected, (
        f"present oracle: expected {expected}, observed {report['presented']}"
    )
    assert report["bridge_calls"] == 0
    if mode == "resize":
        pixels = {tuple(sample["physical_px"]) for sample in report["samples"]}
        assert pixels == {(800, 600), (704, 512)}, f"resize oracle: physical sizes {pixels}"
        assert {sample["scale"] for sample in report["samples"]} == {2}, "DPI oracle: expected 2×"
    if mode == "recycle":
        stats = report["resident"]
        old = json.loads((directory / "old-entity.json").read_text())
        assert len(stats["diagnostic_rows"]) == 2
        replacement = next(row for row in stats["diagnostic_rows"] if row[0] == old[0])
        assert replacement[1] > old[1], "recycled slot retained its old generation"
        np.testing.assert_allclose(replacement[2][:5], [0, 0, 2.5, 0, -1.6], atol=1e-6)
        assert stats["generations_seeded"] == 3 and stats["generations_retired"] == 1, stats
        assert stats["diagnostic_readbacks"] == 1 and stats["diagnostic_bytes"] == 64, stats
        try:
            engine.world.query(app["Car"]).read()
        except RuntimeError as error:
            assert "resident column is unavailable" in str(error)
        else:
            raise AssertionError("closed World returned stale CPU rows")


class _Keyboard:
    def __init__(self, window):
        self.xlib = c.CDLL("libX11.so.6")
        self.xtest = c.CDLL("libXtst.so.6")
        self.xlib.XOpenDisplay.argtypes = [c.c_char_p]
        self.xlib.XOpenDisplay.restype = c.c_void_p
        self.xlib.XSetInputFocus.argtypes = [c.c_void_p, c.c_ulong, c.c_int, c.c_ulong]
        self.xlib.XKeysymToKeycode.argtypes = [c.c_void_p, c.c_ulong]
        self.xlib.XKeysymToKeycode.restype = c.c_ubyte
        self.xlib.XSync.argtypes = [c.c_void_p, c.c_int]
        self.xlib.XCloseDisplay.argtypes = [c.c_void_p]
        self.xtest.XTestFakeKeyEvent.argtypes = [c.c_void_p, c.c_uint, c.c_int, c.c_ulong]
        self.display = self.xlib.XOpenDisplay(None)
        assert self.display, "cannot open the X11 display"
        self.xlib.XSetInputFocus(self.display, int(window, 16), 2, 0)

    def key_w(self, pressed):
        key = self.xlib.XKeysymToKeycode(self.display, ord("w"))
        assert key and self.xtest.XTestFakeKeyEvent(self.display, key, pressed, 0)
        self.xlib.XSync(self.display, 0)

    def close(self):
        self.xlib.XCloseDisplay(self.display)


def window_id(process, title):
    for _ in range(100):
        if process.poll() is not None:
            break
        info = subprocess.run(["xwininfo", "-name", title], capture_output=True, text=True)
        match = re.search(r"Window id: (0x\w+)", info.stdout)
        if match:
            return match[1]
        time.sleep(0.04)
    raise RuntimeError(f"{title!r} did not map on the X11 display")


def capture(window, path):
    subprocess.run(["import", "-window", window, str(path)], check=True, timeout=5)
    with Image.open(path) as png:
        rgb = np.asarray(png.convert("RGB"))
    red = (rgb[:, :, 0] > 225) & (rgb[:, :, 1] < 70) & (rgb[:, :, 2] < 45)
    green = (rgb[:, :, 1] > 200) & (rgb[:, :, 0] < 50) & (rgb[:, :, 2] < 100)
    return tuple(
        float(mask.nonzero()[1].mean()) if mask.sum() > 50 else None for mask in (red, green)
    )


def interact(process, directory):
    window = window_id(process, "Resident Drive")
    time.sleep(0.2)
    for _ in range(40):
        before = capture(window, directory / "before.png")
        if all(x is not None for x in before):
            break
        time.sleep(0.05)
    else:
        raise AssertionError("position oracle: expected both cars on the screen")
    keyboard = _Keyboard(window)
    keyboard.key_w(True)
    try:
        time.sleep(1.2)
        held = capture(window, directory / "held.png")
    finally:
        keyboard.key_w(False)
        keyboard.close()
    time.sleep(1.0)
    released = capture(window, directory / "released.png")
    time.sleep(0.35)
    settled = capture(window, directory / "settled.png")
    return [before, held, released, settled]


def validate_motion(centers, report):
    before, held, released, settled = centers
    assert all(all(x is not None for x in pair) for pair in centers), centers
    assert held[0] - before[0] >= 18, f"motion oracle: W did not move the red car: {centers}"
    assert abs(settled[0] - released[0]) < 4, f"motion oracle: red car did not brake: {centers}"
    assert max(pair[1] for pair in centers) - min(pair[1] for pair in centers) < 2, (
        f"entity oracle: W moved the other car: {centers}"
    )
    stats = report["resident"]
    assert stats["dispatches"] == stats["version"] == report["presented"], stats
    assert stats["field_uploads"] == 2 and stats["field_upload_bytes"] == 8, stats
    assert stats["row_uploads"] == 1 and stats["row_upload_bytes"] == 64, stats
    assert stats["diagnostic_readbacks"] == stats["diagnostic_bytes"] == 0, stats


def verify_binding():
    app = runpy.run_path(str(APP))
    engine = mx.Engine("Binding validation", max_entities=1)
    drive = app["drive"]
    for field, target, expected in (
        ("pos", "cpu", "requires target='wgsl'"),
        ("pos", "wgsl", "unregistered component"),
    ):
        reject_binding(engine, drive, field, target, expected)
    engine.world.register(app["Car"])
    reject_binding(engine, drive, "speed", "wgsl", "vec3<f32>")
    reject_binding(engine, drive, "forward", "wgsl", "written by the kernel")
    engine.bind_position(drive, field="pos", target="wgsl")
    try:
        engine.render_frame()
    except NotImplementedError as error:
        assert "requires the native desktop mode" in str(error)
    else:
        raise AssertionError("resident position unexpectedly allowed offscreen mode")


def reject_binding(engine, drive, field, target, expected):
    try:
        engine.bind_position(drive, field=field, target=target)
    except ValueError as error:
        assert expected in str(error), str(error)
    else:
        raise AssertionError(f"invalid binding was accepted: {field}, {target}")


def exercise(mode, directory):
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "report.json"
    env = dict(os.environ, MANIFOLDX_CORE="rust", MANIFOLDX_DESKTOP_REPORT=str(report_path))
    env.pop("WAYLAND_DISPLAY", None)
    env.pop("WAYLAND_SOCKET", None)
    env["MANIFOLDX_DESKTOP_FRAMES"] = "360" if mode in {"move", "no-dispatch"} else "60"
    env["WINIT_X11_SCALE_FACTOR"] = "2" if mode == "resize" else "1"
    with (directory / "worker.log").open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-I", __file__, mode, str(directory), "--worker"],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            centers = interact(process, directory) if mode in {"move", "no-dispatch"} else None
            status = process.wait(timeout=25)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    (directory / "worker.status").write_text(f"{status}\n")
    if mode == "no-present":
        assert status == 1 and "present oracle" in (directory / "worker.log").read_text(), status
        report = json.loads(report_path.read_text())
        assert report["scheduler_frames"] == 60 and report["presented"] == 59, report
    else:
        assert status == 0, (directory / "worker.log").read_text()
        report = json.loads(report_path.read_text())
    if centers is not None:
        (directory / "pixel-centers.json").write_text(json.dumps(centers) + "\n")
        validate_motion(centers, report)
    print(f"{mode}: {report['presented']} presents, {report['bridge_calls']} graphics bridge calls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("move", "no-dispatch", "no-present", "recycle", "resize", "binding")
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.mode, args.directory.resolve())
    elif args.mode == "binding":
        verify_binding()
    else:
        exercise(args.mode, args.directory.resolve())
