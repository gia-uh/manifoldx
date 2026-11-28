#!/usr/bin/env python3
"""Prove installed native video export keeps size, order, timing and failure cleanup.

Run this from a disposable cwd after a native wheel is installed, so
`manifoldx`, `manifoldx._core_native` and the console script come from
site-packages. It runs the app's documented `--render` entrypoint twice: once
through the installed `manifoldx run ... -- --render` command and once by
running the app script itself, both with `MANIFOLDX_CORE=rust`.

The fixture app paints a 320x200 background red for four frames and green for
the next four, at 10 fps for 0.8 s. The gate checks the requested size (320x200,
not imageio-ffmpeg's 320x208 padding), frame count and rate, monotonic 0.1 s
frame timestamps, the decoded red-then-green order, a failed frame's exit code
and removed partial file, and a frozen negative control that must show no
change. It records every invocation and observation under `--keep`.

Prints `ok N  label` per check and a final `checks: N`; exits 1 on the first
failure.

Usage: native-mp4-export.py [--manifoldx PATH] [--python PATH] [--ffmpeg PATH]
                            [--ffprobe PATH] [--keep DIR]
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

APP = """\
import os

import manifoldx as mx

engine = mx.Engine("e8 mp4", width=320, height=200)
freeze = bool(int(os.environ.get("E8_FREEZE", "0")))
fail_after = int(os.environ.get("E8_FAIL_AFTER", "0"))
frames = 0


@engine.system
def paint(query, dt):
    global frames
    frames += 1
    if fail_after and frames > fail_after:
        raise RuntimeError("deliberate frame failure")
    if not freeze:
        engine.background_color = (1.0, 0.0, 0.0) if frames <= 4 else (0.0, 1.0, 0.0)


if __name__ == "__main__":
    engine.cli()
"""

WIDTH, HEIGHT, FPS, DURATION = 320, 200, 10, 0.8
FRAMES = int(FPS * DURATION)

checks = 0
invocations = []


def check(condition, label):
    global checks
    if not condition:
        raise AssertionError(label)
    checks += 1
    print(f"ok {checks}  {label}", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(argv, cwd, env, process, output=None):
    entry = {
        "argv": [str(arg) for arg in argv],
        "cwd": str(cwd),
        "env": {
            key: env[key] for key in ("MANIFOLDX_CORE", "E8_FREEZE", "E8_FAIL_AFTER") if key in env
        },
        "exit": process.returncode,
        "stdout": process.stdout[-2000:],
        "stderr": process.stderr[-2000:],
    }
    if output is not None:
        entry["output"] = str(output)
        entry["output_sha256"] = sha256(output) if output.exists() else None
        entry["output_bytes"] = output.stat().st_size if output.exists() else None
    invocations.append(entry)
    return process


def run(argv, cwd, env):
    return subprocess.run(
        [str(arg) for arg in argv],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )


def probe(ffprobe, path):
    raw = subprocess.check_output(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,nb_frames,r_frame_rate,avg_frame_rate,duration:"
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(raw)


def decode(ffmpeg, ffprobe, path):
    header = subprocess.check_output(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        text=True,
    ).strip()
    width, height = (int(value) for value in header.split(","))
    raw = subprocess.check_output(
        [ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    )
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def observations(path, ffprobe, ffmpeg):
    info = probe(ffprobe, path)
    stream = info["streams"][0]
    times = [float(frame["best_effort_timestamp_time"]) for frame in info.get("frames", [])]
    frames = decode(ffmpeg, ffprobe, path)
    center = [frames[index, 4, 4].tolist() for index in range(len(frames))]
    return {
        "stream": stream,
        "times": times,
        "decoded_shape": list(frames.shape),
        "corner_rgb": center,
        "first_last_mean_abs_diff": float(
            np.abs(frames[0].astype(np.int16) - frames[-1].astype(np.int16)).mean()
        ),
    }


def assert_video(path, ffprobe, ffmpeg, prefix):
    obs = observations(path, ffprobe, ffmpeg)
    stream = obs["stream"]
    check((stream["width"], stream["height"]) == (WIDTH, HEIGHT), f"{prefix}: requested size")
    check(int(stream["nb_frames"]) == FRAMES, f"{prefix}: {FRAMES} frames")
    check(stream["r_frame_rate"] == f"{FPS}/1", f"{prefix}: {FPS} fps")
    check(abs(float(stream["duration"]) - DURATION) < 1e-3, f"{prefix}: {DURATION:g} s")
    check(stream["codec_name"] == "h264", f"{prefix}: H.264")
    check(
        obs["decoded_shape"] == [FRAMES, HEIGHT, WIDTH, 3],
        f"{prefix}: decoded frame count and size",
    )
    times = obs["times"]
    check(len(times) == FRAMES, f"{prefix}: {FRAMES} frame timestamps")
    check(all(b > a for a, b in zip(times, times[1:])), f"{prefix}: frame timestamps increase")
    check(
        all(abs((b - a) - 1 / FPS) < 1e-3 for a, b in zip(times, times[1:])),
        f"{prefix}: frame timestamps are 1/{FPS} s apart",
    )
    corners = obs["corner_rgb"]
    check(
        all(pixel[0] > 200 and pixel[1] < 30 for pixel in corners[:4]),
        f"{prefix}: first four decoded frames are red",
    )
    check(
        all(pixel[1] > 200 and pixel[0] < 30 for pixel in corners[4:]),
        f"{prefix}: last four decoded frames are green",
    )
    check(obs["first_last_mean_abs_diff"] > 20, f"{prefix}: the scene visibly changes")
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifoldx", default="manifoldx")
    parser.add_argument("--python", default=None)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    parser.add_argument("--keep", default=None)
    args = parser.parse_args()

    console = Path(args.manifoldx).resolve()
    python = Path(args.python).resolve() if args.python else console.with_name("python")
    ffmpeg, ffprobe = args.ffmpeg, args.ffprobe
    check(console.is_file(), f"installed console script {console}")
    check(python.is_file(), f"installed interpreter {python}")
    check(bool(ffmpeg), "ffmpeg is on PATH")
    check(bool(ffprobe), "ffprobe is on PATH")

    root = Path(args.keep).resolve() if args.keep else Path(tempfile.mkdtemp(prefix="e8-mp4-"))
    root.mkdir(parents=True, exist_ok=True)
    app = root / "app.py"
    app.write_text(APP)
    env = {
        key: value for key, value in os.environ.items() if key not in ("DISPLAY", "WAYLAND_DISPLAY")
    }
    env["MANIFOLDX_CORE"] = "rust"

    try:
        cli_video = root / "cli.mp4"
        process = record(
            [
                console,
                "run",
                "app.py",
                "--",
                "--render",
                "--duration",
                str(DURATION),
                "--fps",
                str(FPS),
                "--output",
                str(cli_video),
            ],
            root,
            env,
            run(
                [
                    console,
                    "run",
                    "app.py",
                    "--",
                    "--render",
                    "--duration",
                    str(DURATION),
                    "--fps",
                    str(FPS),
                    "--output",
                    str(cli_video),
                ],
                root,
                env,
            ),
            cli_video,
        )
        check(process.returncode == 0, f"manifoldx run --render exits 0 ({process.returncode})")
        check(cli_video.is_file(), "the CLI entrypoint writes the video")
        cli_obs = assert_video(cli_video, ffprobe, ffmpeg, "CLI")

        app_video = root / "app.mp4"
        process = record(
            [
                python,
                "app.py",
                "--render",
                "--duration",
                str(DURATION),
                "--fps",
                str(FPS),
                "--output",
                str(app_video),
            ],
            root,
            env,
            run(
                [
                    python,
                    "app.py",
                    "--render",
                    "--duration",
                    str(DURATION),
                    "--fps",
                    str(FPS),
                    "--output",
                    str(app_video),
                ],
                root,
                env,
            ),
            app_video,
        )
        check(process.returncode == 0, f"the app entrypoint exits 0 ({process.returncode})")
        app_obs = assert_video(app_video, ffprobe, ffmpeg, "app")

        fail_video = root / "fail.mp4"
        fail_env = {**env, "E8_FAIL_AFTER": "2"}
        process = record(
            [
                python,
                "app.py",
                "--render",
                "--duration",
                str(DURATION),
                "--fps",
                str(FPS),
                "--output",
                str(fail_video),
            ],
            root,
            fail_env,
            run(
                [
                    python,
                    "app.py",
                    "--render",
                    "--duration",
                    str(DURATION),
                    "--fps",
                    str(FPS),
                    "--output",
                    str(fail_video),
                ],
                root,
                fail_env,
            ),
            fail_video,
        )
        check(process.returncode != 0, f"a failed frame exits nonzero ({process.returncode})")
        check("deliberate frame failure" in process.stderr, "the frame error reaches stderr")
        check(not fail_video.exists(), "a failed export removes its partial video")

        frozen_video = root / "frozen.mp4"
        frozen_env = {**env, "E8_FREEZE": "1"}
        process = record(
            [
                python,
                "app.py",
                "--render",
                "--duration",
                str(DURATION),
                "--fps",
                str(FPS),
                "--output",
                str(frozen_video),
            ],
            root,
            frozen_env,
            run(
                [
                    python,
                    "app.py",
                    "--render",
                    "--duration",
                    str(DURATION),
                    "--fps",
                    str(FPS),
                    "--output",
                    str(frozen_video),
                ],
                root,
                frozen_env,
            ),
            frozen_video,
        )
        check(process.returncode == 0, "the frozen control exports")
        frozen = observations(frozen_video, ffprobe, ffmpeg)
        check(
            (frozen["stream"]["width"], frozen["stream"]["height"]) == (WIDTH, HEIGHT),
            "the frozen control keeps the requested size",
        )
        check(
            frozen["first_last_mean_abs_diff"] < 1.0,
            f"the frozen control shows no change ({frozen['first_last_mean_abs_diff']:.3f})",
        )

        (root / "invocations.json").write_text(json.dumps(invocations, indent=1) + "\n")
        manifest = {
            "checks": checks,
            "app_sha256": sha256(app),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "console": str(console),
            "python": str(python),
            "ffmpeg": ffmpeg,
            "ffprobe": ffprobe,
            "cli": cli_obs,
            "app": app_obs,
            "frozen": frozen,
            "source_overlays": [],
        }
        (root / "observations.json").write_text(json.dumps(manifest, indent=1) + "\n")
    finally:
        print(f"run directory: {root}", flush=True)

    print(f"checks: {checks}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
