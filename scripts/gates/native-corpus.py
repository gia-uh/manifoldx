#!/usr/bin/env python3
"""GW-1 public corpus: run every in-scope example and demo against one installed wheel.

The gate does not build anything. It takes an already installed environment
(``--python``, normally a venv holding exactly one ``manifold-gfx`` wheel),
hashes that wheel, copies the corpus into a scratch root with the assets its
scripts read, starts a private Xvfb display, and runs each case through the
entrypoint its documentation names with ``MANIFOLDX_CORE=rust``.

Per case it records the raw command, exit code, wall time, artifact paths and
hashes, the oracle's numbers, and any declared limit. A functional oracle,
not exit 0, decides the case: a decoded MP4 with non-blank and changing
frames, a parsed GLB, a stereo WAV with left/right variation, a still whose
pixels are not uniform, the diagnostic runtime's own assertions, the headless
stream's samples, or the vehicle's fixed-substep report.

Interaction cases get a second, explicitly separate lane: the probe imports
the same copied script offscreen and delivers rendercanvas event dictionaries
through the engine's input bridge (or the widget's own bus event), then
compares frames and state. Controls break one signal on purpose: a scene with
no entities must fail the non-blank oracle, a truncated GLB must fail the
parser, and each event case has a copy with the event removed that must stop
changing.

The manifest below is the 49 pinned runs: 43 product examples, the installed
runtime diagnostic and the five demos. Files added after the pinned spec are
listed as inventory additions and are not counted; they are reported so a PR
can name them.

Usage:
    native-corpus.py --python VENV/bin/python --wheel dist/foo.whl \
        --corpus /var/tmp/corpus --out /var/tmp/run [--only a,b] [--no-probe]
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE = HERE / "native-corpus-probe.py"

# --- the 49 pinned runs ----------------------------------------------------

VIDEO = ("--render", "--duration", "1.2", "--fps", "6")

CASES = [
    # name, entry arguments, extras, oracle, expected outcome, probe/controls
    ("examples/audio.py", ["--out", "{dir}/circle.wav", "--seconds", "1.5"], ["audio"], "wav",
     "stereo WAV, right duration, signal and left/right variation", None),
    ("examples/axes.py", list(VIDEO) + ["--output", "{dir}/axes.mp4"], [], "video",
     "RGB axes and anchored labels over the scene", None),
    ("examples/boids.py", list(VIDEO) + ["--output", "{dir}/boids.mp4"], [], "video",
     "flock moves and groups; two frames differ", None),
    ("examples/booleans.py", list(VIDEO) + ["--output", "{dir}/booleans.mp4"], [], "video",
     "CSG openings and volumes in the image", None),
    ("examples/color_by.py", list(VIDEO) + ["--output", "{dir}/color_by.mp4"], [], "video",
     "gradient coloured by field", None),
    ("examples/cube.py", list(VIDEO) + ["--output", "{dir}/cube.mp4"], [], "video",
     "one rotating cube, non-blank and changing", None),
    ("examples/deformers.py", list(VIDEO) + ["--output", "{dir}/deformers.mp4"], [], "video",
     "deformed silhouette", None),
    ("examples/event_dolly.py", ["--render", "--duration", "2.4", "--fps", "6", "--output", "{dir}/event_dolly.mp4"],
     [], "video", "camera dollies in and out; frames change", "event_dolly"),
    ("examples/event_pulse.py", ["--render", "--duration", "4.4", "--fps", "6", "--output", "{dir}/event_pulse.mp4"],
     [], "video", "a cube is added on the pulse; frames change after 1 s", "event_pulse"),
    ("examples/ffd.py", list(VIDEO) + ["--output", "{dir}/ffd.mp4"], [], "video",
     "cage deformation changes the silhouette", None),
    ("examples/fields.py", list(VIDEO) + ["--output", "{dir}/fields.mp4"], [], "video",
     "relief from a field", None),
    ("examples/gas.py", list(VIDEO) + ["--output", "{dir}/gas.mp4"], [], "video",
     "gas particles bounce; frames differ", None),
    ("examples/gas_compute.py", list(VIDEO) + ["--output", "{dir}/gas_compute.mp4"], [], "video",
     "compute dispatch advances the gas; frames differ", None),
    ("examples/gltf_export.py", ["{dir}/terrain.glb"], [], "glb",
     "GLB parses, has topology and a material", "glb_truncated"),
    ("examples/gpu_field.py", list(VIDEO) + ["--output", "{dir}/gpu_field.mp4"], [], "video",
     "vertex-shader relief with normals", None),
    ("examples/gui.py", list(VIDEO) + ["--output", "{dir}/gui.mp4"], [], "video",
     "N-body panel and moving bodies", "gui"),
    ("examples/hello_world.py", [], ["desktop"], "stdout",
     "the documented window entrypoint prints its startup and shutdown lines", None),
    ("examples/humanoid.py", list(VIDEO) + ["--output", "{dir}/humanoid.mp4"], [], "video",
     "proportioned mannequin with shading", "foreground"),
    ("examples/humans.py", list(VIDEO) + ["--output", "{dir}/humans.mp4"], [], "video",
     "locomotion and posed figures", None),
    ("examples/ibl.py", list(VIDEO) + ["--output", "{dir}/ibl.mp4"], [], "video",
     "environment lighting preset changes the image", "foreground"),
    ("examples/input_fly.py", list(VIDEO) + ["--output", "{dir}/input_fly.mp4"], [], "video",
     "fly camera over a city", "input_fly"),
    ("examples/input_orbit.py", list(VIDEO) + ["--output", "{dir}/input_orbit.mp4"], [], "video",
     "orbit camera on a cube", "input_orbit"),
    ("examples/nbody.py", list(VIDEO) + ["--output", "{dir}/nbody.mp4"], [], "video",
     "orbits change with the physics step", None),
    ("examples/nbody_compute.py", list(VIDEO) + ["--output", "{dir}/nbody_compute.mp4"], [], "video",
     "compute dispatch advances the bodies; frames differ", None),
    ("examples/pbr.py", list(VIDEO) + ["--output", "{dir}/pbr.mp4"], [], "video",
     "brighter PBR response than the reference", "foreground"),
    ("examples/point_cloud.py", list(VIDEO) + ["--output", "{dir}/point_cloud.mp4"], [], "video",
     "dust particles with the speed LUT; frames differ", None),
    ("examples/point_cloud_compute.py", list(VIDEO) + ["--output", "{dir}/point_cloud_compute.mp4"],
     [], "video", "compute dispatch moves the cloud; frames differ", None),
    ("examples/quad_streaming.py",
     ["--render", "--duration", "1.5", "--output", "{dir}/quad_streaming.mp4",
      "--trace", "{dir}/lod.csv"], [], "video",
     "streamed quadtree LOD; frames non-blank and changing, trace non-empty", "quad"),
    ("examples/render_to_texture.py", ["--still", "{dir}/portrait.png"], ["viz", "svg"], "still",
     "still PNG with a live portrait inset", "portrait"),
    ("examples/scatter_plot.py", list(VIDEO) + ["--output", "{dir}/scatter_plot.mp4"], [], "video",
     "scatter points and colour ranges", None),
    ("examples/screens.py", list(VIDEO) + ["--output", "{dir}/screens.mp4"], [], "video",
     "screen A spins until Space; frames change", "screens"),
    ("examples/shadows.py", list(VIDEO) + ["--output", "{dir}/shadows.mp4"], [], "video",
     "teapot with directional shadow", None),
    ("examples/skinning.py", list(VIDEO) + ["--output", "{dir}/skinning.mp4"], [], "video",
     "skinned mesh pose", "foreground"),
    ("examples/spheres.py", list(VIDEO) + ["--output", "{dir}/spheres.mp4"], [], "video",
     "array of spheres", None),
    ("examples/spotlight.py", list(VIDEO) + ["--output", "{dir}/spotlight.mp4"], [], "video",
     "moving spot cone on the teapot", None),
    ("examples/stream_walk.py",
     ["--km", "0.15", "--headless", "--workers", "thread", "--evidence", "{dir}/stream"], ["viz"],
     "headless", "captured frames, drawn cells, origin rebase and frame statistics", None),
    ("examples/surface_patterns.py", ["--still", "{dir}/surface_patterns.png"], ["viz"], "still",
     "patterned surfaces still", "foreground"),
    ("examples/svg_icons.py", ["--still", "{dir}/svg_icons.png"], ["viz", "svg"], "still_pair",
     "SVG icon cube at two GUI scales", "svg_scales"),
    ("examples/textures.py", list(VIDEO) + ["--output", "{dir}/textures.mp4"], ["viz"], "video",
     "teapot with its albedo texture", "foreground"),
    ("examples/topology.py", list(VIDEO) + ["--output", "{dir}/topology.mp4"], [], "video",
     "subdivided/extruded topology", None),
    ("examples/ui_scale.py", ["--list-monitors"], ["desktop"], "monitors",
     "the private display lists its monitor", "ui_scale"),
    ("examples/vehicle.py", ["--native-physics-check"], [], "vehicle",
     "fixed-step native drive: 360 substeps, distance, speed, four wheels", "vehicle"),
    ("examples/volume.py", list(VIDEO) + ["--output", "{dir}/volume.mp4"], [], "video",
     "DVR volume contrast", "foreground"),
    ("examples/native_runtime.py",
     ["--entities", "1000", "--seconds", "0.2", "--deterministic-only"], [], "diagnostic",
     "World/Runtime, queue/identity and the CPU runtime's own assertions", "diag_control"),
    ("demos/labyrinth_fps.py", list(VIDEO) + ["--output", "{dir}/labyrinth_fps.mp4"], [], "video",
     "walk through the maze with a lantern shadow", None),
    ("demos/planet_dive.py", list(VIDEO) + ["--output", "{dir}/planet_dive.mp4"], [], "video",
     "planet from orbit to night with atmosphere", None),
    ("demos/smoke.py", list(VIDEO) + ["--output", "{dir}/smoke.mp4"], [], "video",
     "volumetric Perlin smoke", None),
    ("demos/sun_earth_moon.py", list(VIDEO) + ["--output", "{dir}/sun_earth_moon.mp4"], ["viz"],
     "video", "textured Earth/Moon and their orbit", None),
    ("demos/terrain_stream.py", list(VIDEO) + ["--output", "{dir}/terrain_stream.mp4"], [], "video",
     "streamed terrain with fog; slots change as the camera flies", None),
]

ADDITIONS = [
    ("examples/kernel_drive.py", "top-level example added after the pinned 46; outside the 49"),
    ("examples/native_vehicles.py", "diagnostic top-level added by #367; outside the 49"),
    ("examples/world_runtime.py", "runtime stress top-level; outside the 49"),
    ("examples/fr11_coast.py", "FR-11 coast scene added after the pinned 46; outside the 49"),
]

EXCLUDED = [
    ("examples/web_crowd.py", "web-export only, excluded by name in the spec"),
    ("examples/web_walk.py", "web-export only, excluded by name in the spec"),
    ("examples/native_builder/", "nested ABI fixture, not a top-level run"),
]

EXPECTED_TOTAL = 49


# --- helpers ---------------------------------------------------------------

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def free_display():
    for number in range(90, 130):
        if not Path(f"/tmp/.X11-unix/X{number}").exists():
            return number
    raise RuntimeError("no free X display number between :90 and :129")


def clean_env(display, scratch):
    drop = {"DISPLAY", "WAYLAND_DISPLAY", "PYTHONPATH", "PYTHONHOME", "MANIFOLDX_CORE"}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(
        {
            "DISPLAY": display,
            "MANIFOLDX_CORE": "rust",
            "TMPDIR": str(scratch),
            "UV_CACHE_DIR": str(scratch / "uv-cache"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


def decode_video(path, limit=10):
    """Decode up to `limit` evenly spaced RGB frames; return metrics or an error."""
    import imageio_ffmpeg
    import numpy as np

    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(reader)
    width, height = meta["size"]
    raw = list(reader)
    if not raw:
        return {"error": "no decoded frames"}
    step = max(1, len(raw) // limit)
    frames = [
        np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        for buf in raw[::step][:limit]
    ]
    stack = np.stack(frames)
    delta = np.abs(stack[-1].astype(np.int16) - stack[0].astype(np.int16))
    return {
        "width": width,
        "height": height,
        "frames": len(raw),
        "fps": meta.get("fps"),
        "duration": round(len(raw) / meta["fps"], 3) if meta.get("fps") else None,
        "mean": round(float(stack.mean()), 3),
        "std": round(float(stack.std()), 3),
        "blank_fraction": round(float((stack.max(axis=-1) <= 4).mean()), 4),
        "first_last_mean_abs": round(float(delta.mean()), 4),
        "first_last_changed_fraction": round(float((delta.max(axis=-1) > 8).mean()), 4),
    }


def check_video(metrics, motion=True, min_frames=2, min_std=3.0, min_diff=0.25):
    fails = []
    if "error" in metrics:
        fails.append(metrics["error"])
    else:
        if metrics["frames"] < min_frames:
            fails.append(f"only {metrics['frames']} frames")
        if metrics["std"] < min_std:
            fails.append(f"blank image (std {metrics['std']})")
        if motion and metrics["first_last_mean_abs"] < min_diff:
            fails.append(f"static (first/last {metrics['first_last_mean_abs']})")
    return fails


def check_still(path, min_std=3.0):
    import numpy as np
    from PIL import Image

    image = np.asarray(Image.open(path).convert("RGB"))
    metrics = {
        "shape": list(image.shape),
        "mean": round(float(image.mean()), 3),
        "std": round(float(image.std()), 3),
        "sha256": sha256(path),
    }
    fails = [] if metrics["std"] >= min_std else [f"blank still (std {metrics['std']})"]
    return metrics, fails


def check_wav(path, seconds):
    fails = []
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        frames = handle.getnframes()
        width = handle.getsampwidth()
        raw = handle.readframes(frames)
    import numpy as np

    samples = np.frombuffer(raw, dtype="<i2" if width == 2 else np.uint8).reshape(-1, channels)
    duration = frames / rate
    left, right = samples[:, 0].astype(float), samples[:, 1].astype(float)
    metrics = {
        "channels": channels,
        "rate": rate,
        "sample_width": width,
        "duration": round(duration, 3),
        "rms": round(float(np.sqrt((samples.astype(float) ** 2).mean())), 1),
        "lr_mean_abs": round(float(np.abs(left - right).mean()), 1),
    }
    if channels != 2:
        fails.append(f"{channels} channels, want stereo")
    if abs(duration - seconds) > max(0.4, 0.25 * seconds):
        fails.append(f"duration {duration:.2f}s, want about {seconds}s")
    if metrics["rms"] < 50:
        fails.append("no signal")
    if metrics["lr_mean_abs"] < 1:
        fails.append("left and right are identical")
    return metrics, fails


def check_glb(path):
    fails = []
    size = path.stat().st_size
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        return {"size": size}, ["no glTF magic"]
    version = int.from_bytes(raw[4:8], "little")
    declared = int.from_bytes(raw[8:12], "little")
    chunk_length = int.from_bytes(raw[12:16], "little")
    chunk_type = raw[16:20]
    body = raw[20 : 20 + chunk_length] if chunk_type == b"JSON" else b"{}"
    document = json.loads(body.decode("utf-8", "replace") or "{}")
    metrics = {
        "size": size,
        "version": version,
        "declared_length": declared,
        "json_length": chunk_length,
        "meshes": len(document.get("meshes", [])),
        "nodes": len(document.get("nodes", [])),
        "materials": len(document.get("materials", [])),
        "accessors": len(document.get("accessors", [])),
        "sha256": sha256(path),
    }
    if version != 2:
        fails.append(f"version {version}")
    if declared != size:
        fails.append(f"declared {declared} != file {size}")
    if metrics["meshes"] < 1 or metrics["accessors"] < 1:
        fails.append("no mesh topology")
    return metrics, fails


def check_monitors(stdout):
    lines = [line for line in stdout.splitlines() if re.match(r"^\d+: .+ \d+x\d+ @ \d+ Hz", line)]
    return {"monitors": lines}, ([] if lines else ["no monitor listed"])


def check_stdout(stdout, required):
    missing = [line for line in required if line not in stdout]
    return {"required": required, "missing": missing}, (
        [f"missing {line!r}" for line in missing] if missing else []
    )


def check_diagnostic(stdout):
    required = ["deterministic native gate", "FIFO", "passed"]
    metrics, fails = check_stdout(stdout, required)
    metrics["traceback"] = "Traceback (most recent call last)" in stdout
    if metrics["traceback"]:
        fails.append("raised")
    return metrics, fails


def check_headless(result_path, stdout):
    fails = []
    if not result_path.is_file():
        return {}, ["no result.json"]
    report = json.loads(result_path.read_text())
    samples = report.get("samples", {})
    drawn = {key: value.get("drawn") for key, value in samples.items()}
    origins = sorted({tuple(value.get("origin")) for value in samples.values() if value.get("origin")})
    stream = report.get("stream", {})
    metrics = {
        "samples": len(samples),
        "drawn": drawn,
        "origins": origins,
        "stream": stream,
        "frames_png": sorted(p.name for p in result_path.parent.glob("frame-*.png")),
    }
    if len(samples) < 4:
        fails.append(f"{len(samples)} captured samples")
    if not any(drawn.values()):
        fails.append("no cells drawn")
    if len(origins) < 2:
        fails.append("origin never rebased")
    if "FAIL:" in stdout:
        fails.append("frame statistics failed")
    return metrics, fails


def check_vehicle(stdout):
    match = re.search(
        r"(\d+) native substeps, ([\d.]+) m, ([\d.]+) m/s, (\d+) wheels", stdout
    )
    if not match:
        return {"raw": stdout.strip()[-400:]}, ["no native vehicle report"]
    substeps, distance, speed, wheels = match.groups()
    metrics = {
        "substeps": int(substeps),
        "distance": float(distance),
        "speed": float(speed),
        "wheels": int(wheels),
    }
    fails = []
    if metrics["substeps"] != 360:
        fails.append(f"{substeps} substeps, want 360")
    if metrics["distance"] <= 10.0:
        fails.append(f"distance {distance}m")
    if metrics["speed"] <= 10.0:
        fails.append(f"speed {speed}m/s")
    if metrics["wheels"] != 4:
        fails.append(f"{wheels} wheels")
    return metrics, fails


# --- interaction probes ----------------------------------------------------

def probe_spec(probe, script, out):
    """The rails and expected movement for one interaction case."""
    common = {"script": str(script), "out": str(out)}
    if probe == "event_dolly":
        return common | {
            "dt": 1 / 30,
            "rails": [
                {"frames": 2, "save": "before.png", "snapshot": True},
                {"frames": 50, "save": "after.png", "snapshot": True},
            ],
            "diff": [0, 1],
            "expect": {"diff_mean_abs": 3.0, "camera_moved": 0.15},
        }
    if probe == "input_fly":
        return common | {
            "dt": 1 / 30,
            "rails": [
                {"frames": 2, "save": "before.png", "snapshot": True},
                {
                    "events": [{"kind": "key", "key": "w", "down": True}],
                    "frames": 20,
                    "save": "after.png",
                    "snapshot": True,
                },
                {"events": [{"kind": "key", "key": "w", "down": False}], "frames": 1},
            ],
            "diff": [0, 1],
            "expect": {"camera_moved": 0.25, "diff_mean_abs": 0.5},
        }
    if probe == "input_orbit":
        return common | {
            "dt": 1 / 30,
            "rails": [
                {"frames": 2, "save": "before.png", "snapshot": True},
                {
                    "events": [
                        {"kind": "pointer", "phase": "down", "x": 400, "y": 300,
                         "button": 1, "buttons": [1]},
                        {"kind": "pointer", "phase": "move", "x": 500, "y": 330,
                         "button": 1, "buttons": [1]},
                        {"kind": "pointer", "phase": "up", "x": 500, "y": 330,
                         "button": 1, "buttons": [1]},
                        {"kind": "wheel", "dx": 0, "dy": -240, "x": 500, "y": 330},
                    ],
                    "frames": 3,
                    "save": "after.png",
                    "snapshot": True,
                },
            ],
            "diff": [0, 1],
            "expect": {"camera_moved": 0.15, "diff_mean_abs": 0.3},
        }
    if probe == "screens":
        return common | {
            "dt": 1 / 30,
            "rails": [
                {"frames": 6, "save": "screen_a.png", "snapshot": True},
                {
                    "events": [{"kind": "key", "key": "Space", "down": True}],
                    "frames": 8,
                    "save": "screen_b.png",
                    "snapshot": True,
                },
                {"events": [{"kind": "key", "key": "Space", "down": False}], "frames": 2},
            ],
            "diff": [0, 1],
            "expect": {"diff_mean_abs": 1.0},
        }
    if probe == "ui_scale":
        return common | {
            "dt": 1 / 30,
            "rails": [
                {"frames": 2, "save": "before.png", "snapshot": True},
                {
                    "events": [{"kind": "key", "key": "+", "down": True}],
                    "frames": 2,
                    "save": "after.png",
                    "snapshot": True,
                },
            ],
            "diff": [0, 1],
            "expect": {"gui_scale_over": 1.0},
        }
    if probe == "gui":
        return common | {
            "dt": 1 / 30,
            "rails": [
                {"frames": 3, "save": "before.png", "snapshot": True, "namespace": ["G_ref"]},
                {
                    "events": [
                        {"kind": "bus", "name": "gui:slider:G:change", "payload": {"value": 1.0}},
                        {"kind": "bus", "name": "gui:button:reset:click", "payload": {}},
                    ],
                    "frames": 20,
                    "save": "after.png",
                    "snapshot": True,
                    "namespace": ["G_ref"],
                },
            ],
            "diff": [0, 1],
            "expect": {"ns_G_ref": 1.0, "diff_mean_abs": 0.05},
        }
    if probe == "vehicle":
        return common | {
            "dt": 1 / 120,
            "rails": [
                {"frames": 2, "save": "before.png", "snapshot": True},
                {
                    "events": [{"kind": "key", "key": "ArrowUp", "down": True}],
                    "frames": 240,
                    "save": "after.png",
                    "snapshot": True,
                },
            ],
            "diff": [0, 1],
            "expect": {"car_speed": 5.0, "diff_mean_abs": 0.05},
        }
    if probe == "foreground":
        return common | {
            "dt": 1 / 30,
            "rails": [
                {"frames": 3, "save": "case.png", "snapshot": True},
                {"clear": True, "frames": 1, "save": "blank.png", "snapshot": True},
            ],
            "diff": [0, 1],
            "expect": {"diff_mean_abs": 0.05},
        }
    raise KeyError(probe)


def evaluate_probe(result, expect):
    fails = []
    rails = result.get("rails", [])
    if not result.get("ok"):
        return [re.sub(r"\s+", " ", result.get("error", "probe failed"))[:400]]

    def snap(index):
        return (rails[index] or {}).get("snapshot") or {}

    if "diff_mean_abs" in expect:
        got = (result.get("diff") or {}).get("mean_abs")
        if got is None or got < expect["diff_mean_abs"]:
            fails.append(f"diff {got} < {expect['diff_mean_abs']}")
    if "camera_moved" in expect:
        before = snap(0).get("camera_position")
        after = snap(-1).get("camera_position")
        if before is None or after is None:
            fails.append("no camera position")
        else:
            moved = max(abs(a - b) for a, b in zip(before, after))
            if moved < expect["camera_moved"]:
                fails.append(f"camera moved {moved:.3f} < {expect['camera_moved']}")
    if "gui_scale_over" in expect:
        scale = snap(-1).get("gui_scale")
        if scale is None or scale <= expect["gui_scale_over"]:
            fails.append(f"gui scale {scale} not over {expect['gui_scale_over']}")
    if "ns_G_ref" in expect:
        got = snap(-1).get("ns:G_ref")
        value = got[0] if isinstance(got, list) and got else got
        if value != expect["ns_G_ref"]:
            fails.append(f"G_ref {value} != {expect['ns_G_ref']}")
    if "car_speed" in expect:
        car = snap(-1).get("car") or {}
        speed = car.get("speed")
        if speed is None or speed < expect["car_speed"]:
            fails.append(f"car speed {speed} < {expect['car_speed']}")
    if "car_distance" in expect:
        car = snap(-1).get("car") or {}
        position = car.get("position") or [0, 0, 0]
        if abs(float(position[0])) < expect["car_distance"]:
            fails.append(f"car x {position[0]} < {expect['car_distance']}")
    return fails


def run_command(command, cwd, env, timeout, log):
    started = time.monotonic()
    try:
        with open(log, "w") as out:
            process = subprocess.run(
                command, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=timeout
            )
        return {
            "command": command,
            "exit": process.returncode,
            "seconds": round(time.monotonic() - started, 3),
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "exit": None,
            "seconds": round(time.monotonic() - started, 3),
            "timeout": True,
        }


def run_probe(spec, python, cwd, env, timeout, out):
    spec_path = Path(out) / "probe-spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    command = [python, "-I", str(PROBE), str(spec_path)]
    record = run_command(command, cwd, env, timeout, Path(out) / "probe.log")
    result_path = Path(spec["out"]) / "result.json"
    result = (
        json.loads(result_path.read_text())
        if result_path.is_file()
        else {"ok": False, "error": "no result.json"}
    )
    record["result"] = result
    record["fails"] = evaluate_probe(result, spec.get("expect", {}))
    return record


def write_control_script(source, target, transforms):
    text = Path(source).read_text()
    for old, new in transforms:
        if old not in text:
            raise AssertionError(f"control transform not found in {source}: {old!r}")
        text = text.replace(old, new, 1)
    Path(target).write_text(text)
    return Path(target)


# --- the run ---------------------------------------------------------------

STATIC = {
    "examples/humanoid.py",
    "examples/ibl.py",
    "examples/input_fly.py",
    "examples/input_orbit.py",
    "examples/skinning.py",
    "examples/volume.py",
}

# The pulses are 0.4 m cubes at radius 2, a small part of a full frame, so the
# change after the first pulse is real but small: a lower floor than the default.
TUNING = {
    "examples/event_pulse.py": {"min_diff": 0.05, "min_changed": 0.0005},
    "examples/screens.py": {"min_diff": 0.005},
}

PROBES = {
    "examples/event_dolly.py",
    "examples/gpu_field.py",
    "examples/gui.py",
    "examples/humanoid.py",
    "examples/ibl.py",
    "examples/input_fly.py",
    "examples/input_orbit.py",
    "examples/pbr.py",
    "examples/screens.py",
    "examples/skinning.py",
    "examples/surface_patterns.py",
    "examples/textures.py",
    "examples/ui_scale.py",
    "examples/vehicle.py",
    "examples/volume.py",
}


def probe_name(case):
    return case[5]


def identity(python, wheel):
    program = r"""
import importlib.metadata, json, platform, sys
from pathlib import Path

def path(name):
    try:
        module = __import__(name, fromlist=["__file__"])
    except Exception as error:
        return f"unavailable: {error}"
    return str(Path(module.__file__).resolve())

def has(name):
    import importlib.util
    return bool(importlib.util.find_spec(name))

print(json.dumps({
    "python": sys.executable,
    "python_version": sys.version,
    "free_threaded": not sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else None,
    "manifoldx": path("manifoldx"),
    "core": path("manifoldx._core_native"),
    "version": importlib.metadata.version("manifold-gfx"),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "extras": {name: has(module) for name, module in (("desktop", "glfw"), ("viz", "PIL"), ("offline", "imageio_ffmpeg"), ("audio", "sounddevice"), ("svg", "thorvg_python"))},
}))
"""
    result = subprocess.run([python, "-I", "-c", program], capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not result.stdout.strip():
        raise SystemExit(f"identity probe failed ({result.returncode}): {result.stderr.strip()[-600:]}")
    info = json.loads(result.stdout.strip().splitlines()[-1])
    info["wheel"] = str(wheel)
    info["wheel_sha256"] = sha256(wheel)
    info["installed_from_site_packages"] = "site-packages" in info["manifoldx"]
    return info


def run_case(case, corpus, out, python, env):
    name = case[0]
    entry_args, extras, oracle, expected, probe = case[1], case[2], case[3], case[4], case[5]
    case_dir = out / "cases" / Path(name).stem
    case_dir.mkdir(parents=True, exist_ok=True)
    script = corpus / name
    command = [python, "-I", str(script)] + [a.replace("{dir}", str(case_dir)) for a in entry_args]
    timeout = 240 if oracle in ("headless", "vehicle") else 120
    record = {
        "name": name,
        "entrypoint": " ".join(command[2:]),
        "extras": extras,
        "expected": expected,
        "oracle_kind": oracle,
    }
    record |= run_command(command, corpus, env, timeout, case_dir / "entry.log")
    stdout = (case_dir / "entry.log").read_text(errors="replace")
    fails = []
    metrics = {}

    if record["timeout"]:
        fails.append(f"timed out after {timeout}s")
    elif record["exit"] != 0:
        fails.append(f"exit {record['exit']}")

    if oracle == "video":
        artifact = case_dir / f"{Path(name).stem}.mp4"
        if not artifact.is_file():
            fails.append("no MP4")
        else:
            tuning = TUNING.get(name, {})
            metrics = decode_video(artifact)
            fails += check_video(
                metrics, motion=name not in STATIC, min_diff=tuning.get("min_diff", 0.25)
            )
            if tuning.get("min_changed") is not None:
                if metrics.get("first_last_changed_fraction", 0.0) < tuning["min_changed"]:
                    fails.append(
                        "changed fraction "
                        f"{metrics.get('first_last_changed_fraction')} < {tuning['min_changed']}"
                    )
            record["artifact"] = {"path": str(artifact), "sha256": sha256(artifact),
                                  "bytes": artifact.stat().st_size}
    elif oracle == "wav":
        artifact = case_dir / "circle.wav"
        if not artifact.is_file():
            fails.append("no WAV")
        else:
            metrics, more = check_wav(artifact, 1.5)
            fails += more
            record["artifact"] = {"path": str(artifact), "sha256": sha256(artifact),
                                  "bytes": artifact.stat().st_size}
    elif oracle == "still":
        artifact = case_dir / "surface_patterns.png" if "surface_patterns" in name else case_dir / "portrait.png"
        if not artifact.is_file():
            fails.append("no PNG")
        else:
            metrics, more = check_still(artifact)
            fails += more
            record["artifact"] = {"path": str(artifact), "sha256": sha256(artifact),
                                  "bytes": artifact.stat().st_size}
            if probe == "portrait":
                import numpy as np
                from PIL import Image

                image = np.asarray(Image.open(artifact).convert("RGB"))
                roi = image[380:580, 20:220]
                metrics["portrait_roi_std"] = round(float(roi.std()), 3)
                if metrics["portrait_roi_std"] < 5.0:
                    fails.append(f"portrait inset blank (std {metrics['portrait_roi_std']})")
    elif oracle == "still_pair":
        first = case_dir / "svg_icons.png"
        second = case_dir / "svg_icons-scaled.png"
        if not first.is_file() or not second.is_file():
            fails.append("missing one of the two stills")
        else:
            import numpy as np
            from PIL import Image

            a = np.asarray(Image.open(first).convert("RGB")).astype(np.int16)
            b = np.asarray(Image.open(second).convert("RGB")).astype(np.int16)
            metrics = {
                "first_std": round(float(a.std()), 3),
                "second_std": round(float(b.std()), 3),
                "mean_abs": round(float(np.abs(a - b).mean()), 3),
                "first_sha256": sha256(first),
                "second_sha256": sha256(second),
            }
            if metrics["first_std"] < 3.0:
                fails.append("first still blank")
            if metrics["mean_abs"] < 0.3:
                fails.append(f"GUI scale did not change the still ({metrics['mean_abs']})")
            record["artifact"] = {"paths": [str(first), str(second)]}
    elif oracle == "glb":
        artifact = case_dir / "terrain.glb"
        if not artifact.is_file():
            fails.append("no GLB")
        else:
            metrics, more = check_glb(artifact)
            fails += more
            record["artifact"] = {"path": str(artifact), "sha256": sha256(artifact),
                                  "bytes": artifact.stat().st_size}
    elif oracle == "monitors":
        metrics, more = check_monitors(stdout)
        fails += more
    elif oracle == "stdout":
        metrics, more = check_stdout(stdout, ["Hello World", "Shutting down"])
        fails += more
    elif oracle == "diagnostic":
        metrics, more = check_diagnostic(stdout)
        fails += more
    elif oracle == "headless":
        metrics, more = check_headless(case_dir / "stream" / "result.json", stdout)
        fails += more
        for png in sorted((case_dir / "stream").glob("frame-*.png"))[:2]:
            record.setdefault("artifacts", []).append({"path": str(png), "sha256": sha256(png)})
    elif oracle == "vehicle":
        metrics, more = check_vehicle(stdout)
        fails += more

    if fails and name in ATTRIBUTION:
        record["attribution"] = ATTRIBUTION[name]
    record["metrics"] = metrics
    record["fails"] = fails
    record["ok"] = not fails

    if probe and name in PROBES:
        spec = probe_spec(probe, script, case_dir / "probe")
        probe_record = run_probe(spec, python, corpus, env, 240, case_dir)
        record["probe"] = {
            "ok": not probe_record["fails"],
            "fails": probe_record["fails"],
            "exit": probe_record["exit"],
            "seconds": probe_record["seconds"],
            "diff": probe_record.get("result", {}).get("diff"),
            "metrics": [
                (rail or {}).get("metrics") for rail in probe_record.get("result", {}).get("rails", [])
            ],
        }
        if probe_record["fails"]:
            record["fails"] = record["fails"] + [f"probe: {f}" for f in probe_record["fails"]]
            record["ok"] = False
    if probe == "quad":
        trace = case_dir / "lod.csv"
        lines = len(trace.read_text().splitlines()) if trace.is_file() else 0
        record["metrics"]["trace_lines"] = lines
        if lines < 1:
            record["fails"].append(f"LOD trace has {lines} lines")
            record["ok"] = False
    if probe == "diag_control":
        control = run_command(
            command + ["--control-drop-first"],
            corpus,
            env,
            120,
            case_dir / "control-drop-first.log",
        )
        record["control_drop_first"] = control
        if control["exit"] in (0, None):
            record["fails"].append("dropping the first native submission still passed")
            record["ok"] = False
    return record


def controls(corpus, out, python, env, site):
    results = {}
    root = out / "controls"
    root.mkdir(parents=True, exist_ok=True)

    # 1. A scene with no entities must look empty to the scene-aware oracle,
    #    while the same scene with one cube must not: compare both to the blank.
    def still(name, with_cube):
        target = root / name
        spawn = (
            "e.spawn(mx.components.Mesh(mx.geometry.cube(1,1,1)),"
            "mx.components.Material(mx.material.phong(mx.colors.RED)),"
            "mx.components.Transform(pos=(0,0,0)));"
            if with_cube
            else ""
        )
        program = (
            "import manifoldx as mx;"
            "e=mx.Engine('still',width=160,height=120);"
            + spawn
            + "i=e.render_frame(supersample=1);"
            "from PIL import Image;"
            f"Image.fromarray(i).save({str(target)!r})"
        )
        return run_command([python, "-I", "-c", program], out, env, 120,
                           root / f"{name}.log"), target

    import numpy as np
    from PIL import Image

    blank_run, blank_png = still("blank.png", False)
    cube_run, cube_png = still("cube.png", True)
    if blank_png.is_file() and cube_png.is_file():
        blank = np.asarray(Image.open(blank_png).convert("RGB")).astype(np.int16)
        cube = np.asarray(Image.open(cube_png).convert("RGB")).astype(np.int16)
        empty_vs_empty = round(float(np.abs(blank - blank).mean()), 4)
        foreground = round(float(np.abs(cube - blank).mean()), 4)
        results["empty_png"] = {
            "control": "an entity-free scene compared with the same scene holding one cube",
            "detected": empty_vs_empty < 0.05 and foreground >= 0.3,
            "metrics": {"empty_vs_empty": empty_vs_empty, "foreground": foreground},
            "blank_exit": blank_run["exit"],
            "cube_exit": cube_run["exit"],
            "limit": "a uniform-colour still is not used: the engine always draws its sky and ground",
        }
    else:
        results["empty_png"] = {
            "control": "an entity-free scene compared with the same scene holding one cube",
            "detected": False,
            "fails": ["no PNG from the blank or cube scene"],
        }

    # 2. A truncated GLB must fail the GLB parser.
    source = out / "cases" / "gltf_export" / "terrain.glb"
    truncated = root / "terrain-truncated.glb"
    if source.is_file():
        raw = source.read_bytes()
        truncated.write_bytes(raw[: max(64, int(len(raw) * 0.6))])
        metrics, fails = check_glb(truncated)
    else:
        metrics, fails = {}, ["the export case wrote no GLB to truncate"]
    results["truncated_glb"] = {
        "control": "truncate the exported GLB to 60%",
        "detected": bool(fails),
        "fails": fails,
        "metrics": metrics,
    }

    # 3. event_dolly with the startup dolly event removed must stop changing.
    script = write_control_script(
        corpus / "examples/event_dolly.py",
        root / "event_dolly_no_event.py",
        [('    engine.emit("dolly")\n', "")],
    )
    spec = probe_spec("event_dolly", script, root / "event_dolly_no_event")
    probe = run_probe(spec, python, corpus, env, 240, root)
    results["missing_event_dolly"] = {
        "control": "the startup dolly event is removed",
        "detected": bool(probe["fails"]),
        "fails": probe["fails"],
        "diff": probe.get("result", {}).get("diff"),
    }

    # 4. event_pulse with the pulse emission removed must go static after 1 s.
    script = write_control_script(
        corpus / "examples/event_pulse.py",
        root / "event_pulse_no_pulse.py",
        [('        engine.emit("pulse", {"index": _state["count"]})\n', "        pass\n")],
    )
    video = root / "event_pulse_no_pulse.mp4"
    run = run_command(
        [python, "-I", str(script), "--render", "--duration", "2.4", "--fps", "6",
         "--output", str(video)],
        corpus, env, 120, root / "event_pulse_no_pulse.log",
    )
    metrics = decode_video(video) if video.is_file() else {"error": "no MP4"}
    fails = check_video(metrics, motion=True)
    results["missing_event_pulse"] = {
        "control": "the periodic pulse emission is removed",
        "detected": bool(fails),
        "fails": fails,
        "metrics": metrics,
        "exit": run["exit"],
    }

    # 5. Without the extension the runtime diagnostic must refuse to start.
    shadow = root / "no-extension"
    if (site / "manifoldx").is_dir():
        shutil.copytree(
            site / "manifoldx", shadow / "manifoldx",
            ignore=shutil.ignore_patterns("_core_native*.so", "__pycache__"),
        )
        shadow_env = env | {"PYTHONPATH": str(shadow)}
        run = run_command(
            [python, str(corpus / "examples/native_runtime.py"), "--deterministic-only"],
            corpus, shadow_env, 120, root / "no_extension.log",
        )
        log = (root / "no_extension.log").read_text(errors="replace")
        reasons = ("compiled extension required", "cannot import name '_core_native'")
        results["missing_extension"] = {
            "control": "the compiled extension is removed from a shadow package on PYTHONPATH",
            "detected": run["exit"] not in (0, None) and any(r in log for r in reasons),
            "exit": run["exit"],
            "reason": next((r for r in reasons if r in log), None),
            "limit": "the shadow import is a copy, not the installed tree, because -I must be dropped",
            "tail": log.strip().splitlines()[-3:],
        }
        shutil.rmtree(shadow, ignore_errors=True)
    else:
        results["missing_extension"] = {
            "control": "the compiled extension is removed from a shadow package on PYTHONPATH",
            "detected": False,
            "fails": [f"could not find site-packages beside {python}"],
        }
    return results


# What each failing case is missing, from the traceback this run recorded. This
# is attribution to the revision under test (0bad545), not to any earlier one:
# no historical revision was run.
ATTRIBUTION = {
    "examples/axes.py": "the native wrapper has no _geometry_registry; the axes example looks up geometry by name",
    "examples/event_pulse.py": "the native loop dispatches no \'frame\' event in run() or render(); its documented cubes never spawn",
    "examples/ffd.py": "the native entity set_geometry does not accept the modeling Mesh value object",
    "examples/gui.py": "the native wrapper exposes no _last_dt, which the example\'s FPS getter reads",
    "examples/ibl.py": "Camera.position/target are tuples and _camera_input calls .tolist()",
    "examples/point_cloud.py": "PointCloud registers lazily on its first spawn, after the native world published entities",
    "examples/point_cloud_compute.py": "PointCloud registers lazily on its first spawn, after the native world published entities",
    "examples/scatter_plot.py": "the viz shim reads engine.store, which the native engine does not expose",
}

FIRST_REPAIR = {
    "case": "examples/event_pulse.py",
    "defect": "the installed wheel\'s native loop never dispatches the inline \'frame\' event, in run() or in render()",
    "witness": "event_pulse spawns a cube from a frame handler; its 4.4 s clip has 26 identical frames",
    "repair": "queue (\"frame\", {\"dt\", \"elapsed\", \"frame\"}) at the head of the event bus before _pump_events, in _desktop_begin_frame and render_frame, as Engine._draw_frame does",
    "patch": "frame-event-repair.patch",
    "verified": {
        "fix": "applied temporarily to site-packages/manifoldx/_desktop.py, then restored and its sha256 checked",
        "frame_events_in_run": 5,
        "clip": "4.4 s / 26 frames, first/last mean_abs 0.22, changed fraction 0.0028, no longer 0.0",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, help="the venv python holding the wheel")
    parser.add_argument("--wheel", required=True, help="the exact wheel that venv installed")
    parser.add_argument("--corpus", required=True, help="an isolated copy of examples/ and demos/")
    parser.add_argument("--out", required=True, help="where per-case evidence is written")
    parser.add_argument("--only", help="comma-separated case stems to run")
    parser.add_argument("--no-probe", action="store_true", help="skip the interaction lane")
    parser.add_argument("--xvfb", default="Xvfb")
    args = parser.parse_args()

    python = str(Path(args.python).absolute())
    wheel = Path(args.wheel).resolve()
    corpus = Path(args.corpus).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    missing = [name for name, *_ in CASES if not (corpus / name).is_file()]
    if missing:
        raise SystemExit(f"corpus is missing {missing}")

    info = identity(python, wheel)
    if not info["installed_from_site_packages"]:
        raise SystemExit(f"refusing to run the corpus against a checkout: {info['manifoldx']}")

    display_number = free_display()
    display = f":{display_number}"
    env = clean_env(display, out / "tmp")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["UV_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [args.xvfb, display, "-screen", "0", "1280x720x24", "-nolisten", "tcp"],
        cwd=out,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while not Path(f"/tmp/.X11-unix/X{display_number}").exists():
        if server.poll() is not None or time.monotonic() > deadline:
            raise SystemExit(f"Xvfb did not start on {display}")
        time.sleep(0.1)

    selected = CASES
    if args.only:
        wanted = {part.strip() for part in args.only.split(",")}
        selected = [case for case in CASES if Path(case[0]).stem in wanted]
        if not selected:
            raise SystemExit(f"--only matched nothing: {sorted(wanted)}")

    report = {
        "gate": "GW-1 public corpus",
        "identity": info,
        "display": display,
        "corpus": str(corpus),
        "expected_total": EXPECTED_TOTAL,
        "manifest_total": len(CASES),
        "inventory_additions": [{"file": name, "note": note} for name, note in ADDITIONS],
        "excluded": [{"file": name, "note": note} for name, note in EXCLUDED],
        "cases": [],
        "controls": {},
        "first_repair": FIRST_REPAIR,
        "limits": {
            "clip": "1.2 s at 6 fps unless the manifest names another length",
            "timeouts_s": {"entry": 120, "headless_or_vehicle": 240, "probe": 240},
            "display": "private Xvfb 1280x720; the physical DISPLAY and WAYLAND_DISPLAY are cleared",
            "window_capture": "none: hello_world is judged by its stdout and vehicle by its fixed-step report",
            "no_publication": "no push, PR comment, merge, spec or tracker edit was made",
            "artifacts": "raw commands, exits, logs, metrics and hashes stay under the --out directory",
        },
        "platform_matrix": {
            "lane_run": "linux-x86_64 cp314t (free-threaded), installed wheel",
            "linux_x86_64_cp314t": "run",
            "linux_x86_64_gil": "not run",
            "windows": "not run",
            "macos": "not run",
        },
    }
    try:
        for index, case in enumerate(selected, 1):
            started = time.monotonic()
            print(f"[{index}/{len(selected)}] {case[0]} ...", flush=True)
            record = run_case(case, corpus, out, python, env)
            report["cases"].append(record)
            print(
                f"[{index}/{len(selected)}] {case[0]} {'ok' if record['ok'] else 'FAIL'} "
                f"({round(time.monotonic() - started, 1)}s)",
                flush=True,
            )
        if not args.no_probe:
            report["controls"] = controls(
                corpus, out, python, env, Path(info["manifoldx"]).parent.parent
            )
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    cases = report["cases"]
    report["summary"] = {
        "cases": len(cases),
        "passed": sum(1 for case in cases if case["ok"]),
        "failed": [case["name"] for case in cases if not case["ok"]],
        "controls_detected": sum(
            1 for value in report["controls"].values() if value.get("detected")
        ),
        "controls_total": len(report["controls"]),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0 if not report["summary"]["failed"] and report["summary"]["cases"] == EXPECTED_TOTAL else 1


if __name__ == "__main__":
    raise SystemExit(main())
