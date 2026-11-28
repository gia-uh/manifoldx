#!/usr/bin/env python3
"""One corpus case, driven offscreen through its own module.

The corpus gate runs this in a fresh interpreter per case so a crash or a
stuck GPU device cannot take the rest of the run with it. It imports the
copied example or demo with ``runpy`` under ``run_name="corpus_probe"``, so
the file's ``if __name__ == "__main__":`` block never runs: the module-level
engine, entities and handlers are the ones the script itself builds.

Input arrives through ``Engine._input_bridge._on_event`` with the same
rendercanvas dictionaries a window would deliver, so ``engine.input`` polling
and the ``key_down`` / ``pointer_*`` / ``wheel`` handlers see a real event.
Bus-level widget events (``gui:slider:G:change``) are emitted with
``engine.emit``.

The spec is a JSON file:

    {
      "script": "examples/input_fly.py",
      "out": "/var/tmp/run/input_fly",
      "dt": 0.0333333,
      "rails": [
        {"frames": 2, "save": "before.png", "snapshot": true},
        {"events": [{"kind": "key", "key": "w", "down": true}],
         "frames": 20, "save": "after.png", "snapshot": true}
      ]
    }

It prints one JSON object on stdout with the snapshots, image metrics and
hashes, and exits 0 even when a check fails: the gate decides what a number
means.

An ``axes`` block instead (or as well) freezes the example's orbit at one
pose, renders one offscreen still and reports the projected axis segments and
tip labels. Its ``whole_frame`` palette counts are context; the gate decides
from the projected bands and tip windows only.
"""

import hashlib
import json
import runpy
import sys
import traceback
from pathlib import Path

import numpy as np


def plain(value):
    """A JSON-safe copy of engine state that may hold numpy arrays."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def snapshot(engine, namespace, wanted=()):
    camera = getattr(engine, "camera", None)
    snap = {
        "elapsed": float(getattr(engine, "elapsed", 0.0)),
        "camera_position": plain(getattr(camera, "position", None)),
        "camera_target": plain(getattr(camera, "target", None)),
        "gui_scale": float(getattr(getattr(engine, "gui", None), "scale", 0.0)),
        "car": plain(getattr(engine, "state", {}).get("car"))
        if hasattr(getattr(engine, "state", None), "get")
        else None,
    }
    for name in wanted:
        snap[f"ns:{name}"] = plain(namespace.get(name))
    return snap


def image_metrics(image, path):
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    if path is not None:
        from PIL import Image as PILImage

        PILImage.fromarray(image).save(path)
    return {
        "shape": list(image.shape),
        "mean": round(float(image.mean()), 4),
        "std": round(float(image.std()), 4),
        "max": int(image.max()),
        "nonblank_fraction": round(float((image > 4).mean()), 4),
        "sha256": digest,
        "file": None if path is None else str(path),
    }


def diff(a, b):
    a = np.asarray(a, dtype=np.int16)
    b = np.asarray(b, dtype=np.int16)
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    delta = np.abs(a - b)
    return {
        "mean_abs": round(float(delta.mean()), 4),
        "changed_fraction": round(float((delta.max(axis=-1) > 8).mean()), 4),
        "max": int(delta.max()),
    }


# --- axes scene oracle ------------------------------------------------------
# The axes example documents three world lines (X red, Y green, Z blue) and a
# world-anchored label just past each tip, plus screen-anchored HUD, legend and
# scale-bar labels. These constants are that documented palette and the
# directions its loop spawns. A colour found along the wrong projected
# direction, or a label missing from its tip, is a failure, no matter what the
# whole frame looks like: the screen-anchored overlays are never credited here.

AXES_PALETTE = {"x": (230, 69, 69), "y": (95, 191, 95), "z": (85, 136, 255)}
AXES_DIRECTIONS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def project(engine, points, width, height):
    pixels, _ = engine.camera.project(list(points), width, height)
    return np.asarray(pixels, dtype=float)


def segment_mask(shape, start, end, radius):
    """Pixels whose centre lies within `radius` of the projected segment."""
    height, width = shape
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = dx * dx + dy * dy
    ys, xs = np.mgrid[0:height, 0:width]
    if length == 0.0:
        return (xs - x0) ** 2 + (ys - y0) ** 2 <= radius * radius
    t = ((xs - x0) * dx + (ys - y0) * dy) / length
    t = np.clip(t, 0.0, 1.0)
    nearest_x = x0 + t * dx
    nearest_y = y0 + t * dy
    return (xs - nearest_x) ** 2 + (ys - nearest_y) ** 2 <= radius * radius


def axes_measure(engine, namespace, image, path):
    """Project the scene's axes and measure colour and label anchors.

    Runs after the camera is frozen at a known pose, so the numbers describe
    the scene and not the orbit. Every decision below comes from a projected
    band or tip region; whole-frame counts are recorded only as context.
    """
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    image = image.astype(int)
    height, width = image.shape[:2]
    extent = float(namespace.get("EXTENT", 3.0))
    offset = float(namespace.get("LABEL_OFFSET", extent + 0.4))
    text_roi = (95, 35)

    result = {
        "pose": {
            "position": plain(getattr(getattr(engine, "camera", None), "position", None)),
            "target": plain(getattr(getattr(engine, "camera", None), "target", None)),
        },
        "extent": extent,
        "label_offset": offset,
        "shape": [height, width],
        "whole_frame": {},
        "axes": {},
        "labels": {},
        "gui_excluded": None,
    }
    for key, colour in AXES_PALETTE.items():
        result["whole_frame"][key] = int(np.all(image == np.array(colour), axis=-1).sum())

    origin = project(engine, [(0.0, 0.0, 0.0)], width, height)[0]
    for key, direction in AXES_DIRECTIONS.items():
        colour = np.array(AXES_PALETTE[key])
        negative_point = tuple(-extent * value for value in direction)
        tip_point = tuple(extent * value for value in direction)
        label_point = tuple(offset * value for value in direction)
        negative, tip, label = project(
            engine, [negative_point, tip_point, label_point], width, height
        )

        # The example stretches a unit line to ±EXTENT through the origin, so
        # the required line is the whole projected segment. Sample away from
        # both ends and from the sphere at the origin; the label sits past the
        # positive tip and is checked separately.
        sphere = float(namespace.get("SPHERE_RADIUS", 0.0))
        inner = 0.5 * sphere / max(extent, 1e-6) + 0.01
        step = np.concatenate(
            (np.linspace(0.12, 0.5 - inner, 80), np.linspace(0.5 + inner, 0.88, 80))
        )
        points = negative[None, :] * (1 - step[:, None]) + tip[None, :] * step[:, None]
        hits = {name: 0 for name in AXES_PALETTE}
        for x, y in points:
            x0, x1 = max(0, int(round(x)) - 2), min(width, int(round(x)) + 3)
            y0, y1 = max(0, int(round(y)) - 2), min(height, int(round(y)) + 3)
            window = image[y0:y1, x0:x1]
            for name, rgb in AXES_PALETTE.items():
                if np.any(np.all(window == np.array(rgb), axis=-1)):
                    hits[name] += 1

        band = segment_mask((height, width), negative, tip, 3)
        exact = np.all(image == colour, axis=-1)
        band_pixels = int((exact & band).sum())
        outside_pixels = int((exact & ~band).sum())
        result["axes"][key] = {
            "negative": [round(float(value), 2) for value in negative],
            "origin": [round(float(value), 2) for value in origin],
            "tip": [round(float(value), 2) for value in tip],
            "samples": len(step),
            "hits": hits,
            "own_fraction": round(hits[key] / len(step), 4),
            "band_pixels": band_pixels,
            "outside_band_pixels": outside_pixels,
        }

        # The label must be text at the tip its Transform anchors it to. A
        # missing or displaced label leaves this window without text.
        x, y = label
        x0, x1 = max(0, int(x) - text_roi[0]), min(width, int(x) + text_roi[0])
        y0, y1 = max(0, int(y) - text_roi[1]), min(height, int(y) + text_roi[1])
        window = image[y0:y1, x0:x1]
        neutral = window.min(axis=-1) > 150
        neutral &= (window.max(axis=-1) - window.min(axis=-1)) < 60
        rows, cols = np.nonzero(neutral)
        count = int(neutral.sum())
        centroid = None
        offset_px = None
        if count:
            centroid = [round(float(x0 + cols.mean()), 2), round(float(y0 + rows.mean()), 2)]
            offset_px = round(float(np.hypot(centroid[0] - x, centroid[1] - y)), 2)
        result["labels"][key] = {
            "projection": [round(float(value), 2) for value in label],
            "text_pixels": count,
            "centroid": centroid,
            "offset_px": offset_px,
        }

    offsets = [value["offset_px"] for value in result["labels"].values() if value["offset_px"] is not None]
    result["label_offset_spread"] = round(max(offsets) - min(offsets), 2) if len(offsets) == 3 else None
    result["gui_excluded"] = all(
        axis["outside_band_pixels"] == 0 for axis in result["axes"].values()
    )
    if path is not None:
        from PIL import Image as PILImage

        PILImage.fromarray(image.astype("u1")).save(path)
    return result


def apply_event(engine, event):
    kind = event["kind"]
    if kind == "bus":
        engine.emit(event["name"], event.get("payload"))
        return
    native = hasattr(engine, "_window_events")
    if kind == "key":
        down = event.get("down", True)
        if native:
            engine._window_events(
                [("key_down" if down else "key_up", event["key"], None, 0,
                  tuple(event.get("modifiers", ())))]
            )
        else:
            engine._input_bridge._on_event(
                {
                    "event_type": "key_down" if down else "key_up",
                    "key": event["key"],
                    "modifiers": tuple(event.get("modifiers", ())),
                }
            )
        return
    if kind == "pointer":
        phase = event.get("phase", "move")
        if native:
            engine._window_events(
                [
                    (
                        f"pointer_{phase}",
                        None,
                        (float(event.get("x", 0.0)), float(event.get("y", 0.0)), 0.0, 0.0),
                        int(event.get("button", 1)),
                        tuple(event.get("modifiers", ())),
                    )
                ]
            )
        else:
            engine._input_bridge._on_event(
                {
                    "event_type": f"pointer_{phase}",
                    "x": float(event.get("x", 0.0)),
                    "y": float(event.get("y", 0.0)),
                    "button": int(event.get("button", 1)),
                    "buttons": tuple(event.get("buttons", ())),
                    "modifiers": tuple(event.get("modifiers", ())),
                    "ntouches": 0,
                    "touches": {},
                }
            )
        return
    if kind == "wheel":
        if native:
            engine._window_events(
                [
                    (
                        "wheel",
                        None,
                        (
                            float(event.get("x", 0.0)),
                            float(event.get("y", 0.0)),
                            float(event.get("dx", 0.0)),
                            float(event.get("dy", 0.0)),
                        ),
                        0,
                        tuple(event.get("modifiers", ())),
                    )
                ]
            )
        else:
            engine._input_bridge._on_event(
                {
                    "event_type": "wheel",
                    "dx": float(event.get("dx", 0.0)),
                    "dy": float(event.get("dy", 0.0)),
                    "x": float(event.get("x", 0.0)),
                    "y": float(event.get("y", 0.0)),
                    "buttons": tuple(event.get("buttons", ())),
                    "modifiers": tuple(event.get("modifiers", ())),
                }
            )
        return
    raise ValueError(f"unknown event kind {kind!r}")


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    dt = float(spec.get("dt", 1 / 30))
    result = {"script": spec["script"], "rails": [], "ok": False}
    try:
        namespace = runpy.run_path(spec["script"], run_name="corpus_probe")
        engine = namespace["engine"]
        result["module_imported"] = True
        axes_spec = spec.get("axes")
        if axes_spec:
            # Freeze the example's orbit at one pose so the projection oracle
            # measures the scene instead of the camera.
            engine.camera.orbit = lambda *a, **k: None
            engine.camera.set_pose(
                position=tuple(axes_spec.get("position", (6.0, 4.0, 6.0))),
                target=tuple(axes_spec.get("target", (0.0, 0.0, 0.0))),
            )
            still = engine.render_frame(supersample=1)
            result["axes"] = axes_measure(
                engine, namespace, still, out / axes_spec.get("save", "axes.png")
            )
        rails = []
        images = []
        wanted = tuple(spec.get("namespace", ()))
        for index, rail in enumerate(spec.get("rails", [])):
            for event in rail.get("events", []):
                apply_event(engine, event)
            if rail.get("clear"):
                engine.clear_scene()
            image = None
            for _ in range(int(rail.get("frames", 1))):
                image = engine.render_frame(dt=dt)
            rails.append(snapshot(engine, namespace, wanted) if rail.get("snapshot", True) else None)
            save = rail.get("save")
            path = out / save if save else None
            metrics = image_metrics(image, path)
            images.append(np.asarray(image))
            result["rails"].append({"metrics": metrics, "snapshot": rails[-1]})
        pair = spec.get("diff")
        if pair is not None:
            result["diff"] = diff(images[pair[0]], images[pair[1]])
        result["ok"] = True
    except Exception:  # noqa: BLE001  reported, not raised: the gate owns the verdict
        result["error"] = traceback.format_exc(limit=6)
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
