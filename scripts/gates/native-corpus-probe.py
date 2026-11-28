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
        rails = []
        images = []
        wanted = tuple(spec.get("namespace", ()))
        for index, rail in enumerate(spec["rails"]):
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
