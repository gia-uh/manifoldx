"""Replay: run one script of input frames through an app in Python and through its web export
in Node, trace the same values on both sides, and report the first frame where they differ.

A frame is what engine.input reports for one frame:

    {"keys": [str], "just": [str], "d": [dx, dy], "buttons": [int], "clicks": [int], "vp": [w, h],
     "up": [str], "released": [int], "pos": [x, y], "wheel": [dx, dy], "mods": [str]}

keys and buttons are held, just and clicks pressed this frame, d the mouse motion and vp the
viewport size. The last five are optional and read as empty when absent: up and released are
the keys and buttons released this frame, pos the pointer position in CSS pixels from the
canvas's top-left, wheel the wheel's motion, mods the held modifiers among Alt, Control, Meta
and Shift, sorted. Each side reads one frame, then runs the app's systems once with a fixed dt and
traces:

    {"camera": {"position", "target"}, "gui": [str], "log": [...], "state": engine.state,
     "paths": {path: value}}

gui is what each ValueDisplay shows; log the ["material", entity, material_id] and
["capture", on] entries this frame added; a path ("rig.pos", "selected") starts at a global of
the app's main module or, when it has no such global, at a key of engine.state, and goes on
through attributes and dict keys.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manifoldx.input import _key_name

DT = 1 / 60

Frame = dict  # the frame form above


class ScriptInput:
    """engine.input fed one frame at a time, as js/manifoldx.js's ScriptInput is: the getters
    of InputState, answered from the frame's fields (keys, just, d, buttons, clicks, vp, and
    the optional up, released, pos, wheel and mods, empty when absent). capture_pointer always takes, as in a
    headless browser, and logs ["capture", on]; a new viewport size resizes the engine."""

    def __init__(self, engine, log: list) -> None:
        self._engine, self._log = engine, log
        self._keys: frozenset[str] = frozenset()
        self._just: frozenset[str] = frozenset()
        self._buttons: frozenset[int] = frozenset()
        self._clicks: frozenset[int] = frozenset()
        self._delta = (0.0, 0.0)
        self._up: frozenset[str] = frozenset()
        self._released: frozenset[int] = frozenset()
        self._pos = (0.0, 0.0)
        self._wheel = (0.0, 0.0)
        self._mods: tuple[str, ...] = ()
        self._viewport = (engine.w, engine.h)
        self._captured = False

    def set_frame(self, frame: Frame) -> None:
        self._keys = frozenset(map(_key_name, frame["keys"]))
        self._just = frozenset(map(_key_name, frame["just"]))
        self._buttons = frozenset(frame["buttons"])
        self._clicks = frozenset(frame["clicks"])
        self._delta = (float(frame["d"][0]), float(frame["d"][1]))
        self._up = frozenset(map(_key_name, frame.get("up", ())))
        self._released = frozenset(frame.get("released", ()))
        pos, wheel = frame.get("pos", (0, 0)), frame.get("wheel", (0, 0))
        self._pos = (float(pos[0]), float(pos[1]))
        self._wheel = (float(wheel[0]), float(wheel[1]))
        self._mods = tuple(frame.get("mods", ()))
        size = (int(frame["vp"][0]), int(frame["vp"][1]))
        if size != self._viewport:
            self._viewport = size
            self._engine.w, self._engine.h = size

    def is_pressed(self, key: str) -> bool:
        return _key_name(key) in self._keys

    def just_pressed(self, key: str) -> bool:
        return _key_name(key) in self._just

    def just_released(self, key: str) -> bool:
        return _key_name(key) in self._up

    @property
    def pressed_keys(self) -> frozenset[str]:
        return self._keys

    @property
    def modifiers(self) -> tuple[str, ...]:
        return self._mods

    @property
    def mouse_pos(self) -> tuple[float, float]:
        return self._pos

    @property
    def mouse_delta(self) -> tuple[float, float]:
        return self._delta

    @property
    def wheel_delta(self) -> tuple[float, float]:
        return self._wheel

    def is_mouse_pressed(self, button: int) -> bool:
        return button in self._buttons

    def just_mouse_pressed(self, button: int) -> bool:
        return button in self._clicks

    def just_mouse_released(self, button: int) -> bool:
        return button in self._released

    @property
    def pressed_buttons(self) -> tuple[int, ...]:
        return tuple(sorted(self._buttons))

    @property
    def viewport_size(self) -> tuple[int, int]:
        return self._viewport

    @property
    def pointer_captured(self) -> bool:
        return self._captured

    def capture_pointer(self, on: bool) -> bool:
        self._captured = bool(on)
        self._log.append(["capture", self._captured])
        return True


def replay_python(
    module: str | None,
    script: str | None,
    frames: list[Frame],
    paths: list[str],
    dt: float = DT,
    app_args: list[str] | None = None,
) -> list[dict]:
    """The app (a script, or a module as python -m runs it) started as export --web starts it,
    with app_args as its sys.argv[1:] (give the export's, so both take the same branch), then
    run through frames: one trace entry per frame, as JSON reads it back."""
    from .export import _as_main, capture_engine
    from .reach import find_roots

    engine, main = capture_engine(module, script, list(app_args or []))
    if engine is None:
        raise ValueError(f"{module or script} finished without calling engine.run()")
    with _as_main(main):
        if engine._device is None and engine._event_bus._handlers.get("startup"):
            engine._ensure_offscreen(1)  # startup runs as export --web runs it
        getters = [r.func for r in find_roots(engine) if r.kind == "getter"]
        log: list = []
        engine.input = ScriptInput(engine, log)
        set_material = engine.set_material

        def recorded(entity, material):
            set_material(entity, material)
            index = entity if isinstance(entity, (int, np.integer)) else entity.index
            log.append(["material", int(index), engine._material_registry.register(material)])

        engine.set_material = recorded
        trace = []
        for frame in frames:
            start = len(log)
            engine.input.set_frame(frame)
            engine.systems.run_all(engine, dt)
            cam = engine.camera
            entry = {
                "camera": {"position": cam.position, "target": cam.target},
                "gui": [v if isinstance(v, str) else str(v) for v in (g() for g in getters)],
                "log": log[start:],
                "state": engine.state,
                "paths": {p: _path(vars(main), engine.state, p) for p in paths},
            }
            trace.append(json.loads(json.dumps(entry, default=_plain)))
    return trace


def replay_node(out_dir: Path | str, frames: list[Frame], paths: list[str], dt: float = DT):
    """The folder export --web wrote, run through frames in Node by its replay.mjs."""
    done = subprocess.run(
        ["node", str(Path(out_dir) / "replay.mjs"), "--replay"],
        input=json.dumps({"frames": frames, "paths": paths, "dt": dt}),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if done.returncode != 0:
        raise RuntimeError(f"replay.mjs failed:\n{done.stderr}")
    return json.loads(done.stdout)


@dataclass(frozen=True)
class Mismatch:
    """The first difference: the frame, where in its trace entry, and the two values."""

    frame: int
    key: str
    left: object
    right: object

    def __str__(self) -> str:
        return f"frame {self.frame}, {self.key}: {self.left!r} != {self.right!r}"


# Python's math calls the C library and the browser has its own sin, cos, exp...: the same
# call may differ in the last bit (V8's sin and cos against glibc's: 62 of 2226 values in the
# web_walk replay). Two floats are equal within 2^-30 of the larger, 64 times finer than a
# float32 step, so a value that skipped float32 rounding still differs; and within 2^-40
# near zero, where cancellation leaves noise.
REL, ABS = 2.0**-30, 2.0**-40


def compare(a: list[dict], b: list[dict], rel: float = REL) -> Mismatch | None:
    """The first frame and key where traces a and b differ, or None. Numbers compare by
    value (JSON does not tell 1 from 1.0), never against a bool, and floats within rel of
    the larger (rel=0: bit for bit)."""
    for i, (x, y) in enumerate(zip(a, b)):
        found = _diff(x, y, "", rel)
        if found is not None:
            return Mismatch(i, *found)
    if len(a) != len(b):
        return Mismatch(min(len(a), len(b)), "frames", len(a), len(b))
    return None


def _diff(x, y, key: str, rel: float):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in list(x) + [k for k in y if k not in x]:
            if k not in x or k not in y:
                return (_join(key, k), x.get(k), y.get(k))
            found = _diff(x[k], y[k], _join(key, k), rel)
            if found is not None:
                return found
        return None
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return (key, x, y)
        for i, (u, v) in enumerate(zip(x, y)):
            found = _diff(u, v, f"{key}[{i}]", rel)
            if found is not None:
                return found
        return None
    if isinstance(x, bool) != isinstance(y, bool):
        return (key, x, y)
    if isinstance(x, float) or isinstance(y, float):
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if abs(x - y) <= max(rel * max(abs(x), abs(y)), ABS if rel else 0.0):
                return None
    if x != y:
        return (key, x, y)
    return None


def _join(key: str, k: str) -> str:
    return f"{key}.{k}" if key else k


def _path(globals_: dict, state: dict, path: str):
    head, *rest = path.split(".")
    v = globals_[head] if head in globals_ else state[head]
    for name in rest:
        v = v[name] if isinstance(v, dict) else getattr(v, name)
    return v


def _plain(v):
    """What json cannot write as is: numpy values and sets."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (set, frozenset)):
        return list(v)
    raise TypeError(f"a {type(v).__name__} in the trace cannot be written as JSON")
