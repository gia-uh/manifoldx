"""The state the browser starts from: what setup left in the objects reached code reads.

snapshot(roots, reach, engine) walks from engine.state, the roots (a bound method's object, a
closure's cells and defaults) and the module globals reached code reads, and writes JSON:

    {"table": [entry, ...], "roots": [root, ...], "globals": {module: {name: value}},
     "state": value, "engine": {"w", "h", "origin": [x, z],
                                "camera": {"position", "target", "fov", "up", "near", "far"}}}

A root is {"kind", "label", "event", "components", "fn": value}, in the order find_roots gave
them; components names its query (a system's, or a handler's that takes one), else null.
"state" is engine.state as a dict entry; "engine" is what the runtime's createEngine takes.

A value is null, a bool, a number, a str, a tuple as a JSON array, or one of
    {"__ref": n}                    the table's entry n, so a shared object stays one
    {"__float": "nan" | "inf" | "-inf"}
    {"__engine": "Engine" | "input" | "camera" | "gui"}
    {"__material": id}              registered in the engine's material registry
    {"__entity": index}
    {"__geometry": id}              registered in the engine's geometry registry
    {"__widget": n}                 the n-th widget walking engine.gui's panels
    {"__function": "module:name"}   a module function reached code has
    {"__bound": value, "name": str} a method of the object `value`
    {"__closure": "module:line:col", "cells": {name: value}, "defaults": [value, ...]}

A table entry is a list (a JSON array), {"__dict": [[key, value], ...]} (str keys, in
order), {"__set": [value, ...]}, {"__f32": [...]} or {"__f64": [...]} (a 1-d float numpy
array), or {"__type": "module.Class", "fields": {name: value}} with only the fields
reached code reads.

ecs_bytes(engine) writes the entity store as the header build_export puts under "ecs" and
the bytes of ecs.bin:

    {"max_entities", "high_water", "free": [int], "alive": [offset, nbytes],
     "components": [{"name", "dtype": "f4" | "u4", "cols", "fields": {field: [col, size]},
                     "defaults": {field: number | [number]},
                     "rows": [offset, nbytes], "members": [offset, nbytes]}],
     "spawnable": [{"name", "dtype", "cols", "fields", "defaults"}]}

defaults are a Component subclass's class-level field defaults, as float32 values. build_export
adds "spawnable": the Component subclasses reached code constructs that the store has not
registered, which the page registers when it first spawns one, as Engine.spawn does.

ecs.bin holds the alive mask as uint8 for slots [0, high_water), then per component in the
store's registration order its rows [0, high_water) row-major little-endian in its dtype and
its membership as uint8. Every block starts on a multiple of 4.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np

from manifoldx.components import _COMPONENT_CLASSES
from manifoldx.gltf.engine_export import _widgets

from .reach import Reach, closure_site, is_closure
from .transpile import js_name
from .types import class_name, component_fields, is_geometry


class SnapshotError(Exception):
    """A value that cannot cross to the browser, at its path from the root."""

    def __init__(self, path_from_root: str, what: str):
        super().__init__(path_from_root, what)
        self.path_from_root, self.what = path_from_root, what

    def __str__(self) -> str:
        return f"{self.path_from_root}: {self.what} cannot cross to the browser"


def snapshot(roots, reach: Reach, engine, left_out: list | None = None) -> dict:
    """The JSON state of roots (reach.Root objects) and what they reach, and engine.state;
    registers every material it meets in the engine's material registry, so the glTF export
    writes it. A key of engine.state that cannot cross raises SnapshotError, or with
    left_out given is skipped and appended to it as the SnapshotError."""
    return _Writer(reach, engine).write(roots, left_out)


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


class _Writer:
    def __init__(self, reach: Reach, engine):
        from manifoldx.engine import EntityHandle
        from manifoldx.gui.widgets import Widget
        from manifoldx.resources import Material

        self.reach, self.engine = reach, engine
        self.table: list = []
        self.refs: dict[int, int] = {}
        self.keep: list = []  # what the table holds stays alive, so ids stay unique
        self.handles = {
            id(engine): "Engine",
            id(engine.input): "input",
            id(engine.camera): "camera",
            id(engine.gui): "gui",
        }
        walk = (w for panel in engine.gui._panels for w in _widgets(panel))
        self.widgets = {id(w): i for i, w in enumerate(walk)}
        self.Entity, self.Widget, self.Material = EntityHandle, Widget, Material

    def write(self, roots, left_out: list | None = None) -> dict:
        state = self.state(left_out)
        out = []
        for r in roots:
            path = type(r.owner).__name__ if r.owner is not None else ""
            comps = list(r.components) if r.components is not None else None
            fn = self.value(r.func, path)
            out.append(
                {"kind": r.kind, "label": r.label, "event": r.event, "components": comps, "fn": fn}
            )
        globals_: dict[str, dict] = {}
        if not self.reach.everything:
            for module, name in sorted(self.reach.globals):
                namespace = vars(sys.modules[module])
                if name in namespace:
                    value = self.value(namespace[name], f"{module}.{name}")
                    globals_.setdefault(module, {})[name] = value
        cam = self.engine.camera
        engine = {
            "w": int(self.engine.w),
            "h": int(self.engine.h),
            "origin": [float(self.engine._origin[0]), float(self.engine._origin[1])],
            "camera": {
                "position": [float(x) for x in cam.position],
                "target": [float(x) for x in cam.target],
                "fov": float(cam.fov),
                "up": [float(x) for x in cam.up],
                "near": float(cam.near),
                "far": float(cam.far),
            },
        }
        return {
            "table": self.table,
            "roots": out,
            "globals": globals_,
            "state": state,
            "engine": engine,
        }

    def state(self, left_out: list | None):
        """engine.state as a dict entry, written first so what roots share with it stays
        shared. A key that fails is rolled back out of the table when left_out is given."""
        state = self.engine.state
        n = len(self.table)
        self.table.append(None)
        self.refs[id(state)] = n
        self.keep.append(state)
        pairs = []
        for k, x in state.items():
            path = f"engine.state[{k!r}]"
            mark = len(self.table)
            try:
                if not isinstance(k, str):
                    raise SnapshotError(path, f"a {type(k).__name__} key")
                pairs.append([k, self.value(x, path)])
            except SnapshotError as e:
                if left_out is None:
                    raise
                left_out.append(e)
                del self.table[mark:]
                self.refs = {i: j for i, j in self.refs.items() if j < mark}
        self.table[n] = {"__dict": pairs}
        return {"__ref": n}

    # ---------------------------------------------------------- values
    def value(self, v, path: str):
        if v is None or isinstance(v, str):
            return v
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            if abs(int(v)) > 2**53:
                raise SnapshotError(path, f"the int {int(v)}, beyond a double's exact integers,")
            return int(v)
        if isinstance(v, (float, np.floating)):
            return _float(float(v))
        if isinstance(v, tuple):
            return [self.value(x, f"{path}[{i}]") for i, x in enumerate(v)]
        if isinstance(v, (types.FunctionType, types.MethodType)):
            return self.function(v, path)
        if id(v) in self.handles:
            return {"__engine": self.handles[id(v)]}
        if isinstance(v, self.Material):
            return {"__material": self.engine._material_registry.register(v)}
        if isinstance(v, self.Entity):
            return {"__entity": int(v.index)}
        if isinstance(v, self.Widget):
            if id(v) not in self.widgets:
                raise SnapshotError(path, "a GUI widget that is not in the engine's GUI")
            return {"__widget": self.widgets[id(v)]}
        geometries = self.engine._geometry_registry
        if id(v) in geometries._object_to_id or is_geometry(v):
            return {"__geometry": geometries.register(v)}
        if id(v) in self.refs:
            return {"__ref": self.refs[id(v)]}
        if isinstance(v, (list, dict, set, frozenset, np.ndarray)) or self.app_object(v, path):
            n = len(self.table)
            self.table.append(None)
            self.refs[id(v)] = n
            self.keep.append(v)
            self.table[n] = self.entry(v, path)
            return {"__ref": n}
        raise SnapshotError(path, f"a {class_name(type(v))}, a library object,")

    def app_object(self, v, path: str) -> bool:
        name = class_name(type(v))
        if name not in self.reach.env.classes:
            return False
        if not self.reach.class_(name):
            raise SnapshotError(path, f"an object of {name}, a class reached code does not use,")
        return True

    def entry(self, v, path: str):
        if isinstance(v, list):
            return [self.value(x, f"{path}[{i}]") for i, x in enumerate(v)]
        if isinstance(v, dict):
            pairs = []
            for k, x in v.items():
                if not isinstance(k, str):
                    raise SnapshotError(path, f"a dict with the {type(k).__name__} key {k!r}")
                pairs.append([k, self.value(x, f"{path}[{k!r}]")])
            return {"__dict": pairs}
        if isinstance(v, (set, frozenset)):
            return {"__set": [self.value(x, f"{path}{{}}") for x in v]}
        if isinstance(v, np.ndarray):
            return self.array(v, path)
        name = class_name(type(v))
        own = vars(v) if hasattr(v, "__dict__") else {}
        fields = {
            f: self.value(own[f], _join(path, f))
            for f in sorted(self.reach.fields.get(name, ()))
            if f in own
        }
        return {"__type": name, "fields": fields}

    def array(self, a: np.ndarray, path: str):
        if a.ndim != 1 or a.dtype not in (np.float32, np.float64):
            raise SnapshotError(path, f"a numpy array of shape {a.shape} and dtype {a.dtype}")
        if a.base is not None:
            raise SnapshotError(path, "a numpy view of another array (copy it at setup)")
        items = [_float(x) for x in a.tolist()]
        return {"__f32" if a.dtype == np.float32 else "__f64": items}

    # ---------------------------------------------------------- functions
    def function(self, f, path: str):
        reach = self.reach
        key = reach.key_of(f)
        if key is None or not reach.function(key):
            what = getattr(f, "__qualname__", repr(f))
            raise SnapshotError(path, f"the function {what}, which is not reached code,")
        if isinstance(f, types.MethodType):
            return {"__bound": self.value(f.__self__, path), "name": f.__func__.__name__}
        if not is_closure(key):
            return {"__function": f"{key[0]}:{js_name(key[1])}"}
        line, col = closure_site(reach.env, key)
        cells = {}
        for name, cell in zip(f.__code__.co_freevars, f.__closure__ or ()):
            try:
                contents = cell.cell_contents
            except ValueError:
                raise SnapshotError(_join(path, name), "an empty closure cell") from None
            cells[name] = self.value(contents, _join(path, name))
        defaults = [
            self.value(d, f"{path}<default {i}>") for i, d in enumerate(f.__defaults__ or ())
        ]
        return {"__closure": f"{key[0]}:{line}:{col}", "cells": cells, "defaults": defaults}


_DTYPES = {np.dtype("f4"): "f4", np.dtype("u4"): "u4"}


def ecs_bytes(engine) -> tuple[dict, bytes]:
    """The entity store as ecs.bin's bytes and their header (the module docstring's form);
    raises SnapshotError for a component of a dtype other than float32 or uint32."""
    store = engine.store
    hw = store._high_water
    data = bytearray()

    def block(raw: bytes) -> list[int]:
        data.extend(bytes(-len(data) % 4))
        start = len(data)
        data.extend(raw)
        return [start, len(raw)]

    alive = block(store._alive[:hw].astype(np.uint8).tobytes())
    components = []
    for name, array in store._components.items():
        code = _DTYPES.get(array.dtype)
        if code is None:
            raise SnapshotError(
                f"the component {name}",
                f"dtype {array.dtype}; ecs.bin carries float32 and uint32 and it",
            )
        cols = int(np.prod(array.shape[1:]))
        rows = array[:hw].reshape(hw, cols).astype(array.dtype.newbyteorder("<"))
        components.append(
            {
                "name": name,
                "dtype": code,
                "cols": cols,
                "fields": {f: [int(c), int(n)] for f, (c, n) in component_fields(name).items()},
                "defaults": _defaults(_COMPONENT_CLASSES.get(name)),
                "rows": block(rows.tobytes()),
                "members": block(store._members[name][:hw].astype(np.uint8).tobytes()),
            }
        )
    header = {
        "max_entities": int(store.max_entities),
        "high_water": int(hw),
        "free": [int(i) for i in store._free_list],
        "alive": alive,
        "components": components,
    }
    return header, bytes(data)


def spawnable(engine, components: dict[str, type]) -> list[dict]:
    """The layouts of the Component subclasses in components ({store name: class}) that the
    engine's store has not registered, as ecs_bytes writes a component's."""
    out = []
    for name, cls in components.items():
        if name in engine.store._components:
            continue
        out.append(
            {
                "name": name,
                "dtype": _DTYPES[np.dtype(cls._dtype)],
                "cols": int(cls._shape[0]),
                "fields": {
                    f: [int(c), int(n)] for f, (c, n) in component_fields(name, cls).items()
                },
                "defaults": _defaults(cls),
            }
        )
    return out


def _defaults(cls) -> dict:
    """A Component subclass's class-level field defaults as float32 values (none for others)."""
    found = getattr(cls, "_field_defaults", None) or {}
    return {f: np.asarray(v, np.float32).tolist() for f, v in found.items()}


def _float(x: float):
    if math.isfinite(x):
        return x
    return {"__float": "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")}
