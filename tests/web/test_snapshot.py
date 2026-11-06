"""The state snapshot: setup's objects written in Python, loaded in Node by loadState, and
the transpiled roots run on them."""

import ast
import importlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import manifoldx as mx
from manifoldx.web import SnapshotError, snapshot, transpile_app
from manifoldx.web.reach import Reach, find_roots
from manifoldx.web.types import LiveTypes, TypeEnv

JS = Path(__file__).parents[2] / "src" / "manifoldx" / "web" / "js"

# A resolver for the handles, without the runtime's engine: the engine records
# set_material calls, a material is its id, an entity its index.
RESOLVE = """(() => {
  const log = [];
  const engine = { log, set_material: (e, m) => log.push([e.index, m.id]) };
  const cache = new Map();
  return (kind, v) => {
    if (kind === "engine") return engine;
    const key = `${kind}:${v}`;
    if (!cache.has(key)) cache.set(key, kind === "material" ? { id: v } : { kind, index: v });
    return cache.get(key);
  };
})()"""

REPLACER = """(k, v) => {
  if (ArrayBuffer.isView(v)) return Array.from(v);
  if (v instanceof Set) return [...v].sort();
  if (v instanceof Map) return Object.fromEntries(v);
  return v;
}"""


@pytest.fixture
def app(tmp_path):
    """app(sources, setup) imports sources as modules, runs setup(modules, engine), takes
    the snapshot and transpiles what the roots reach. Returns an App."""
    names = []
    sys.path.insert(0, str(tmp_path))

    def _app(sources: dict[str, str], setup):
        for name, src in sources.items():
            (tmp_path / f"{name}.py").write_text(src)
            sys.modules.pop(name, None)
            names.append(name)
        importlib.invalidate_caches()
        mods = {name: importlib.import_module(name) for name in sources}
        engine = mx.Engine("t", width=64, height=64)
        setup(mods, engine)
        return App(tmp_path, sources, mods, engine)

    yield _app
    sys.path.remove(str(tmp_path))
    for name in names:
        sys.modules.pop(name, None)


class App:
    def __init__(self, folder: Path, sources, mods, engine):
        self.folder, self.sources, self.mods, self.engine = folder, sources, mods, engine
        self.roots = find_roots(engine)
        trees = {name: ast.parse(src) for name, src in sources.items()}
        self.reach = Reach(self.roots, TypeEnv(trees, LiveTypes(set(sources))))

    def snapshot(self) -> dict:
        snap = snapshot(self.roots, self.reach, self.engine)
        # What crosses is JSON: no NaN, nothing json cannot write.
        return json.loads(json.dumps(snap, allow_nan=False))

    def node(self, view: str, snap: dict | None = None):
        """Load the snapshot in Node and return view(roots, modules) as a JSON value."""
        snap = self.snapshot() if snap is None else snap
        out = self.folder / "js"
        out.mkdir(exist_ok=True)
        live = LiveTypes(set(self.sources))
        for name, mod in transpile_app(self.sources, roots=self.roots, live=live).items():
            (out / f"{name}.js").write_text(mod.code)
        for f in ("pyrt.js", "manifoldx.js"):
            shutil.copy(JS / f, out / f)
        (out / "state.json").write_text(json.dumps(snap))
        imports = "\n".join(
            f'import * as m{i} from "./{name}.js";' for i, name in enumerate(self.sources)
        )
        modules = ", ".join(f'"{name}": m{i}' for i, name in enumerate(self.sources))
        script = f"""{imports}
import {{ loadState }} from "./manifoldx.js";
import {{ readFileSync }} from "node:fs";
const modules = {{ {modules} }};
const resolve = {RESOLVE};
const roots = loadState(JSON.parse(readFileSync(new URL("./state.json", import.meta.url))), modules, resolve);
const view = {view};
console.log(JSON.stringify(view(roots, modules, resolve), {REPLACER}));
"""
        (out / "main.mjs").write_text(script)
        done = subprocess.run(
            ["node", str(out / "main.mjs")], capture_output=True, text=True, timeout=60
        )
        if done.returncode != 0:
            raise RuntimeError(f"node failed:\n{done.stderr}")
        return json.loads(done.stdout)


def _entries(snap, type_name):
    return [e for e in snap["table"] if isinstance(e, dict) and e.get("__type") == type_name]


SHARED = """
import numpy as np

class Ground:
    def __init__(self):
        self.h = 1.5
        self.secret = "setup only"

class Rig:
    def __init__(self, ground):
        self.ground = ground
        self.pos = np.zeros(3, dtype=np.float32)
        self.vel = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.tilt = np.array([1.0, 2.0, 3.0])
        self.notes = "setup only"

    def update(self, dt):
        self.ground.h = self.ground.h + dt
        self.pos += self.vel * dt
        self.tilt += dt

class Scene:
    def __init__(self, ground):
        self.terrain = ground

    def tick(self, dt):
        self.terrain.h = self.terrain.h * 2

rig = None
scene = None

def step(query, dt):
    rig.update(dt)
    scene.tick(dt)
    return [rig.ground.h, scene.terrain.h, rig.tilt[1]]

def setup(engine):
    global rig, scene
    g = Ground()
    rig = Rig(g)
    scene = Scene(g)
    engine.system(step)
"""

SHARED_VIEW = """(roots, m) => {
  const before = m.app.rig.ground === m.app.scene.terrain;
  const out = roots[0].fn(null, 0.5);
  return { same: before, out, pos: m.app.rig.pos,
           f32: m.app.rig.pos instanceof Float32Array,
           f64: m.app.rig.tilt instanceof Float64Array };
}"""


def test_shared_objects_stay_one_object(app):
    a = app({"app": SHARED}, lambda mods, e: mods["app"].setup(e))
    js = a.node(SHARED_VIEW)
    mod = a.mods["app"]
    assert js["same"] is True
    assert js["out"] == mod.step(None, 0.5) == [4.0, 4.0, 2.5]
    assert js["pos"] == mod.rig.pos.tolist()


def test_only_reached_fields_are_written(app):
    a = app({"app": SHARED}, lambda mods, e: mods["app"].setup(e))
    snap = a.snapshot()
    assert [set(e["fields"]) for e in _entries(snap, "app.Ground")] == [{"h"}]
    assert [set(e["fields"]) for e in _entries(snap, "app.Rig")] == [
        {"ground", "pos", "vel", "tilt"}
    ]
    assert "setup only" not in json.dumps(snap)


def test_float32_arrays_load_as_float32array(app):
    a = app({"app": SHARED}, lambda mods, e: mods["app"].setup(e))
    snap = a.snapshot()
    assert {"__f32": [0.0, 0.0, 0.0]} in snap["table"]
    assert {"__f32": [1.0, 2.0, 3.0]} in snap["table"]
    assert {"__f64": [1.0, 2.0, 3.0]} in snap["table"]
    js = a.node(SHARED_VIEW)
    assert js["f32"] is True and js["f64"] is True


MATERIAL = """
class Picker:
    def __init__(self, engine, box, hi):
        self.engine = engine
        self.box = box
        self.hi = hi

    def pick(self, query, dt):
        self.engine.set_material(self.box, self.hi)
        return 1
"""


def _picker(mods, engine):
    box = engine.spawn(
        mx.components.Mesh(mx.resources.cube(1, 1, 1)),
        mx.components.Material(mx.resources.BasicMaterial("#ff0000")),
        mx.components.Transform(pos=(0, 0, 0)),
    )
    hi = mx.resources.BasicMaterial("#00ff00")
    p = mods["app"].Picker(engine, box, hi)
    engine.system(p.pick)
    mods["app"].picker = p


def _glb_json(path: Path) -> dict:
    data = path.read_bytes()
    (n,) = struct.unpack_from("<I", data, 12)
    return json.loads(data[20 : 20 + n])


def test_a_material_reached_but_never_spawned_is_registered_and_exported(app, tmp_path):
    a = app({"app": MATERIAL}, _picker)
    hi = a.mods["app"].picker.hi
    registry = a.engine._material_registry
    assert id(hi) not in registry._object_to_id
    snap = a.snapshot()
    mid = registry._object_to_id[id(hi)]
    assert registry.get(mid) is hi
    assert {"__material": mid} in _entries(snap, "app.Picker")[0]["fields"].values()
    a.engine.export_gltf(tmp_path / "s.glb")
    ids = _glb_json(tmp_path / "s.glb")["scenes"][0]["extras"]["manifoldx"]["mx_materials"]
    assert str(mid) in ids
    box = a.mods["app"].picker.box.index
    js = a.node("(roots, m, resolve) => { roots[0].fn(null, 0); return resolve('engine').log; }")
    assert js == [[box, mid]]


DICTS = """
class Viewer:
    def __init__(self):
        self.d = {"constructor": 1, "__proto__": 2, "toString": 3}
        self.o = {"a": 1, "10": 2, "2": 3}

    def check(self, query, dt):
        d = self.d
        return [
            "constructor" in d, "__proto__" in d, "toString" in d, "valueOf" in d,
            d.get("constructor"), d.get("__proto__"), d.get("toString"), d.get("valueOf", -1),
            d["constructor"], d["__proto__"], d["toString"], list(d), len(d),
        ]

    def order(self, query, dt):
        o = self.o
        o["b"] = 4
        o["1"] = 5
        return [list(o), [k for k in o], list(o.values()), list(o.items())]

    def missing(self, query, dt):
        return self.d["valueOf"]

def setup(engine):
    v = Viewer()
    engine.system(v.check)
    engine.system(v.order)
    engine.system(v.missing)
    return v
"""


def test_dict_keys_named_like_js_prototype_members_round_trip(app):
    a = app({"app": DICTS}, lambda mods, e: setattr(mods["app"], "v", mods["app"].setup(e)))
    py = a.mods["app"].v.check(None, 0)
    js = a.node("(roots) => roots[0].fn(null, 0)")
    assert js == py
    assert py[:4] == [True, True, True, False]


def test_dict_iteration_follows_python_insertion_order(app):
    a = app({"app": DICTS}, lambda mods, e: setattr(mods["app"], "v", mods["app"].setup(e)))
    js = a.node("(roots) => roots[1].fn(null, 0)")
    py = json.loads(json.dumps(a.mods["app"].v.order(None, 0)))
    assert py[0] == ["a", "10", "2", "b", "1"]
    assert js == py


def test_a_missing_key_raises_key_error_in_reached_code(app):
    a = app({"app": DICTS}, lambda mods, e: setattr(mods["app"], "v", mods["app"].setup(e)))
    with pytest.raises(KeyError):
        a.mods["app"].v.missing(None, 0)
    js = a.node(
        "(roots) => { try { return roots[2].fn(null, 0); } catch (e) { return e.message; } }"
    )
    assert isinstance(js, str) and js.startswith("KeyError")


INT_KEYS = """
class Viewer:
    def __init__(self):
        self.names = {3: "x"}

    def show(self, query, dt):
        return len(self.names)
"""


def test_a_dict_with_an_int_key_fails_with_its_path(app):
    a = app({"app": INT_KEYS}, lambda mods, e: e.system(mods["app"].Viewer().show))
    with pytest.raises(SnapshotError) as err:
        a.snapshot()
    assert err.value.path_from_root == "Viewer.names"
    assert "Viewer.names" in str(err.value) and "int" in str(err.value)


GETTERS = """
from manifoldx.gui import Panel, ValueDisplay

class Viewer:
    def __init__(self):
        self.names = ["Matemática", "Física"]
        self.t = 0.25

class Scale:
    def __init__(self, k):
        self.k = k

    def __call__(self, x):
        return x * self.k

def setup(engine):
    v = Viewer()
    scale = Scale(4.0)
    rows = [ValueDisplay(lambda i=i: f"{i}: {v.names[i]} {v.t:.2f}") for i in range(2)]
    rows.append(ValueDisplay(lambda: f"{scale(v.t):.1f}"))

    def tick(query, dt):
        v.t = v.t + dt
        return v.t

    engine.system(tick)
    engine.gui.append(Panel(children=rows))
"""


def test_a_getter_lambda_with_a_default_returns_what_python_returns(app):
    a = app({"app": GETTERS}, lambda mods, e: mods["app"].setup(e))
    getters = [r for r in a.roots if r.kind == "getter"]
    js = a.node("(roots) => roots.filter((r) => r.kind === 'getter').map((r) => r.fn())")
    assert js == [r.func() for r in getters] == ["0: Matemática 0.25", "1: Física 0.25", "1.0"]
    snap = a.snapshot()
    closures = [r["fn"] for r in snap["roots"] if r["kind"] == "getter"]
    assert closures[0]["__closure"].startswith("app:")
    assert closures[0]["defaults"] == [0] and closures[1]["defaults"] == [1]


def test_a_nested_def_root_closes_over_the_same_objects_as_the_getters(app):
    a = app({"app": GETTERS}, lambda mods, e: mods["app"].setup(e))
    js = a.node(
        """(roots) => {
          const tick = roots.find((r) => r.kind === "system").fn;
          tick(null, 0.5);
          return roots.filter((r) => r.kind === "getter").map((r) => r.fn());
        }"""
    )
    a.roots[0].func(None, 0.5)
    assert js == [r.func() for r in a.roots if r.kind == "getter"]
    assert js[2] == "3.0"


def test_roots_keep_their_kind_label_and_event(app):
    a = app({"app": DICTS}, lambda mods, e: setattr(mods["app"], "v", mods["app"].setup(e)))
    snap = a.snapshot()
    assert [r["kind"] for r in snap["roots"]] == ["system"] * 3
    assert [r["label"] for r in snap["roots"]] == [r.label for r in a.roots]
    assert snap["roots"][0]["fn"]["name"] == "check"
