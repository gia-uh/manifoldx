"""Roots from an engine, and reach through types: what the browser runs, and which fields of
which classes it reads."""

import ast
import importlib
import sys

import pytest

import manifoldx as mx
from manifoldx.web import TranspileError, transpile_app
from manifoldx.web.reach import Reach, Root, find_roots
from manifoldx.web.types import LiveTypes, TypeEnv


@pytest.fixture
def load(tmp_path):
    """Import sources ({module: source}) as real modules, and a fresh TypeEnv over them."""
    names = []

    def _load(sources):
        for name, src in sources.items():
            (tmp_path / f"{name}.py").write_text(src)
            sys.modules.pop(name, None)
            names.append(name)
        importlib.invalidate_caches()
        mods = {name: importlib.import_module(name) for name in sources}
        trees = {name: ast.parse(src) for name, src in sources.items()}
        return mods, (lambda: TypeEnv(trees, LiveTypes(set(sources))))

    sys.path.insert(0, str(tmp_path))
    yield _load
    sys.path.remove(str(tmp_path))
    for name in names:
        sys.modules.pop(name, None)


APP = """
from manifoldx.gui import Panel, Text, ValueDisplay

class Building:
    def __init__(self):
        self.parts = [1, 2]
        self.name = "b"

    def count(self):
        return len(self.parts)

class Rig:
    def __init__(self):
        self.parts = ["x"]

    def count(self):
        return len(self.parts) * 10

class Viewer:
    def __init__(self, selected):
        self.selected = selected
        self.rig = Rig()
        self.t = 0.0
        self.n = 0

    def update(self, query, dt):
        self.t += dt
        self.n = self.selected.count()

    def on_key(self, payload):
        self.n = 0

def spin(query, dt):
    pass

def startup(payload):
    pass

def setup(engine):
    v = Viewer(Building())
    engine.system(v.update)
    engine.system(spin)
    engine.on("key_down")(v.on_key)
    engine.on("startup")(startup)
    engine.gui.append(Panel(children=[Text("hi"), Panel(children=[
        ValueDisplay(lambda: f"{v.t:.1f}"),
    ])]))
    return v
"""


def _engine():
    return mx.Engine("t", width=64, height=64)


def test_find_roots_returns_systems_handlers_and_getters(load):
    mods, _ = load({"app": APP})
    app = mods["app"]
    e = _engine()
    v = app.setup(e)
    roots = find_roots(e)
    by_kind = {}
    for r in roots:
        by_kind.setdefault(r.kind, []).append(r)
    assert [r.func for r in by_kind["system"]] == [v.update, app.spin]
    # startup and the GUI bridge's own handlers are not roots.
    assert [r.func for r in by_kind["handler"]] == [v.on_key]
    assert by_kind["handler"][0].event == "key_down"
    getter = e.gui._panels[0].children[1].children[0]._getter
    assert [r.func for r in by_kind["getter"]] == [getter]
    assert all(isinstance(r, Root) and r.label for r in roots)
    assert by_kind["system"][0].owner is v
    assert by_kind["getter"][0].cells == {"v": v}


def test_reach_follows_a_method_only_on_the_receivers_classes(load):
    mods, env_of = load({"app": APP})
    app = mods["app"]
    v = app.Viewer(app.Building())
    reach = Reach([Root("system", "system Viewer.update", v.update)], env_of())
    assert ("app", "Building.count") in reach.functions
    assert ("app", "Rig.count") not in reach.functions
    assert reach.fields["app.Viewer"] == {"t", "selected"}
    assert reach.fields["app.Building"] == {"parts"}
    assert "app.Rig" not in reach.fields
    assert reach.chain(("app", "Building.count")) == [
        "system Viewer.update",
        "app.Building.count",
    ]


def test_a_getter_lambda_reads_through_its_closure_cells(load):
    mods, env_of = load({"app": APP})
    app = mods["app"]
    e = _engine()
    app.setup(e)
    getters = [r for r in find_roots(e) if r.kind == "getter"]
    reach = Reach(getters, env_of())
    assert reach.fields["app.Viewer"] == {"t"}


PATH = """from pathlib import Path

def tick(query, dt):
    return Path("x").exists()
"""


def test_a_method_of_a_library_object_fails_naming_the_class_line_and_root(load):
    mods, env_of = load({"app": PATH})
    with pytest.raises(TranspileError) as err:
        Reach([Root("system", "system tick", mods["app"].tick)], env_of())
    msg = str(err.value)
    assert "pathlib.Path" in msg
    assert "app.py:4:" in msg
    assert "system tick" in msg


LIVE_PATH = """
class Doc:
    def __init__(self, path):
        self.path = path

    def show(self, payload):
        return self.path.exists()
"""


def test_a_live_library_object_fails_too(load, tmp_path):
    mods, env_of = load({"app": LIVE_PATH})
    d = mods["app"].Doc(tmp_path)
    with pytest.raises(TranspileError, match=r"pathlib\.\w*Path.*on_show"):
        Reach([Root("handler", "on_show", d.show)], env_of())


QUERY = """
def move(query, dt):
    for x in query:
        pass

def tick(query, dt):
    return dt * 2
"""


def test_a_systems_query_types_from_its_registration(load):
    from manifoldx.web.types import INT, query

    mods, env_of = load({"app": QUERY})
    root = Root("system", "system move", mods["app"].move, components=("Transform",))
    reach = Reach([root], env_of())
    assert reach.env.local(("app", "move"), "query") == query(("Transform",))
    assert reach.env.local(("app", "move"), "x") == INT  # iterating it gives entity indices
    # An unused query is fine, and dt is a number.
    reach = Reach([Root("system", "system tick", mods["app"].tick)], env_of())
    assert reach.functions == {("app", "tick")}


def test_transpile_app_takes_roots_and_emits_only_what_they_reach(load):
    mods, _ = load({"app": APP})
    app = mods["app"]
    v = app.Viewer(app.Building())
    live = LiveTypes({"app"})
    out = transpile_app(
        {"app": APP}, roots=[Root("system", "system Viewer.update", v.update)], live=live
    )
    code = out["app"].code
    assert "class Building" in code and "count()" in code
    assert "class Rig" not in code
    assert "function spin" not in code and "function setup" not in code
