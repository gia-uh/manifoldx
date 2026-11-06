"""Types from live objects and from call sites: what the transpiler knows about a value when
the app annotates nothing."""

import ast
import importlib
import sys

import numpy as np
import pytest

from manifoldx.web import TranspileError, transpile_app
from manifoldx.web.types import (
    INT,
    NONE,
    NUM,
    NUM32,
    NUM64,
    STR,
    LiveTypes,
    TypeEnv,
    list_of,
    obj,
    union,
    vec,
)

from .parity import run_both


def same(a, b):
    """a and b are the same type, a union's members in any order."""
    return set(a.members) == set(b.members) and len(a.members) == len(b.members)


@pytest.fixture
def load(tmp_path):
    """Import sources ({module: source}) as real modules, so live objects carry the names the
    transpiler gives their classes; and a TypeEnv over the same sources."""
    names = []

    def _load(sources, live=None):
        for name, src in sources.items():
            (tmp_path / f"{name}.py").write_text(src)
            sys.modules.pop(name, None)
            names.append(name)
        importlib.invalidate_caches()
        mods = {name: importlib.import_module(name) for name in sources}
        trees = {name: ast.parse(src) for name, src in sources.items()}
        return mods, (lambda live: TypeEnv(trees, live))

    sys.path.insert(0, str(tmp_path))
    yield _load
    sys.path.remove(str(tmp_path))
    for name in names:
        sys.modules.pop(name, None)


VIEWER = """
class Scene:
    def __init__(self):
        self.names = {"a": "Alpha"}

    def by_id(self, pid):
        if pid is None:
            return None
        return self.names.get(pid)

class Viewer:
    def __init__(self, scene, selected):
        self.scene = scene
        self.selected = selected

    def label(self):
        return f(self.scene, self.selected)

def f(scene, pid):
    return scene.by_id(pid)
"""


def test_call_site_arguments_type_parameters_from_live_fields(load):
    mods, env_of = load({"app": VIEWER})
    app = mods["app"]
    live = LiveTypes()
    live.observe(app.Viewer(app.Scene(), "a"))
    live.observe(app.Viewer(app.Scene(), None))
    env = env_of(live)
    env.solve([("app", "Viewer.label"), ("app", "f"), ("app", "Scene.by_id")])
    f = env.function(("app", "f"))
    assert f.param_types["scene"] == obj("app.Scene")
    assert same(f.param_types["pid"], union(STR, NONE))
    assert same(env.function(("app", "Scene.by_id")).param_types["pid"], union(STR, NONE))
    # Returns join back into callers: by_id gives str | None, and so does label.
    assert same(env.function(("app", "Viewer.label")).returns, union(NONE, STR))


HOLDER = """
class Building:
    def __init__(self, name):
        self.name = name

class Landmark:
    def __init__(self, name):
        self.name = name

class Holder:
    def __init__(self, item):
        self.item = item
"""


def test_a_field_holding_two_classes_types_as_their_union(load):
    mods, env_of = load({"app": HOLDER})
    app = mods["app"]
    live = LiveTypes()
    live.observe(app.Holder(app.Building("b")))
    live.observe(app.Holder(app.Landmark("l")))
    assert same(live.fields["app.Holder"]["item"], union(obj("app.Building"), obj("app.Landmark")))
    # observe walks into the objects a field holds.
    assert live.fields["app.Building"]["name"] == STR
    env = env_of(live)
    env.solve([])
    assert same(
        env.classes["app.Holder"].fields["item"], union(obj("app.Building"), obj("app.Landmark"))
    )


NUMPY_FIELDS = """
class Body:
    def __init__(self):
        self.pos = None
        self.mass = None
        self.weight = None
        self.tags = []
        self.n = 0
"""


def test_live_numpy_values_keep_their_precision(load):
    mods, _ = load({"app": NUMPY_FIELDS})
    b = mods["app"].Body()
    b.pos = np.zeros(3, np.float32)
    b.mass = np.float32(2.0)
    b.weight = np.float64(2.0)
    b.tags = ["x"]
    b.n = 3
    other = mods["app"].Body()
    other.pos = np.zeros(3)
    other.mass = np.float32(1.0)
    other.weight = 1.5
    other.tags = []
    live = LiveTypes()
    live.observe(b)
    live.observe(other)
    f = live.fields["app.Body"]
    assert same(f["pos"], union(vec("f32"), vec("f64")))
    assert f["mass"] == NUM32
    assert same(f["weight"], union(NUM64, NUM))
    assert f["tags"] == list_of(STR)
    assert f["n"] == INT


ANNOTATED = """
class Counter:
    count: int

    def __init__(self):
        self.count = 0

def k(x: float):
    return x
"""


def test_an_annotation_wins_over_a_live_type_and_a_call_site(load):
    mods, env_of = load({"app": ANNOTATED + "\ndef main():\n    return k(1)\n"})
    c = mods["app"].Counter()
    c.count = 2.5  # the live value disagrees with the annotation
    live = LiveTypes()
    live.observe(c)
    env = env_of(live)
    env.solve()
    assert env.classes["app.Counter"].fields["count"] == INT
    assert env.function(("app", "k")).param_types["x"] == NUM


def test_a_parameter_takes_the_types_its_callers_pass():
    src = (
        "def twice(x, k=2):\n    return x * k\n\n"
        "def chain(y):\n    return twice(y) + 1\n\n"
        "def main():\n    return [chain(3), twice(1.5, k=3)]\n"
    )
    py, js = run_both(src)
    assert js == py


def test_recursion_types_through_its_own_call_site():
    src = (
        "def count(n, acc):\n    if n == 0:\n        return acc\n    return count(n - 1, acc + n)\n\n"
        "def main():\n    return count(4, 0)\n"
    )
    py, js = run_both(src)
    assert js == py


def test_a_plus_whose_operand_stays_unknown_asks_for_an_annotation():
    src = "def g(a, b):\n    return a + b\n\ndef main(q):\n    return g(q, 1)\n"
    with pytest.raises(TranspileError, match=r"annotate `a`") as err:
        transpile_app({"mod": src})
    assert "a + b" in str(err.value)
    assert ":2:" in str(err.value)


def test_solve_reports_the_first_unknown_operator():
    tree = ast.parse("def h(v):\n    s = 'x'\n    return f'{v}' + s\n")
    env = TypeEnv({"mod": tree})
    with pytest.raises(TranspileError, match=r"annotate `v`"):
        env.solve([("mod", "h")])


def test_equality_of_two_unknowns_asks_for_an_annotation():
    with pytest.raises(TranspileError, match=r"annotate"):
        transpile_app({"mod": "def same(a, b):\n    return a == b\n"})


def test_equality_with_a_scalar_needs_no_type():
    out = transpile_app({"mod": "def is_a(a):\n    return a == 'a'\n"})
    assert "is_a" in out["mod"].code


CYCLE = """
class Doc:
    def __init__(self):
        self.n = 1

    def bump(self):
        self.n += helper(self.n)

def helper(k):
    return k * 2
"""


def test_a_field_fed_by_a_call_that_reads_it_keeps_its_live_type(load):
    mods, env_of = load({"app": CYCLE})
    live = LiveTypes()
    live.observe(mods["app"].Doc())
    env = env_of(live)
    env.solve([("app", "Doc.bump"), ("app", "helper")])
    assert env.classes["app.Doc"].fields["n"] == INT
    assert env.function(("app", "helper")).param_types["k"] == INT
