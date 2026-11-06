"""Classes, closures and collections: each snippet's main() gives the same JSON value in
Python and in Node.

Parameters take their types from the arguments their call sites pass."""

import re

import pytest

from manifoldx.web import TranspileError, transpile_app

from .parity import run_both, run_modules

DATACLASS = """
from dataclasses import dataclass, field

@dataclass
class Bag:
    name: str
    items: list[int] = field(default_factory=list)
    scale: float = 1.0

    def total(self) -> float:
        return sum_(self.items) * self.scale

def sum_(xs):
    t = 0
    for x in xs:
        t += x
    return t

def main():
    a = Bag("a")
    b = Bag("b", scale=2.0)
    a.items.append(1)
    a.items.append(2)
    c = Bag(scale=0.5, name="c", items=[4])
    return [a.name, a.items, b.items, b.scale, a.total(), c.total(), c.name]
"""

PROPERTY = """
class Rect:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    @property
    def area(self) -> float:
        return self.w * self.h

    def grow(self, by):
        self.w = self.w + by

def main():
    r = Rect(2.0, 3.5)
    a = r.area
    r.grow(1.0)
    return [a, r.area, r.w, hasattr(r, "area"), hasattr(r, "depth")]
"""

CALLABLE = """
class Scale:
    def __init__(self, k):
        self.k = k
        self.name = "scale"

    def __call__(self, x):
        return x * self.k

def apply(f, x):
    return f(x)

def main():
    s = Scale(3.0)
    return [s(2.0), apply(s, 1.5), s.name, isinstance(s, Scale)]
"""

CLOSURE_DEFAULTS = """
def main():
    fs = [lambda i=i: i * 10 for i in range(3)]
    gs = []
    for j in range(3):
        gs.append(lambda j=j: j + 1)
    late = []
    for k in range(3):
        late.append(lambda: k)
    return [f() for f in fs] + [g() for g in gs] + [h() for h in late]
"""

NESTED_DEF = """
def make_counter(start):
    count = start

    def step(by=1):
        return count + by

    return step

def main():
    f = make_counter(10)
    adders = []
    for i in range(3):
        def add(x, i=i):
            return x + i
        adders.append(add)
    return [f(), f(5), [a(100) for a in adders]]
"""

TWO_CLASSES = """
class Building:
    def __init__(self, name, floors):
        self.name = name
        self.floors = floors

class Landmark:
    def __init__(self, name):
        self.name = name

def describe(x):
    if isinstance(x, Building):
        return x.name + " " + str(x.floors)
    if hasattr(x, "name"):
        return x.name
    return "?"

def main():
    xs = [Building("A", 3), Landmark("B")]
    return [describe(x) for x in xs] + [
        hasattr(xs[0], "floors"),
        hasattr(xs[1], "floors"),
        hasattr(xs[1], "constructor"),
        isinstance(xs[1], Building),
        isinstance(xs[1], (Building, Landmark)),
    ]
"""

DICTS = """
def main():
    d = {"a": 1, "b": 2}
    d["c"] = 3
    total = 0
    for k in d:
        total += d[k]
    pairs = [k + "=" + str(v) for k, v in d.items()]
    return [d.get("a"), d.get("z"), d.get("z", 9), "b" in d, "z" in d, total, len(d),
            pairs, list(d.keys()), list(d.values()), d]
"""

LISTS = """
def main():
    a = [1, 2]
    b = a + [3]
    c = [0] * 3
    d = 2 * ["x"]
    e = [[1]] * 2
    b.append(4)
    return [a, b, c, d, e, [] * 4, a * 0]
"""

NEXT_GENEXP = """
def main():
    xs = [3, 8, 1, 9]
    first = next(x for x in xs if x > 5)
    missing = next((x for x in xs if x > 100), None)
    # Lazy as Python: the item after the first is never computed (10 // 0 would raise).
    lazy = next(10 // x for x in [5, 0])
    return [first, missing, lazy]
"""

CASES = {
    "dataclass_factory": DATACLASS,
    "property": PROPERTY,
    "callable_instance": CALLABLE,
    "closure_defaults": CLOSURE_DEFAULTS,
    "nested_def": NESTED_DEF,
    "two_classes": TWO_CLASSES,
    "dicts": DICTS,
    "lists": LISTS,
    "next_genexp": NEXT_GENEXP,
}


@pytest.mark.parametrize("name", CASES)
def test_classes(name):
    py, js = run_both(CASES[name])
    assert js == py
    code = transpile_app({"mod": CASES[name]})["mod"].code
    assert not re.search(r"\beval\b", code)


def test_method_self_is_this():
    code = transpile_app({"mod": PROPERTY})["mod"].code
    assert "this.w = (this.w + by)" in code
    assert not re.search(r"\bself\b", code)


GEOM = """
def scale(v, k=1.0, offset=0.0):
    return [x * k + offset for x in v]

class Box:
    def __init__(self, w, h=1.0):
        self.w = w
        self.h = h
"""

USES_GEOM = """
import geom
from geom import scale, Box

def main():
    b = geom.Box(h=4.0, w=2.0)
    c = Box(3.0)
    return [geom.scale([1.0, 2.0], offset=0.5), scale([1.0], k=3.0), [b.w, b.h, c.w, c.h]]
"""


def test_keyword_arguments_across_modules():
    py, js = run_modules({"geom": GEOM, "mod": USES_GEOM})
    assert js == py


def test_equality_of_dataclasses_is_outside_the_subset():
    src = DATACLASS + "\ndef same():\n    return Bag('a') == Bag('a')\n"
    with pytest.raises(TranspileError, match="dataclass"):
        transpile_app({"mod": src})
