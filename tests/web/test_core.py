"""The core subset: each snippet's main() gives the same JSON value in Python and in Node."""

import re

import pytest

from manifoldx.web import transpile_app

from .parity import run_both

CASES = {
    "mod_sign": "def main(): return [-7 % 3, 7 % -3, -7.5 % 2, -7 // 2]",
    "chained": "def main(): x = 2; return [0 < x < 3, 0 < x > 3]",
    "truthy": "def main(): return [bool([]), not {}, bool(''), 1 if [0] else 2]",
    "and_or_values": "def main(): return [[] or 'd', 'a' and 'b', 0 or None]",
    "kwargs": "def f(a, b=2, c=3): return [a, b, c]\ndef main(): return [f(1, c=9), f(1)]",
    "loops": "def main():\n    out = []\n    for i in range(3):\n        if i == 1: continue\n        out.append(i)\n    while out: out.pop()\n    return out",
    "comprehensions": "def main(): return [[x * x for x in range(4) if x % 2], {k for k in 'aba'} == {'a', 'b'}]",
    "unpacking": "def main(): (a, b), c = (1, 2), 3; return [a, b, c]",
    "minmax_key": "def main(): return [min([3, 1, 2]), max(['bb', 'a'], key=len), min(2, 1.5)]",
}


@pytest.mark.parametrize("name", CASES)
def test_core(name):
    src = CASES[name]
    py, js = run_both(src)
    assert js == py
    code = transpile_app({"mod": src})["mod"].code
    assert not re.search(r"\beval\b", code)
    assert not re.search(r"\bwith\b", code)
