"""Set methods (gia-uh/manifoldx#149): add, discard, remove, clear and update give the same
values in Python and in Node; remove of a missing item throws KeyError as Python raises it."""

import tempfile
from pathlib import Path

import pytest

from manifoldx.web import TranspileError, transpile_app

from .parity import run_both, run_node, run_python

CASES = {
    "add": "def main():\n    s = {'a'}\n    s.add('b')\n    s.add('a')\n    return [s, len(s)]",
    "add_to_empty": "def main():\n    s = set()\n    s.add(3)\n    s.add(1)\n    return [s, 3 in s, 2 in s]",
    "discard": "def main():\n    s = {1, 2, 3}\n    s.discard(2)\n    s.discard(9)\n    return s",
    "remove": "def main():\n    s = {'x', 'y'}\n    s.remove('x')\n    return [s, 'x' in s]",
    "clear": "def main():\n    s = {1, 2}\n    s.clear()\n    return [s, len(s), bool(s)]",
    "update": "def main():\n    s = {1}\n    s.update([2, 3], {3, 4}, range(5, 7))\n    return s",
    "update_str": "def main():\n    s: set[str] = set()\n    s.update('abca')\n    return s",
    "returns_none": "def main():\n    s = {1}\n    return [s.add(2), s.discard(1), s.remove(2), s.update([5]), s.clear()]",
    "field": (
        "class Seen:\n"
        "    def __init__(self):\n"
        "        self.names: set[str] = set()\n"
        "def main():\n"
        "    seen = Seen()\n"
        "    for n in ['b', 'a', 'b']:\n"
        "        seen.names.add(n)\n"
        "    seen.names.discard('a')\n"
        "    return seen.names"
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_set_method(name):
    py, js = run_both(CASES[name])
    assert js == py


REMOVE_MISSING = "def main():\n    s = {'a'}\n    s.remove('b')\n    return s"


def test_remove_of_a_missing_item_throws_key_error_as_python_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(KeyError):
            run_python({"mod": REMOVE_MISSING}, Path(tmp), "mod")
        with pytest.raises(RuntimeError, match="KeyError: 'b'"):
            run_node({"mod": REMOVE_MISSING}, Path(tmp), "mod")


def test_an_item_js_compares_by_identity_is_outside_the_subset():
    """A JS Set holds two equal tuples as two items, so adding one is refused."""
    src = "def main():\n    s = {(1, 2)}\n    s.add((1, 2))\n    return len(s)"
    with pytest.raises(TranspileError, match=r"mod.py:3: .*\.add\(\) of a tuple"):
        transpile_app({"mod": src})
