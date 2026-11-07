"""The tab's engine.state back to Python, for manifoldx run --web.

The tab encodes its state with the snapshot's value forms (manifoldx.js's encodeState):
{"__dict": [[k, v], ...]}, {"__set": [...]}, {"__f32": [...]}, {"__f64": [...]},
{"__float": "nan" | "inf" | "-inf"}, arrays as JSON arrays, and {"__skip": what} for what is
not plain data (an object of a transpiled class, a handle, a function, a cycle). The tab has
one kind of array, so decode reads each value against what Python had at the same path.
"""

from __future__ import annotations

import numpy as np

FLOATS = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}


def decode(value, like=None):
    """A value the tab encoded as Python's: an array is a tuple where like is a tuple, each
    item decoded against like's item when the lengths match; a dict's keys against like's; a
    whole number is a float where like is a float (JSON from the tab writes 6.0 as 6)."""
    if isinstance(like, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, list):
        same = isinstance(like, (list, tuple)) and len(like) == len(value)
        items = [decode(x, like[i] if same else None) for i, x in enumerate(value)]
        return tuple(items) if isinstance(like, tuple) else items
    if not isinstance(value, dict):
        return value
    if "__dict" in value:
        old = like if isinstance(like, dict) else {}
        return {k: decode(x, old.get(k)) for k, x in value["__dict"]}
    if "__set" in value:
        return {decode(x) for x in value["__set"]}
    if "__f32" in value:
        return np.array([decode(x) for x in value["__f32"]], dtype=np.float32)
    if "__f64" in value:
        return np.array([decode(x) for x in value["__f64"]], dtype=np.float64)
    if "__float" in value:
        return FLOATS[value["__float"]]
    raise ValueError(f"a state value of unknown form: {str(value)[:80]}")


def _skipped(value) -> str | None:
    """The what of the first {"__skip": what} inside value, or None."""
    if isinstance(value, list):
        return next((w for w in map(_skipped, value) if w is not None), None)
    if not isinstance(value, dict):
        return None
    if "__skip" in value:
        return value["__skip"]
    if "__dict" in value:
        return _skipped([x for _, x in value["__dict"]])
    return _skipped(value.get("__set"))


def merge_tab_state(state, tab: dict, crossed: frozenset[str]) -> list[str]:
    """Merge the tab's encoded engine.state into state (hot.State) in place: a crossed key the
    tab no longer has is deleted, one holding a __skip anywhere keeps Python's value, any
    other is replaced; a key that did not cross keeps Python's value; a key new in the tab is
    added. Returns the keys kept from Python, each with why."""
    kept = []
    pairs = dict(tab["__dict"])
    for k in [k for k in state if k not in pairs]:
        if k in crossed:
            del state[k]
        else:
            kept.append(f"{k}: left out of the snapshot")
    for k, v in pairs.items():
        what = _skipped(v)
        if what is not None:
            kept.append(f"{k}: {what} stays as Python had it")
        elif k in state and k not in crossed:
            kept.append(f"{k}: left out of the snapshot")
        else:
            state[k] = decode(v, state.get(k))
    return kept
