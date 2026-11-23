"""Optional native steps for the simulation examples.

The extension is built from ``crates/spike-nbody-boids`` on this branch. When it
is not built, or when ``MANIFOLDX_NUMPY=1`` is set, ``load()`` returns ``None``
and the examples use their numpy path. A packaged build would expose the same
module as ``manifoldx._native``; the local build is a development hook.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

_MODULE: ModuleType | None = None
_LOADED = False


def _candidates():
    configured = os.environ.get("MANIFOLDX_NATIVE_SO")
    if configured:
        yield Path(configured)
    root = Path(__file__).resolve().parents[2]
    yield root / "crates" / "spike-nbody-boids" / "target" / "release" / "lib_spike_nb.so"
    yield root / "target" / "release" / "lib_spike_nb.so"


def load() -> ModuleType | None:
    """The native module, or None when it is not available or numpy is forced."""
    global _MODULE, _LOADED
    if _LOADED:
        return _MODULE
    _LOADED = True
    if os.environ.get("MANIFOLDX_NUMPY"):
        return None
    try:
        from manifoldx import _native  # a packaged build

        _MODULE = _native
        return _MODULE
    except ImportError:
        pass
    for path in _candidates():
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("_spike_nb", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE = module
        return _MODULE
    return None


def available() -> bool:
    return load() is not None
