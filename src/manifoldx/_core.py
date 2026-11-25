"""Private startup selection. A native bootstrap is not an Engine runtime."""

import importlib
import os


def _selection():
    """Return the selected core and its bootstrap; unset/empty means Python."""
    name = os.environ.get("MANIFOLDX_CORE") or "python"
    if name not in ("python", "rust"):
        raise ValueError(
            f"Invalid MANIFOLDX_CORE={name!r}; expected 'python' or 'rust' (empty means 'python')"
        )
    if name == "python":
        return name, None
    try:
        native = importlib.import_module("manifoldx._core_native")
    except ModuleNotFoundError as exc:
        if exc.name != "manifoldx._core_native":
            raise ImportError(
                "MANIFOLDX_CORE=rust: manifoldx._core_native could not load; reinstall or rebuild the native wheel"
            ) from exc
        raise ImportError(
            "MANIFOLDX_CORE=rust requires manifoldx._core_native; install a native manifold-gfx wheel or build with make native-build"
        ) from exc
    except (ImportError, OSError) as exc:
        raise ImportError(
            "MANIFOLDX_CORE=rust: manifoldx._core_native could not load; reinstall or rebuild the native wheel"
        ) from exc
    return name, dict(native.bootstrap())


def _validate_engine_core():
    name, bootstrap = _selection()
    if name == "rust":
        raise NotImplementedError(
            f"MANIFOLDX_CORE=rust: native Engine runtime is not implemented "
            f"(bootstrap runtime={bootstrap.get('runtime')!r}); use MANIFOLDX_CORE=python"
        )


def _make_native_engine(*args, **kwargs):
    _, bootstrap = _selection()
    native = importlib.import_module("manifoldx._core_native")
    if not hasattr(native, "_NativeGraphics"):
        raise NotImplementedError(
            f"MANIFOLDX_CORE=rust: native Engine runtime is not implemented "
            f"(bootstrap runtime={bootstrap.get('runtime')!r}); use MANIFOLDX_CORE=python"
        )
    from manifoldx._desktop import _NativeEngine

    return _NativeEngine(*args, **kwargs)
