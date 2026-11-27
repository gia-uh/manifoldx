"""Reload a game's native cell builder without losing the last valid build.

A game exports its builder as a ``manifoldx.builder.v1`` capsule
(``crates/manifoldx-sdk``). A rebuild is a new cdylib. This loader copies the
bytes to a content-addressed filename before opening them, so a regenerated file
never resolves to the platform's copy of an older mapping, and opens only the
capsule name it already held. The active builder changes after the new capsule
validates; a bad library leaves the previous builder serving calls.

No library is unloaded. The old module and builder stay referenced for work that
was already in flight, and for the rest of the process.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import sys
import tempfile
import threading
from pathlib import Path

_SUFFIX = ".pyd" if os.name == "nt" else ".so"


def _module_name(path: Path) -> str:
    """The extension's own module name, from its filename as Python names it.

    ``libnative.so`` exports ``PyInit_native``; ``native.cpython-...so`` exports
    ``PyInit_native``; ``native.pyd`` exports ``PyInit_native``.
    """
    name = path.name
    for suffix in sorted(
        (*importlib.machinery.EXTENSION_SUFFIXES, ".so", ".pyd", ".dylib"), key=len, reverse=True
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    else:
        name = path.stem
    if os.name != "nt" and name.startswith("lib"):
        name = name[3:]
    if not name.isidentifier():
        raise ValueError(
            f"{path.name}: the filename must name the extension module "
            "(a Python identifier); rename the build output"
        )
    return name


class NativeBuilder:
    """A reloadable native cell builder, usable as ``Streamer(build=...)``.

    ``poll()`` once a frame reloads ``path`` when it changed. ``reload()`` loads
    explicitly and raises on a bad capsule; ``poll()`` records the error in
    ``last_error`` and keeps the active builder. Builds that started before a
    swap finish with the builder they started on.

    The source filename's stem names the extension module (``libnative.so``
    exports ``PyInit_native``), so rebuilds keep that name and the loader
    versions its own staged copy.
    """

    def __init__(self, path, *, attribute="build_cell"):
        self.attribute = attribute
        self.path = Path(path)
        self.last_error = None
        self._active = None
        self._version = None
        self._stamp = None
        self._libraries = []
        self._state = threading.Lock()
        self._reload = threading.Lock()
        self._cache = Path(tempfile.mkdtemp(prefix="manifoldx-native-"))
        self.reload(self.path)

    @property
    def version(self):
        """SHA-256 of the active build."""
        return self._version

    def build(self, x, y, size, lod, seed):
        with self._state:
            active = self._active
        if active is None:
            raise RuntimeError("native builder has no active library")
        return active.build(x, y, size, lod, seed)

    def reload(self, path=None):
        """Load and validate a build, then make it active. Raises on failure."""
        target = Path(path) if path is not None else self.path
        with self._reload:
            builder, digest, stamp = self._open(target)
            with self._state:
                self._active = builder
                self._version = digest
                self.path = target
                self._stamp = stamp
            self.last_error = None
            return digest

    def poll(self):
        """Reload if ``path`` changed; keep the active builder if it is invalid."""
        try:
            stamp = self._stamp_of(self.path)
        except OSError as error:
            self.last_error = error
            return False
        if stamp == self._stamp:
            return False
        try:
            self.reload(self.path)
            return True
        except Exception as error:
            self._stamp = stamp
            self.last_error = error
            return False

    def _open(self, path: Path):
        with path.open("rb") as source:
            stamp = self._file_identity(os.fstat(source.fileno()))
            data = source.read()
            if self._file_identity(os.fstat(source.fileno())) != stamp:
                raise ValueError(f"{path}: native build changed during read")
        if not data:
            raise ValueError(f"{path}: native build is empty")
        digest = hashlib.sha256(data).hexdigest()
        staged = self._cache / f"{_module_name(path)}-{digest[:16]}{_SUFFIX}"
        if not staged.exists():
            staged.write_bytes(data)
            staged.chmod(0o700)
        module = self._load_module(path, staged)
        # Every dlopen stays referenced, including one whose capsule is rejected:
        # unloading native code is not safe.
        self._libraries.append(module)
        from manifoldx import _core_native

        try:
            builder = _core_native._StreamBuilder(getattr(module, self.attribute))
        except AttributeError as error:
            raise ValueError(f"{path}: no {self.attribute!r} capsule") from error
        except Exception as error:
            raise ValueError(f"{path}: capsule ABI rejected: {error}") from error
        return builder, digest, stamp

    @staticmethod
    def _load_module(path: Path, staged: Path):
        name = _module_name(path)
        previous = sys.modules.pop(name, None)
        try:
            spec = importlib.util.spec_from_file_location(name, staged)
            if spec is None or spec.loader is None:
                raise ImportError(f"{path}: not a loadable native extension")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(name, None)
            if previous is not None:
                sys.modules[name] = previous
        return module

    def _stamp_of(self, path: Path):
        try:
            status = path.stat()
        except OSError as error:
            raise FileNotFoundError(f"{path}: native build is not readable") from error
        return self._file_identity(status)

    @staticmethod
    def _file_identity(status):
        return status.st_dev, status.st_ino, status.st_mtime_ns, status.st_size
