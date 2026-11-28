"""Load native desktop system declarations without giving app code the Engine."""

from contextvars import ContextVar
from importlib import import_module, invalidate_caches
from importlib.machinery import PathFinder, SourceFileLoader
from inspect import (
    isasyncgenfunction,
    iscoroutinefunction,
    isfunction,
    isgeneratorfunction,
    signature,
)
from pathlib import Path
import runpy
import sys
from time import monotonic
import traceback

from manifoldx.hot import _mtime


_entry = ContextVar("native_run_entry", default=None)
_preparing = ContextVar("native_reload_preparing", default=False)


class _DevChannelHost:
    """What devchannel.Channel asks of a reloader: the engine that is running now.

    `look` reads the next presented frame from the native surface. A surface
    that cannot copy its images answers 501 with that reason.
    """

    looks = True

    def __init__(self, engine):
        self.engine = engine


def _run_app(script, module, argv, port=None):
    from manifoldx._desktop import _NativeEngine
    from manifoldx import devchannel

    path = Path(script).resolve() if script else None
    entry = path or module
    root = path.parent if path else Path.cwd()
    original = _NativeEngine.run

    def run(engine):
        channel = devchannel.Channel(
            root, _DevChannelHost(engine), devchannel.PORT if port is None else port
        )
        engine._dev_channel = channel
        devchannel.LOG.info("started %s", entry)
        print(f"manifoldx run: dev page and API at {channel.url}", file=sys.stderr)
        try:
            return original(engine)
        finally:
            engine._dev_channel = None
            devchannel.LOG.info("closed")
            channel.close()

    _NativeEngine.run = run

    token = _entry.set(entry)
    old_argv, old_path = sys.argv, sys.path[:]
    sys.argv = [str(entry), *argv]
    sys.path.insert(0, str(path.parent if path else Path.cwd()))
    try:
        if module:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv, sys.path[:] = old_argv, old_path
        _entry.reset(token)
        _NativeEngine.run = original


def _check_function(function, *args):
    if (
        not isfunction(function)
        or iscoroutinefunction(function)
        or isgeneratorfunction(function)
        or isasyncgenfunction(function)
    ):
        raise TypeError(f"expected a synchronous function, got {function!r}")
    signature(function).bind(*args)


class _SourceLoader(SourceFileLoader):
    def get_code(self, fullname):
        # A quick same-size save can otherwise select the previous .pyc.
        return self.source_to_code(self.get_data(self.path), self.path)


class _RegistrationReload:
    def __init__(self, names):
        self.names = names
        self.registrations = ()
        self._mtimes = {}
        self._next_poll = 0

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.names:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if (
            spec is None
            or not isinstance(spec.loader, SourceFileLoader)
            or spec.submodule_search_locations is not None
        ):
            raise ImportError(f"reload registration {fullname!r} must be a Python source module")
        source = Path(spec.origin).resolve()
        if _entry.get() in (fullname, source):
            raise ValueError("the application entry cannot be a reload registration module")
        self._mtimes[source] = _mtime(source)
        spec.loader = _SourceLoader(fullname, spec.origin)
        return spec

    def prepare(self, validate):
        """Import a fresh generation and retain it only if all declarations validate."""
        previous = {name: sys.modules.get(name) for name in self.names}
        # Clear the parent attributes too, since `from pkg import child` checks
        # those before importing child. On failure restore the previous generation.
        parents = {}
        for name in self.names:
            package, _, child = name.rpartition(".")
            parent = sys.modules.get(package)
            if parent is not None and child in vars(parent):
                parents[name] = vars(parent).pop(child)
            sys.modules.pop(name, None)
        token = _preparing.set(True)
        sys.meta_path.insert(0, self)
        try:
            invalidate_caches()
            callbacks = []

            def system(callback):
                _check_function(callback, None, 0.0)
                callbacks.append(callback)
                return callback

            for name in self.names:
                module = import_module(name)
                register = getattr(module, "register", None)
                _check_function(register, system)
                register(system)
            self.registrations = validate(callbacks)
        except BaseException:
            for name, old in previous.items():
                sys.modules.pop(name, None)
                if old is not None:
                    sys.modules[name] = old
                package, _, child = name.rpartition(".")
                parent = sys.modules.get(package)
                if parent is not None:
                    vars(parent).pop(child, None)
                    if name in parents:
                        vars(parent)[child] = parents[name]
            raise
        finally:
            sys.meta_path.remove(self)
            _preparing.reset(token)

    def poll(self, validate):
        now = monotonic()
        if now < self._next_poll:
            return False
        self._next_poll = now + 0.25
        mtimes = {path: _mtime(path) for path in self._mtimes}
        if mtimes == self._mtimes:
            return False
        self._mtimes = mtimes
        try:
            self.prepare(validate)
        except Exception, SystemExit:
            print(
                "manifoldx run: registration reload failed; keeping the running systems",
                file=sys.stderr,
            )
            traceback.print_exc()
            return False
        return True
