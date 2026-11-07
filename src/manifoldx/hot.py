"""Hot reload for `manifoldx run`: rerun the app on every source change, in the same window.

Each reload rebuilds the scene from the new code. Only `engine.state` carries over,
so keep there what should survive (camera angles, animation step, game progress),
as plain data: numbers, strings, lists, dicts, numpy arrays. Instances of classes
from your own modules go stale, because those modules are imported afresh.
"""

import importlib.util
import os
import runpy
import sys
import time
import traceback

from manifoldx import devchannel
from manifoldx.devchannel import LOG

_state = None  # the State every Engine shares while HotReloader loads an app

_MANIFOLDX_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


class State(dict):
    """A dict whose keys also read and write as attributes: `engine.state.camera`."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


def _mtime(path):
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _desktop_canvas(engine):
    from manifoldx.backends import get_desktop_canvas

    return get_desktop_canvas(
        width=engine.w, height=engine.h, fullscreen=engine.fullscreen, title=engine.title, monitor=engine.monitor
    )


class HotReloader:
    """Run an app (a script, or a module like `python -m`) and reload it when its sources change.

    The app's modules are those loaded from the script's folder (the working
    directory for a module), outside site-packages and manifoldx itself.
    """

    def __init__(self, script=None, module=None, argv=(), canvas_factory=None, poll=0.25, channel=True,
                 port=devchannel.PORT, debug=False):
        self.script = os.path.abspath(script) if script else None
        self.module = module
        self.argv = list(argv)
        self.canvas_factory = canvas_factory or _desktop_canvas
        self.poll = poll  # seconds between checks for changed files
        self.state = State()
        self.engine = None
        self.main = None  # the app's __main__ module when it called engine.run()
        self._root = (os.path.dirname(self.script) if script else os.getcwd()) + os.sep
        self._mtimes = {}
        self._last_poll = time.monotonic()
        self._paused = False  # the running code raised; wait for an edit
        self._open_channel = channel  # the dev page and API, and .manifoldx/ (devchannel)
        self.port = port
        self.debug = debug  # the channel's watchers: metrics, slow frames and collections, keys
        self.channel = None

    def _owned(self, path):
        path = os.path.abspath(path)
        return path.startswith(self._root) and "site-packages" not in path and not path.startswith(_MANIFOLDX_DIR)

    def _load(self):
        """Run the app until it calls engine.run(); return that engine, or None if it never did."""
        global _state
        from manifoldx.engine import Engine

        captured = []
        original, saved_argv, saved_path = Engine.run, sys.argv, list(sys.path)
        Engine.run = lambda engine: captured.append((engine, sys.modules.get("__main__")))
        sys.argv = [self.module or self.script, *self.argv]
        sys.path.insert(0, self._root.rstrip(os.sep))
        _state = self.state
        try:
            if self.module:
                runpy.run_module(self.module, run_name="__main__", alter_sys=True)
            else:
                runpy.run_path(self.script, run_name="__main__")
        finally:
            entry = self._entry()  # while the app's folder is still on sys.path
            _state = None
            Engine.run, sys.argv = original, saved_argv
            sys.path[:] = saved_path
            # Keep watching files a broken edit failed to import, so fixing them reloads.
            files = set(self._mtimes) | entry
            files |= {m.__file__ for m in list(sys.modules.values()) if getattr(m, "__file__", None) and self._owned(m.__file__)}
            self._mtimes = {f: _mtime(f) for f in files}
        if not captured:
            return None
        engine, self.main = captured[-1]  # the app's __main__ module, for export --web
        return engine

    def _entry(self):
        """The file the app starts from: the script, or the -m module's, which runs as __main__
        and so never stays in sys.modules to be watched like the modules it imports."""
        if self.script:
            return {self.script}
        try:
            spec = importlib.util.find_spec(self.module)
        except (ImportError, ValueError):
            return set()
        return {spec.origin} if spec and spec.origin and self._owned(spec.origin) else set()

    def _activate(self, engine):
        engine._running = True
        engine._startup_pending = True
        self.engine = engine
        self._paused = False

    def start(self):
        engine = self._load()
        if engine is None:
            raise SystemExit(f"manifoldx run: {self.module or self.script} finished without calling engine.run()")
        engine._init_canvas(self.canvas_factory(engine))
        self._activate(engine)
        if self._open_channel:
            self.channel = devchannel.Channel(self._root, self, self.port, debug=self.debug)
            LOG.info("started %s", self.module or self.script)
            print(f"manifoldx run: dev page and API at {self.channel.url}", file=sys.stderr)

    def changed(self):
        return any(_mtime(f) != t for f, t in self._mtimes.items())

    def _drop_modules(self):
        """Forget the app's modules, so the next _load imports them afresh."""
        dropped = [name for name, m in list(sys.modules.items())
                   if getattr(m, "__file__", None) and self._owned(m.__file__)]
        for name in dropped:
            module = sys.modules.pop(name)
            # A parent that stays (a namespace package has no file) still holds the old module as an
            # attribute, and `from pkg import mod` would hand it back instead of importing the new one.
            parent, _, child = name.rpartition(".")
            if parent in sys.modules and getattr(sys.modules[parent], child, None) is module:
                delattr(sys.modules[parent], child)

    def reload(self):
        """Rerun the app on the open window; on failure keep the running version."""
        self._drop_modules()
        try:
            engine = self._load()
        except Exception as e:
            traceback.print_exc()
            print("manifoldx run: reload failed, kept the running version", file=sys.stderr)
            LOG.info("reload failed, kept the running version: %r", e)
            return
        if engine is None:
            print("manifoldx run: the app no longer calls engine.run(), kept the running version", file=sys.stderr)
            return
        old = self.engine
        try:
            old._detach()
        except Exception:
            traceback.print_exc()
        engine._init_canvas(old._render_canvas, gpu=(old._adapter, old._device, old._texture_format))
        self._activate(engine)
        print("manifoldx run: reloaded", file=sys.stderr)
        LOG.info("reloaded")

    def step(self):
        """One frame: reload if a source changed, then tick the engine unless paused on an error."""
        now = time.monotonic()
        if now - self._last_poll >= self.poll:
            self._last_poll = now
            if self.changed():
                self.reload()
        if self.channel is not None:
            self.channel.run_jobs()
        if self._paused:
            return
        try:
            self.engine._tick()
        except Exception as e:
            traceback.print_exc()
            print("manifoldx run: paused until the next edit", file=sys.stderr)
            LOG.info("paused until the next edit: %r", e)
            self._paused = True
            return
        if self.channel is not None:
            self.channel.frame(self.engine)

    def run(self):
        """Open the window and loop until it closes."""
        from rendercanvas.glfw import loop

        self.start()
        canvas = self.engine._render_canvas

        def frame():
            self.step()
            canvas.request_draw(frame)

        canvas.request_draw(frame)
        try:
            loop.run()
        finally:
            self.engine._running = False
            self.engine.shutdown_events()
            if self.channel is not None:
                LOG.info("closed")
                self.channel.close()
