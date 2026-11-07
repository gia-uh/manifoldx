"""manifoldx run --web: rebuild the export on every save and push each version to the open tab.

The WebReloader is a HotReloader with no window. On a save it asks the tab for its
engine.state, merges it into the shared State, reruns the app's setup, builds the export in
memory and writes it as a numbered version beside files every version shares:

.manifoldx/web/
  index.html, browser.js, dev.js, ...   the page, the runtime, vendor/three/, fonts/ (written once)
  v{n}/app/, v{n}/state.json, v{n}/report.json   a version; the last three are kept
  scenes/<sha>.glb                      a scene by the hash of its bytes, written when new

Then it pushes the version message to the tab, which swaps to it in place. A step that fails
pushes a failed message naming the Python line and keeps the running version.
"""

from __future__ import annotations

import gzip
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from manifoldx import devchannel
from manifoldx.devchannel import LOG
from manifoldx.hot import HotReloader, _mtime

from .devstate import encode, merge_tab_state, state_sha
from .export import build_export, page_html, write_static, write_version

STATE_WAIT = 0.5  # seconds a reload waits for the tab's state
KEEP = 3  # versions kept on disk


class NoTab(Exception):
    """No tab is open to answer."""


class Tab:
    """The open tabs. Until a tab connects, calls raise NoTab and pushes go nowhere."""

    def push(self, message):
        pass

    def call(self, kind, body=None, timeout=5.0):
        raise NoTab


def error_line(exc: BaseException, root) -> str:
    """The Python line a failed reload names: `app.py:12: SyntaxError: ...`, the path from root."""
    from .gui import GuiUnsupported
    from .snapshot import SnapshotError
    from .types import TranspileError

    if isinstance(exc, (TranspileError, SnapshotError, GuiUnsupported)):
        return str(exc)
    root = os.path.abspath(root) + os.sep
    kind = type(exc).__name__
    if isinstance(exc, SyntaxError) and exc.filename:
        return f"{os.path.relpath(exc.filename, root)}:{exc.lineno}: {kind}: {exc.msg}"
    where = None
    for frame, line in traceback.walk_tb(exc.__traceback__):
        path = os.path.abspath(frame.f_code.co_filename)
        if path.startswith(root):
            where = f"{os.path.relpath(path, root)}:{line}"
    return f"{where}: {kind}: {exc}" if where else f"{kind}: {exc}"


class WebReloader(HotReloader):
    """Run an app's setup and rebuild its export when its sources change, for the tab."""

    def __init__(
        self,
        script=None,
        module=None,
        argv=(),
        poll=0.25,
        channel=True,
        port=devchannel.PORT,
        web_root=None,
    ):
        super().__init__(
            script=None if script is None else str(script),
            module=module,
            argv=argv,
            poll=poll,
            channel=channel,
            port=port,
        )
        self.web_root = Path(web_root) if web_root else Path(self._root) / devchannel.FOLDER / "web"
        self.tab = Tab()
        self.version = 0
        self.current: dict | None = None  # the last version message
        self.built = None  # the running version's Built
        self._scenes: dict[int, str] = {}  # version: scene sha, of the versions on disk
        self._sizes: dict[str, tuple[int, int]] = {}  # scene sha: (bytes, gzipped bytes)
        self._kept_said: set[str] = set()  # the kept keys already logged this run

    @property
    def label(self) -> str:
        return f"-m {self.module}" if self.module else os.path.relpath(self.script, self._root)

    def start(self):
        engine = self._load()
        if engine is None:
            raise SystemExit(
                f"manifoldx run --web: {self.module or self.script} finished without calling engine.run()"
            )
        self.engine = engine
        write_static(self.web_root, dev=True)
        built = build_export(
            engine,
            self.main,
            self._root,
            app_label=self.label,
            runtime="../../pyrt.js",
            scene_camera=False,
        )
        self._write(built)
        size, gz = self._sizes[built.scene_sha]
        (self.web_root / "index.html").write_text(page_html(built.title, size, gz, dev=True))
        for e in built.left_out:
            _say(f"left out of the state: {e}")
        if self._open_channel:
            self.channel = devchannel.Channel(self._root, self, self.port)
            LOG.info("started %s", self.module or self.script)

    def _write(self, built) -> dict:
        """Write built as the next version and make it current; returns its message."""
        n = self.version + 1
        scene = f"scenes/{built.scene_sha}.glb"
        write_version(built, self.web_root / f"v{n}", scene_path=f"../{scene}")
        if built.scene_sha not in self._sizes:
            self._sizes[built.scene_sha] = (len(built.glb), len(gzip.compress(built.glb, 6)))
        size, gz = self._sizes[built.scene_sha]
        tab_state = {k: v for k, v in self.state.items() if k in built.crossed}
        self.version, self.built = n, built
        self._scenes[n] = built.scene_sha
        self.current = {
            "type": "version",
            "version": n,
            "base": f"v{n}/",
            "scene": scene,
            "scene_sha": built.scene_sha,
            "scene_bytes": size,
            "scene_gzip_bytes": gz,
            "title": built.title,
            "state_sha": state_sha(encode(tab_state)),
        }
        return self.current

    def _prune(self):
        for n in [n for n in self._scenes if n <= self.version - KEEP]:
            shutil.rmtree(self.web_root / f"v{n}", ignore_errors=True)
            del self._scenes[n]
        kept = set(self._scenes.values())
        for f in (self.web_root / "scenes").glob("*"):
            if f.name.split(".")[0] not in kept:
                f.unlink()
                self._sizes.pop(f.name.split(".")[0], None)

    def _tab_state(self):
        """Merge the tab's engine.state into self.state; without a tab, keep it as it is."""
        try:
            reply = self.tab.call("reload_state", timeout=STATE_WAIT)
        except (NoTab, TimeoutError):
            return
        if not reply.get("ok") or reply.get("state") is None:
            return
        kept = merge_tab_state(self.state, reply["state"], self.built.crossed)
        for line in kept:
            if line not in self._kept_said:
                self._kept_said.add(line)
                LOG.info("kept from Python: %s", line)

    def reload(self, reason="edit"):
        """Rebuild from the tab's state and the new code; on failure keep the running version."""
        clock = time.perf_counter()
        began = clock
        laps = {}

        def lap(name):
            nonlocal clock
            now = time.perf_counter()
            laps[name], clock = now - clock, now

        # A save from here on, even during this reload, reloads once more after it.
        seen = {f: _mtime(f) for f in self._mtimes}
        old = self.engine
        try:
            self._tab_state()
            lap("state")
            self._drop_modules()
            engine = self._load()
            if engine is None:
                raise RuntimeError(f"{self.label} finished without calling engine.run()")
            lap("setup")
            gpu = (
                (old._adapter, old._device, old._texture_format)
                if old._device is not None
                else None
            )
            built = build_export(
                engine,
                self.main,
                self._root,
                app_label=self.label,
                runtime="../../pyrt.js",
                gpu=gpu,
                scene_camera=False,
            )
            clock = time.perf_counter()
            scene = "kept" if built.scene_sha == self.built.scene_sha else "new"
            message = self._write(built)
            self._prune()
            lap("write")
        except Exception as e:  # noqa: BLE001  the tab keeps the running version and says why
            text = error_line(e, self._root)
            self.tab.push({"type": "failed", "running": self.version, "text": text})
            _say(f"reload failed, kept v{self.version}: {text}")
            LOG.info("reload failed, kept v%d: %s", self.version, text)
            return
        finally:
            self._mtimes.update({f: t for f, t in seen.items() if f in self._mtimes})
        self.engine = engine
        try:
            old._detach()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        self.tab.push(message)
        t = built.timings
        phases = [("state", laps["state"]), ("setup", laps["setup"])]
        phases += [(k, t[k]) for k in ("startup", "reach", "snapshot", "transpile", "gui")]
        phases += [(f"scene {scene}", t["scene"]), ("write", laps["write"])]
        line = (
            f"reloaded v{self.version} in {time.perf_counter() - began:.2f} s ("
            + ", ".join(f"{k} {v:.2f}" for k, v in phases)
            + ")"
        )
        if reason != "edit":
            line += f", reason {reason}"
        _say(line)
        LOG.info("%s", line)
        for e in built.left_out:
            line = f"{e.path_from_root}: left out of the snapshot"
            if line not in self._kept_said:
                self._kept_said.add(line)
                LOG.info("kept from Python: %s", line)

    def step(self):
        """Reload if a source changed, then run what requests asked of the app."""
        now = time.monotonic()
        if now - self._last_poll >= self.poll:
            self._last_poll = now
            if self.changed():
                self.reload()
        if self.channel is not None:
            self.channel.run_jobs()

    def run(self):
        """Serve the tab and rebuild on every save until interrupted."""
        self.start()
        try:
            while True:
                self.step()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            if self.channel is not None:
                LOG.info("closed")
                self.channel.close()


def _say(line: str) -> None:
    print(f"manifoldx run --web: {line}", file=sys.stderr)
