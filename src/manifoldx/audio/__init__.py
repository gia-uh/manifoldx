"""Sound for manifoldx apps: play, mix and place what an app gives it.

manifoldx makes no sound of its own. An app registers sources (anything with block(n) that
answers n float32 samples, mono or stereo), one-shot voices fired by events, and streams of
chunks rendered ahead; manifoldx mixes them into five buses, places them around a listener and
plays them. All of it runs in a sound process, so the app's frame never waits on audio and a
garbage collection in the app never stops the sound.

    audio = Audio(controls=["rpm", *pose_names("car")])
    audio.source("engine", make_engine, "engine", placed=("car", 5.0, 300.0))
    audio.set("rpm", 2400.0); audio.tick()        # once per frame

A factory runs inside the sound process: it must be picklable (a module-level function or a
functools.partial of one). factory(ctl) builds a source, where ctl(name) reads a control.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from collections.abc import Callable, Sequence
from pathlib import Path

from manifoldx.audio.controls import Controls
from manifoldx.audio.mixer import BUSES, Mixer, Placed, Pose
from manifoldx.audio.process import main, pose_names, state_names

__all__ = ["Audio", "BUSES", "Controls", "Mixer", "Placed", "Pose", "StreamWriter", "pose_names"]

LISTENER = [*pose_names("listener"), "listener.yaw"]


class StreamWriter:
    """Pushes chunks of samples into one stream of the sound process; picklable, so another
    process (a music composer) can hold it."""

    def __init__(self, name: str, q, sspec: dict):
        self.name, self._q, self._sspec, self._st = name, q, sspec, None

    def __getstate__(self):
        return {"name": self.name, "_q": self._q, "_sspec": self._sspec, "_st": None}

    def push(self, start: float, samples) -> None:
        """Add samples (n,) or (n, 2) at start seconds of the stream's clock; the chunks overlap
        and add, so a chunk may carry a tail past the next one's start."""
        self._q.put((float(start), samples))

    def clock(self) -> float:
        """Seconds of this stream the sound process has played."""
        if self._st is None:
            self._st = Controls.attach(self._sspec)
        return self._st.get(f"stream.{self.name}")


class Audio:
    """The app's handle on the sound process. Every call returns without waiting: a full queue
    drops the message, and once the process is gone the calls do nothing."""

    def __init__(self, rate: int = 48_000, block: int = 512, backend: str = "device", path=None,
                 stems: bool = False, controls: Sequence[str] = (), streams: dict[str, str] | None = None):
        if backend not in ("device", "file", "null"):
            raise ValueError(f"backend must be device, file or null, not {backend!r}")
        if backend == "file" and path is None:
            raise ValueError("the file backend needs a path")
        streams = streams or {}
        ctx = mp.get_context("spawn")
        self.controls = Controls.create(list(dict.fromkeys([*controls, *LISTENER])))
        self._st = Controls.create(state_names(streams))
        self._error = ctx.RawArray("c", 256)
        self._msgs = ctx.Queue(maxsize=4096)
        qs = {name: (bus, ctx.Queue(maxsize=64)) for name, bus in streams.items()}
        self._writers = {name: StreamWriter(name, q, self._st.spec()) for name, (bus, q) in qs.items()}
        config = {"rate": rate, "block": block, "backend": backend, "stems": stems,
                  "path": str(Path(path)) if path is not None else None}
        self._process = ctx.Process(target=main, args=(self.controls.spec(), self._st.spec(), self._error,
                                                       self._msgs, qs, config), daemon=True)
        self._process.start()
        self._closed = False
        self.backend = backend

    @property
    def alive(self) -> bool:
        return not self._closed and self._process.is_alive()

    def _send(self, *msg) -> None:
        if not self.alive:
            return
        try:
            self._msgs.put_nowait(msg)
        except (queue.Full, ValueError, OSError):
            pass

    def source(self, name: str, factory: Callable, bus: str, placed: tuple | None = None) -> None:
        """Register (or replace) a source: factory(ctl) runs in the sound process. placed is None
        or (prefix, ref, far), and the pose is read from the controls pose_names(prefix)."""
        self._send("source", name, factory, bus, placed)

    def remove(self, name: str) -> None:
        self._send("remove", name)

    def voice(self, event: str, factory: Callable, bus: str, cap: int = 4, placed: bool = False) -> None:
        """Register what an event plays: factory(**values) builds a one-shot that plays for its
        duration (10 s without one). Past cap at once, the oldest is cut. A placed one-shot
        stands at the event's x, y, z."""
        self._send("voice", event, factory, bus, cap, placed)

    def event(self, name: str, **values) -> None:
        self._send("event", name, values)

    def stream(self, name: str) -> StreamWriter:
        """The writer of a stream declared with Audio(streams={name: bus})."""
        return self._writers[name]

    def set(self, name: str, value: float) -> None:
        if not self._closed:
            self.controls.set(name, value)

    def tick(self) -> None:
        """End the frame: what was set becomes the previous value of each control."""
        if not self._closed:
            self.controls.tick()

    def gain(self, bus: str, value: float) -> None:
        self._send("gain", bus, float(value))

    def duck(self, bus: str, db: float, seconds: float) -> None:
        self._send("duck", bus, float(db), float(seconds))

    def lowpass(self, cutoff_hz: float) -> None:
        self._send("lowpass", float(cutoff_hz))

    def reload(self, modules: list[str]) -> None:
        """Re-import modules in the sound process before the registrations that follow, so a
        factory edited while the app runs is the one used."""
        self._send("reload", list(modules))

    def state(self) -> dict:
        """Whether the process runs, its backend and error, its CPU share, underruns, seconds
        played, and the voices and level (dBFS) of each bus."""
        if self._closed:
            return {"alive": False, "backend": None, "error": "closed"}
        g = self._st.get
        return {
            "alive": self.alive and g("alive") == 1.0,
            "backend": "null" if g("null") == 1.0 else self.backend,
            "error": self._error.value.decode(errors="replace"),
            "cpu": round(g("cpu"), 3),
            "underruns": int(g("underruns")),
            "time": round(g("time"), 3),
            "voices": {b: int(g(f"voices.{b}")) for b in BUSES},
            "levels": {b: round(g(f"level.{b}"), 1) for b in BUSES},
        }

    def close(self) -> None:
        """Stop the process, finish its files and free the shared memory."""
        if self._closed:
            return
        self._send("stop")
        self._process.join(3)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(1)
        self._closed = True
        for c in (self.controls, self._st):
            c.close()
            c.unlink()
