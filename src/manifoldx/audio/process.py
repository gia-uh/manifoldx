"""The sound process: it builds the app's sources, mixes a block at a time and feeds a backend.

It runs apart from the app, started with the spawn context, so neither the app's frame nor
its garbage collector, which stops every thread of a process, can make it miss a block. The
app reaches it through three channels: controls in shared memory (read when a block is
rendered), a queue of messages (register a source, fire an event, change a gain) and one
queue of chunks per stream.
"""

from __future__ import annotations

import collections
import gc
import importlib
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

from manifoldx.audio.controls import Controls
from manifoldx.audio.mixer import BUSES, RATE, Mixer, Placed, Pose

RING = 3  # blocks rendered ahead of the device
MAX_VOICE = 10.0  # seconds a one-shot without a duration plays
POSE = ("x", "y", "z", "vx", "vy", "vz")


def pose_names(prefix: str) -> list[str]:
    """The six control names a placed source's pose is read from."""
    return [f"{prefix}.{k}" for k in POSE]


def state_names(streams) -> list[str]:
    """The slots the sound process reports through."""
    names = ["alive", "null", "cpu", "underruns", "time"]
    names += [f"voices.{b}" for b in BUSES] + [f"level.{b}" for b in BUSES]
    return names + [f"stream.{s}" for s in streams]


def _duration(obj) -> float:
    d = getattr(obj, "duration", None)
    if callable(d):
        d = d()
    return MAX_VOICE if d is None else min(float(d), MAX_VOICE)


class _Wav:
    """A 16-bit stereo WAV written block by block."""

    def __init__(self, path: Path, rate: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.w = wave.open(str(path), "wb")
        self.w.setnchannels(2)
        self.w.setsampwidth(2)
        self.w.setframerate(rate)

    def write(self, x: np.ndarray) -> None:
        self.w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())

    def close(self) -> None:
        self.w.close()


class Engine:
    """What the sound process holds: the mixer, the sources, the voices and the streams."""

    def __init__(self, ctl: Controls, block: int):
        self.ctl, self.block = ctl, block
        self.mixer = Mixer()
        self.sources: dict[str, tuple] = {}  # name -> (unit, bus, Placed | None, prefix)
        self.kinds: dict[str, tuple] = {}  # event name -> (factory, bus, cap, placed)
        self.voices: dict[str, list] = {b: [] for b in BUSES}  # [unit, samples left, Placed | None, pose]
        self.streams: dict[str, dict] = {}  # name -> {"bus", "queue", "parts", "clock"}
        self.read = ctl.get

    def pose(self, prefix: str) -> Pose:
        g = self.ctl.get
        return Pose(*(g(f"{prefix}.{k}") for k in POSE))

    def listener(self) -> Pose:
        g = self.ctl.get
        return Pose(*(g(f"listener.{k}") for k in POSE), yaw=g("listener.yaw"))

    def prepare(self, msg: tuple):
        """Do the slow part of a message (unpickled, a factory run, a module reloaded) on the
        builder thread, and return the quick change the render loop makes, or None."""
        kind = msg[0]
        if kind == "source":
            _, name, factory, bus, placed = msg
            unit = factory(self.read)
            p = Placed(unit, placed[1], placed[2]) if placed else None
            return lambda: self.sources.__setitem__(name, (unit, bus, p, placed[0] if placed else None))
        if kind == "remove":
            return lambda: self.sources.pop(msg[1], None)
        if kind == "voice":
            _, name, factory, bus, cap, placed = msg
            self.kinds[name] = (factory, bus, cap, placed)  # read only here, on the builder thread
            return None
        if kind == "event":
            return self._voice(msg[1], msg[2])
        if kind == "gain":
            return lambda: self.mixer.gain(msg[1], msg[2])
        if kind == "duck":
            return lambda: self.mixer.duck(msg[1], msg[2], msg[3])
        if kind == "lowpass":
            return lambda: self.mixer.lowpass(msg[1])
        if kind == "reload":
            for m in msg[1]:
                if m in sys.modules:
                    importlib.reload(sys.modules[m])
                else:
                    importlib.import_module(m)
        return None

    def _voice(self, name: str, values: dict):
        if name not in self.kinds:
            return None
        factory, bus, cap, placed = self.kinds[name]
        pose = None
        if placed:  # the position places the voice; the factory gets the other values
            values = dict(values)
            pose = Pose(values.pop("x", 0.0), values.pop("y", 0.0), values.pop("z", 0.0), 0, 0, 0)
        unit = factory(**values)
        voice = [unit, round(_duration(unit) * RATE), Placed(unit) if placed else None, pose, name]

        def start():
            live = self.voices[bus]
            same = [v for v in live if v[4] == name]
            if len(same) >= cap:
                live.remove(same[0])
            live.append(voice)
        return start

    def render(self) -> np.ndarray:
        n = self.block
        lis = self.listener()
        for unit, bus, placed, prefix in self.sources.values():
            x = placed.block(n, self.pose(prefix), lis) if placed else unit.block(n)
            self.mixer[bus].add(x)
        for bus, live in self.voices.items():
            for v in list(live):
                x = v[2].block(n, v[3], lis) if v[2] else v[0].block(n)
                self.mixer[bus].add(x)
                v[1] -= n
                if v[1] <= 0:
                    live.remove(v)
        for s in self.streams.values():
            self.mixer[s["bus"]].add(self._stream_block(s, n))
        return self.mixer.out(n)

    def _stream_block(self, s: dict, n: int) -> np.ndarray:
        while True:
            try:
                start, samples = s["queue"].get_nowait()
            except queue.Empty:
                break
            x = np.asarray(samples, np.float32)
            s["parts"].append((round(start * RATE), np.stack([x, x], 1) if x.ndim == 1 else x))
        out = np.zeros((n, 2), np.float32)
        a, b = s["clock"], s["clock"] + n
        keep = []
        for start, x in s["parts"]:
            lo, hi = max(a, start), min(b, start + len(x))
            if hi > lo:
                out[lo - a:hi - a] += x[lo - start:hi - start]
            if start + len(x) > b:
                keep.append((start, x))
        s["parts"] = keep
        s["clock"] = b
        return out


def main(cspec, sspec, error, msgs, streams, config) -> None:
    """The sound process's loop. config: rate, block, backend, path, stems."""
    gc.disable()  # a collection stops every thread: the render loop allocates no cycles, so none is needed
    ctl, st = Controls.attach(cspec), Controls.attach(sspec)
    rate, block = config["rate"], config["block"]
    eng = Engine(ctl, block)
    for name, (bus, q) in streams.items():
        eng.streams[name] = {"bus": bus, "queue": q, "parts": [], "clock": 0}
    backend = config["backend"]
    ring: collections.deque = collections.deque()
    counters = {"underruns": 0}
    device = None

    def fail(why: str) -> None:
        error.value = why.encode()[:255]

    if backend == "device":
        try:
            import sounddevice as sd

            name = os.environ.get("MANIFOLDX_AUDIO_DEVICE") or None

            def callback(out, frames, _time, _status):
                try:
                    out[:] = ring.popleft()
                except IndexError:
                    out[:] = 0
                    counters["underruns"] += 1

            device = sd.OutputStream(samplerate=rate, blocksize=block, channels=2, dtype="float32",
                                     callback=callback, device=name)
            device.start()
        except Exception as e:  # no PortAudio, no such device, a busy device: play on in silence
            fail(f"audio device unavailable: {e}")
            print(f"manifoldx.audio: {e}; playing silently", file=sys.stderr)
            backend, device = "null", None
    st.set("null", 1.0 if backend == "null" else 0.0)
    files = {}
    if backend == "file":
        path = Path(config["path"])
        files["master"] = _Wav(path, rate)
        if config.get("stems"):
            for b in BUSES:
                files[b] = _Wav(path.with_name(f"{path.stem}.{b}{path.suffix}"), rate)

    parent = mp.parent_process()
    ready: collections.deque = collections.deque()  # quick changes for the render loop
    flags = {"stop": False}

    def builder():
        """Takes messages off the queue and prepares them, so a slow factory or an import
        never holds up a block."""
        while not flags["stop"]:
            try:
                msg = msgs.get(timeout=0.2)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                flags["stop"] = True
                break
            if msg[0] == "stop":
                flags["stop"] = True
                break
            try:
                op = eng.prepare(msg)
            except Exception as e:  # a factory that fails must not stop the sound
                fail(f"{msg[0]} {msg[1]!r}: {e}")
                print(f"manifoldx.audio: {msg[0]} {msg[1]!r} failed: {e}", file=sys.stderr)
                continue
            if op is not None:
                ready.append(op)
            if msg[0] in ("source", "voice"):  # what was built lives on: keep the collector off it
                gc.freeze()

    threading.Thread(target=builder, name="audio-builder", daemon=True).start()
    st.set("alive", 1.0)
    t0, produced, was_late = time.monotonic(), 0, False
    cpu_t, wall_t, last_check = time.process_time(), time.monotonic(), time.monotonic()
    while not flags["stop"]:
        while ready:
            ready.popleft()()
        now = time.monotonic()
        if device is not None:
            ahead = len(ring) < RING
        else:
            played = (now - t0) * rate
            late = produced + block < played - RING * block  # a device would have run dry
            if late and not was_late:  # one underrun per late spell; a file keeps every sample and catches up
                counters["underruns"] += 1
            was_late = late
            ahead = produced < played + RING * block
        if ahead:
            x = eng.render()
            produced += block
            if device is not None:
                ring.append(x)
            for name, f in files.items():
                f.write(x if name == "master" else eng.mixer.last[name])
            st.set("time", produced / rate)
            for b in BUSES:
                st.set(f"voices.{b}", len(eng.voices[b]))
                st.set(f"level.{b}", eng.mixer[b].level)
            for s, d in eng.streams.items():
                st.set(f"stream.{s}", d["clock"] / rate)
        elif device is not None:
            time.sleep(block / rate / 3)
        else:  # until the clock reaches RING blocks before what is produced
            time.sleep(max(0.0, (produced - RING * block) / rate - (time.monotonic() - t0)) + 0.0005)
        if now - last_check > 0.5:
            c, w = time.process_time(), time.monotonic()
            st.set("cpu", (c - cpu_t) / max(w - wall_t, 1e-6))
            cpu_t, wall_t, last_check = c, w, now
            st.set("underruns", counters["underruns"])
            if parent is not None and not parent.is_alive():
                flags["stop"] = True
    st.set("underruns", counters["underruns"])
    st.set("alive", 0.0)
    if device is not None:
        device.stop()
        device.close()
    for f in files.values():
        f.close()
    ctl.close()
    st.close()
