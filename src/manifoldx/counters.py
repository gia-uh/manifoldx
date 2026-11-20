"""What the engine reports while it works, for `manifoldx bench` to record.

The engine calls `add` where it does work worth counting (a draw, a bundle recorded, a material read),
`event` where a streamer changes a node, and `write_buffer` for every upload to the GPU. Apps call
`span`, `metric` and `worker_span` through `manifoldx.bench`. Outside a run every call is one branch:
nothing is kept and nothing is written.

A worker process has no channel back to the run, so `worker_span` writes one JSON line per job to a
file of its own in the folder named by the environment variable MANIFOLDX_BENCH_JOBS, which the run
sets before the app starts its workers and reads at the end.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

JOBS_ENV = "MANIFOLDX_BENCH_JOBS"  # the folder worker processes write their job lines to

_on = False
_t0 = 0.0  # perf_counter at start(); perf_counter is system-wide, so a worker process's job times compare with it
frame = 0  # the runner's frame number, stamped on events
system = threading.local()  # .name: the system this thread runs; Engine._in_system sets it
_counts: dict = {}
_spans: dict = {}
_metrics: dict = {}
_events: list = []
_jobs_dir: Path | None = None
_lock = threading.Lock()  # counts, spans and metrics: systems add from several threads
_job_lock = threading.Lock()
_job_file = None  # (folder, open file) of this process


@dataclass
class Frame:
    """What one frame added: counts, span milliseconds by name, and the app's metrics."""

    counts: dict = field(default_factory=dict)
    spans: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


def now() -> float:
    """Seconds since start(), on the clock events carry."""
    return time.perf_counter() - _t0


def recording() -> bool:
    """Whether a run is recording."""
    return _on


def add(name: str, n: int = 1) -> None:
    """Count n of `name` in this frame."""
    if _on:
        with _lock:
            _counts[name] = _counts.get(name, 0) + n


def write_buffer(queue, buffer, offset, data) -> None:
    """queue.write_buffer, counting the bytes as bytes_uploaded; every upload in manifoldx goes through it."""
    if _on:
        n = memoryview(data).nbytes
        with _lock:
            _counts["bytes_uploaded"] = _counts.get("bytes_uploaded", 0) + n
    queue.write_buffer(buffer, offset, data)


def event(kind: str, **fields) -> None:
    """Note an event with its time (seconds since the run started) and frame."""
    if _on:
        _events.append({"t": round(time.perf_counter() - _t0, 5), "frame": frame, "kind": kind, **fields})


def metric(name: str, value) -> None:
    """Publish an app's number for this frame; the run keeps it under `app`."""
    if _on:
        with _lock:
            _metrics[name] = value


@contextmanager
def span(name: str):
    """Time a named part of a system; it is reported as `<system>/<name>`."""
    if not _on:
        yield
        return
    owner = getattr(system, "name", None)
    key = f"{owner}/{name}" if owner else name
    t = time.perf_counter()
    try:
        yield
    finally:
        ms = 1000.0 * (time.perf_counter() - t)
        with _lock:
            _spans[key] = _spans.get(key, 0.0) + ms


@contextmanager
def worker_span(kind: str, key=None):
    """Time a job in a worker, a thread or a process, and write it where the run reads it."""
    where = os.environ.get(JOBS_ENV)
    if not where:
        yield
        return
    t = time.perf_counter()
    try:
        yield
    finally:
        _write_job({"kind": kind, "key": _plain(key), "at": t, "ms": 1000.0 * (time.perf_counter() - t),
                    "pid": os.getpid(), "thread": threading.get_ident()}, where)


def _plain(key):
    if isinstance(key, (tuple, list)):
        return [_plain(k) for k in key]
    if hasattr(key, "item"):  # a numpy scalar
        return key.item()
    return key


def _write_job(line: dict, where: str) -> None:
    """Append a job line; a job that ends after its run removed the folder is dropped, never raised."""
    global _job_file
    with _job_lock:
        try:
            if _job_file is None or _job_file[0] != where:
                if _job_file is not None:
                    _job_file[1].close()
                _job_file = None
                _job_file = (where, open(Path(where) / f"jobs-{os.getpid()}.jsonl", "a", buffering=1))
            _job_file[1].write(json.dumps(line) + "\n")
        except OSError:
            pass


def start(jobs_dir=None) -> None:
    """Start recording, from zero. With a folder, worker spans write their jobs there."""
    global _on, _t0, frame, _counts, _spans, _metrics, _events, _jobs_dir
    _counts, _spans, _metrics, _events, frame = {}, {}, {}, [], 0
    _t0 = time.perf_counter()
    _jobs_dir = Path(jobs_dir) if jobs_dir is not None else None
    if _jobs_dir is not None:
        _jobs_dir.mkdir(parents=True, exist_ok=True)
        os.environ[JOBS_ENV] = str(_jobs_dir)
    _on = True


def stop() -> None:
    """Stop recording, close this process's job file and remove the jobs folder."""
    global _on, _job_file, _jobs_dir
    _on = False
    with _job_lock:
        if _job_file is not None:
            _job_file[1].close()
            _job_file = None
    os.environ.pop(JOBS_ENV, None)
    if _jobs_dir is not None:
        shutil.rmtree(_jobs_dir, ignore_errors=True)
        _jobs_dir = None


def take() -> Frame:
    """This frame's counts, spans and metrics; the next frame starts empty.

    Under the lock every add, so a thread adding while the frame ends adds to one frame or the next."""
    global _counts, _spans, _metrics
    with _lock:
        f = Frame(_counts, _spans, _metrics)
        _counts, _spans, _metrics = {}, {}, {}
    return f


def events() -> list:
    """The events so far, in the order they were noted."""
    return list(_events)


def jobs() -> list:
    """Every job line written so far, with `t`: its start in seconds since start()."""
    if _jobs_dir is None or not _jobs_dir.exists():
        return []
    with _job_lock:
        if _job_file is not None:
            _job_file[1].flush()
    out = []
    for p in sorted(_jobs_dir.glob("jobs-*.jsonl")):
        for line in p.read_text().splitlines():
            try:
                j = json.loads(line)
            except ValueError:  # empty, or a worker still writing it
                continue
            j["t"] = round(j.pop("at") - _t0, 5)
            out.append(j)
    return sorted(out, key=lambda j: j["t"])
