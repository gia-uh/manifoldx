"""manifoldx.counters: what the engine reports while it works, kept only while a run records."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from manifoldx import counters


@pytest.fixture
def rec(tmp_path):
    counters.start(tmp_path / "jobs")
    yield tmp_path / "jobs"
    counters.stop()


def test_nothing_is_kept_outside_a_run():
    counters.add("draws")
    counters.event("node", ev="drawn")
    counters.metric("cars", 3)
    with counters.span("walls"):
        pass
    assert not counters.recording()
    counters.start(None)
    try:
        f = counters.take()
        assert f.counts == {} and f.spans == {} and f.metrics == {} and counters.events() == []
    finally:
        counters.stop()


def test_counts_spans_and_metrics_are_per_frame(rec):
    counters.add("draws")
    counters.add("draws", 2)
    counters.metric("cars", 7)
    with counters.span("walls"):
        pass
    f = counters.take()
    assert f.counts == {"draws": 3} and f.metrics == {"cars": 7} and set(f.spans) == {"walls"}
    assert counters.take().counts == {}  # taken: the next frame starts at zero


def test_a_span_inside_a_system_is_named_after_it(rec):
    counters.system.name = "App.drive"
    try:
        with counters.span("walls"):
            pass
    finally:
        counters.system.name = None
    assert set(counters.take().spans) == {"App.drive/walls"}


def test_write_buffer_counts_bytes_and_writes():
    class Q:
        def write_buffer(self, buf, off, data):
            self.got = (buf, off, bytes(data))

    q = Q()
    counters.write_buffer(q, "a", 0, b"\x00")  # outside a run: written, not counted
    assert q.got == ("a", 0, b"\x00")
    counters.start(None)
    try:
        counters.write_buffer(q, "b", 4, b"\x01\x02\x03")
        assert q.got == ("b", 4, b"\x01\x02\x03") and counters.take().counts == {"bytes_uploaded": 3}
    finally:
        counters.stop()


def test_events_carry_time_and_frame(rec):
    counters.frame = 5
    counters.event("node", ev="drawn", key=[3, 1, 2])
    (e,) = counters.events()
    assert e["kind"] == "node" and e["frame"] == 5 and e["t"] >= 0 and e["key"] == [3, 1, 2]


def _job(k):
    from manifoldx import counters

    with counters.worker_span("cell", (k, k)):
        return os.getpid()


def test_worker_spans_come_back_from_processes(rec):
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context("spawn")) as pool:
        pids = set(pool.map(_job, range(6)))
    jobs = counters.jobs()
    assert len(jobs) == 6 and {j["pid"] for j in jobs} == pids
    assert all(j["kind"] == "cell" and j["ms"] >= 0 and j["t"] >= 0 for j in jobs)
    assert sorted(tuple(j["key"]) for j in jobs) == [(k, k) for k in range(6)]


def test_worker_spans_from_threads_share_one_file(rec):
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(_job, range(20)))
    assert len(counters.jobs()) == 20


def test_stop_removes_the_jobs_folder(tmp_path):
    counters.start(tmp_path / "jobs")
    with counters.worker_span("cell", 1):
        pass
    counters.stop()
    assert not (tmp_path / "jobs").exists() and os.environ.get(counters.JOBS_ENV) is None


def test_a_worker_span_outside_a_run_writes_nothing(tmp_path):
    with counters.worker_span("cell", 1):
        pass
    assert list(tmp_path.iterdir()) == []
