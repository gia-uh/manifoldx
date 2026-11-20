"""The runner: phases, loops, fixed step, host, shots, failures and the run folder."""

import importlib.metadata
import json
import time

import numpy as np
import pytest

import manifoldx as mx
from manifoldx import bench
from manifoldx.bench.folder import write_run
from manifoldx.bench.runner import run_scenario
from manifoldx.bench.scenario import Scenario
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import StandardMaterial, cube


class App:
    def __init__(self, engine):
        self.engine, self.dts, self.closed = engine, [], False
        engine.camera.fit(20)
        engine.spawn(Mesh(cube(0.5, 0.5, 0.5)), Material(StandardMaterial("#88aaff")),
                     Transform(pos=np.random.default_rng(0).uniform(-5, 5, (50, 3)).astype(np.float32)), n=50)

        @engine.system
        def sleepy(q: mx.Query[Transform], dt: float):
            self.dts.append(dt)
            time.sleep(0.002)

    def close(self):
        self.closed = True


def make(steps, **kw):
    holder = {}

    def fn(engine, p):
        holder["app"] = App(engine)
        return holder["app"], list(steps)

    return Scenario("tiny", fn, kw.get("params", {}), {"smoke": 4, "full": None}), holder


def test_phases_get_their_own_frames_and_systems():
    sc, _ = make([bench.mark("a"), bench.hold(frames=5), bench.mark("b"), bench.hold(frames=7)])
    run = run_scenario(sc, size=(64, 48))
    assert run.error is None
    a, b = run.summary["phases"]["a"], run.summary["phases"]["b"]
    assert (a["frames"], b["frames"]) == (5, 7) and "main" not in run.summary["phases"]
    sleepy = [ms for name, ms in b["systems_ms"].items() if name.endswith("sleepy")]
    assert sleepy and sleepy[0] >= 1.5
    assert b["p50_ms"] > 0 and b["max_ms"] >= b["p95_ms"] >= b["p50_ms"]
    assert set(b["split_ms"]) >= {"update", "render"}


def test_fixed_step_hands_systems_that_dt():
    sc, holder = make([bench.hold(frames=4)])
    run_scenario(sc, size=(64, 48), step_dt=1 / 30)
    assert holder["app"].dts and all(abs(dt - 1 / 30) < 1e-9 for dt in holder["app"].dts)


def test_pipelined_loop_reports_overlap():
    sc, _ = make([bench.hold(frames=6)])
    run = run_scenario(sc, size=(64, 48), loop="pipelined")
    assert run.error is None and "overlap" in run.summary["phases"]["main"]["split_ms"]
    assert run.summary["loop"] == "pipelined"


def test_host_says_whether_the_adapter_is_software():
    sc, _ = make([bench.hold(frames=2)])
    host = run_scenario(sc, size=(64, 48)).summary["host"]
    assert isinstance(host["software"], bool) and host["adapter"] and host["cpus"] >= 1
    assert host["load_before"] is not None and host["load_max"] >= 0


def test_versions_come_from_metadata_and_label_is_free():
    sc, _ = make([bench.hold(frames=2)])
    run = run_scenario(sc, size=(64, 48), label="abc123", app_name="no-such-app")
    assert run.summary["versions"] == {"manifoldx": importlib.metadata.version("manifold-gfx"), "app": None}
    assert run.summary["label"] == "abc123"


def test_smoke_caps_each_waiting_step():
    sc, _ = make([bench.hold(frames=50), bench.mark("b"), bench.hold(frames=50)])
    run = run_scenario(sc, size=(64, 48), smoke=True)
    assert run.summary["phases"]["main"]["frames"] == 4 and run.summary["phases"]["b"]["frames"] == 4


def test_shot_lands_in_the_folder(tmp_path):
    sc, _ = make([bench.hold(frames=2), bench.shot("x")])
    run = run_scenario(sc, size=(64, 48))
    folder = write_run(run, tmp_path / "runs", None)
    from manifoldx.image import read_png

    assert read_png(folder / "shots" / "x.png").shape[:2] == (48, 64)


def test_a_failing_step_still_writes_the_run(tmp_path):
    @bench.step("explode")
    def explode(app):
        raise RuntimeError("boom")

    sc, holder = make([bench.hold(frames=3), explode()])
    run = run_scenario(sc, size=(64, 48))
    assert "RuntimeError: boom" in run.error and holder["app"].closed
    folder = write_run(run, tmp_path / "runs", None)
    saved = json.loads((folder / "run.json").read_text())
    assert saved["error"] == run.error and saved["phases"]["main"]["frames"] == 3
    assert len(np.load(folder / "frames.npz")["wall_ms"]) == 3


def test_a_hold_that_times_out_fails_the_run():
    sc, _ = make([bench.hold(until=lambda: False, max_s=0.1)])
    assert "did not finish" in run_scenario(sc, size=(64, 48)).error


def test_interrupt_writes_a_partial_run(tmp_path):
    @bench.step("ctrl_c")
    def ctrl_c(app):
        raise KeyboardInterrupt

    sc, _ = make([bench.hold(frames=2), ctrl_c()])
    run = run_scenario(sc, size=(64, 48))
    assert run.error == "interrupted"
    assert json.loads((write_run(run, tmp_path / "runs", None) / "run.json").read_text())["error"] == "interrupted"


def test_history_gets_one_line_per_run(tmp_path):
    history = tmp_path / "bench" / "history.jsonl"
    for _ in range(2):
        sc, _ = make([bench.hold(frames=2)])
        write_run(run_scenario(sc, size=(64, 48)), tmp_path / "runs", history)
    lines = [json.loads(x) for x in history.read_text().splitlines()]
    assert len(lines) == 2 and lines[0].keys() == lines[1].keys()
    assert {"scenario", "params", "loop", "host", "versions", "phases", "error", "app"} <= lines[0].keys()


def test_params_reach_the_scenario():
    seen = {}

    def fn(engine, p):
        seen["tau"] = p.tau
        return App(engine), [bench.hold(frames=1)]

    run = run_scenario(Scenario("p", fn, {"tau": 16}), size=(64, 48), overrides={"tau": "24"})
    assert seen["tau"] == 24 and run.summary["params"] == {"tau": 24}


def test_events_record_marks_and_steps():
    sc, _ = make([bench.mark("a"), bench.hold(frames=1)])
    kinds = [(e["kind"], e.get("name")) for e in run_scenario(sc, size=(64, 48)).events]
    assert ("mark", "a") in kinds and ("step", "hold") in kinds


@pytest.mark.parametrize("loop", ["serial", "check"])
def test_check_loop_runs(loop):
    sc, _ = make([bench.hold(frames=3)])
    assert run_scenario(sc, size=(64, 48), loop=loop).error is None


def test_the_runner_closes_the_engine_so_shutdown_fires():
    fired = []

    def fn(engine, p):
        engine.on("shutdown")(lambda payload: fired.append(1))
        return App(engine), [bench.hold(frames=1)]

    assert run_scenario(Scenario("s", fn), size=(64, 48)).error is None
    assert fired == [1]


def test_an_engine_whose_setup_raised_is_closed_too():
    fired = []

    def fn(engine, p):
        engine.on("shutdown")(lambda payload: fired.append(1))
        raise RuntimeError("setup broke")

    run = run_scenario(Scenario("s", fn), size=(64, 48))
    assert "setup broke" in run.error and fired == [1]


# --- step 2: counters, spans, metrics, alignment ------------------------------------------------


def test_counters_reach_the_phase_summary_and_the_arrays():
    sc, _ = make([bench.mark("a"), bench.hold(frames=5)])
    run = run_scenario(sc, size=(64, 48))
    c = run.summary["phases"]["a"]["counters"]
    assert c["draws"]["mean"] >= 1 and "bytes_uploaded" in c
    names = list(run.frames["counter_names"])
    assert "draws" in names and run.frames["counters"].shape == (len(run.frames["wall_ms"]), len(names))


def test_a_span_in_a_system_and_a_metric_are_recorded():
    def fn(engine, p):
        app = App(engine)

        @engine.system
        def walled(q: mx.Query[Transform], dt: float):
            with bench.span("walls"):
                time.sleep(0.001)
            bench.metric("cars", 12)

        return app, [bench.hold(frames=4)]

    run = run_scenario(Scenario("spans", fn, {}, {"smoke": 4, "full": None}), size=(64, 48))
    spans = run.summary["phases"]["main"]["spans_ms"]
    assert any(k.endswith("walled/walls") and v >= 0.8 for k, v in spans.items())
    assert run.summary["app"] == {"cars": 12}
    assert run.summary["phases"]["main"]["app"]["cars"]["max"] == 12


def test_arrays_stay_aligned_when_a_frame_is_cut(monkeypatch):
    from manifoldx.bench import probes

    calls = {"n": 0}
    real = probes.Columns.add

    def cut(self, row):
        calls["n"] += 1
        if calls["n"] == 11:  # the third frame's first column probe (5 a frame), after frames.add ran
            raise KeyboardInterrupt
        real(self, row)

    monkeypatch.setattr(probes.Columns, "add", cut)
    sc, _ = make([bench.hold(frames=10)])
    run = run_scenario(sc, size=(64, 48))
    assert run.error == "interrupted"
    lengths = {k: len(v) for k, v in run.frames.items() if not k.endswith("_names")}
    assert len(set(lengths.values())) == 1, lengths


def test_the_error_event_is_written_once():
    sc, _ = make([bench.hold(frames=1), bench.each_frame(lambda app: 1 / 0, frames=1)])
    run = run_scenario(sc, size=(64, 48))
    assert [e["kind"] for e in run.events].count("error") == 1


def test_cpu_is_process_time_and_draw_is_the_engine_frame():
    import threading

    stop = threading.Event()

    def burn():  # a thread busy beside the frame: process CPU counts it, the engine's frame does not
        while not stop.is_set():
            sum(range(1000))

    def fn(engine, p):
        app = App(engine)
        threading.Thread(target=burn, daemon=True).start()
        return app, [bench.hold(frames=6)]

    try:
        run = run_scenario(Scenario("burn", fn, {}, {"smoke": 6, "full": None}), size=(64, 48))
    finally:
        stop.set()
    ph = run.summary["phases"]["main"]
    assert ph["draw_p50_ms"] > 0 and ph["cpu_p50_ms"] > ph["draw_p50_ms"]
    assert len(run.frames["draw_ms"]) == len(run.frames["cpu_ms"]) == 6


def test_host_says_whether_the_build_is_free_threaded():
    import sys

    sc, _ = make([bench.hold(frames=1)])
    host = run_scenario(sc, size=(64, 48)).summary["host"]
    assert host["free_threaded"] == (not sys._is_gil_enabled()) and host["python"]


def test_counters_are_off_after_a_run():
    from manifoldx import counters

    sc, _ = make([bench.hold(frames=1)])
    run = run_scenario(sc, size=(64, 48))
    assert not counters.recording() and run.summary["counters_exact"] is True


# --- step 2: streamers and workers ---------------------------------------------------------------


def _walk_scenario(workers="thread", far=None):
    """A Streamer only a system's closure holds; the app is None. With `far`, a second phase elsewhere."""
    from manifoldx.streaming import Streamer
    from test_streaming import build_cell

    def fn(engine, p):
        s = Streamer(engine, cell=8.0, build=build_cell, workers=workers)
        at = {"x": 4.0}

        @engine.system
        def walk(q: mx.Query[Transform], dt: float):
            s.focus(at["x"], 4.0)

        def go(app):
            at["x"] = far

        steps = [bench.mark("a"), bench.hold(until=bench.settled(s), max_s=30)]
        if far is not None:
            steps += [bench.mark("b"), bench.each_frame(go, frames=1), bench.hold(until=bench.settled(s), max_s=30)]
        return None, steps

    return Scenario("walk", fn, {}, {"smoke": 400, "full": None})


def test_a_streamer_in_a_closure_is_found_and_closed():
    from manifoldx import streaming

    before = set(map(id, streaming.live()))
    run = run_scenario(_walk_scenario(), size=(64, 48))
    assert run.error is None
    assert not [s for s in streaming.live() if id(s) not in before]
    assert run.summary["phases"]["a"]["streamer"]["drawn"]


def test_process_workers_report_their_jobs():
    import glob
    import tempfile

    run = run_scenario(_walk_scenario("process"), size=(64, 48))
    w = run.summary["phases"]["a"]["workers"]["stream"]
    assert run.error is None and w["jobs"] >= 1 and w["processes"] == 1 and w["max_ms"] >= w["p50_ms"]
    assert any(e["kind"] == "job" for e in run.events)
    assert not glob.glob(f"{tempfile.gettempdir()}/manifoldx-bench-*")


def test_settle_time_is_per_phase():
    run = run_scenario(_walk_scenario(far=4.0 + 8 * 40), size=(64, 48))
    a, b = run.summary["phases"]["a"]["streamer"], run.summary["phases"]["b"]["streamer"]
    assert a["settled_s"] is not None and b["settled_s"] is not None
    assert a["first_drawn_s"] <= a["settled_s"] and b["first_drawn_s"] <= b["settled_s"]


# --- step 2: memory and origin -------------------------------------------------------------------


def test_memory_is_recorded():
    sc, _ = make([bench.hold(frames=3)])
    mem = run_scenario(sc, size=(64, 48)).summary["memory"]
    assert mem["rss_mb"]["max"] >= mem["rss_mb"]["start"] > 0
    assert len(mem["gc"]["runs"]) == 3
    assert mem["gpu_mb"] is None or mem["gpu_mb"]["max"] >= 0


def test_gc_inside_a_frame_is_counted():
    import gc

    def fn(engine, p):
        app = App(engine)

        @engine.system
        def collect(q: mx.Query[Transform], dt: float):
            gc.collect()

        return app, [bench.hold(frames=2)]

    run = run_scenario(Scenario("gc", fn, {}, {"smoke": 2, "full": None}), size=(64, 48))
    assert sum(run.summary["memory"]["gc"]["runs"]) >= 2 and run.summary["phases"]["main"]["gc_ms"] > 0


def test_folders_named_by_the_app_are_sized(tmp_path):
    def write(app):
        (tmp_path / "blob.bin").write_bytes(b"\0" * (1 << 20))

    sc, _ = make([bench.each_frame(write, frames=1)])
    mem = run_scenario(sc, size=(64, 48), folders={"cache": tmp_path}).summary["memory"]
    grew = mem["folders_mb"]["cache"]["end"] - mem["folders_mb"]["cache"]["start"]
    assert abs(grew - 1.0) < 0.01


def test_origin_is_the_farthest_thing_drawn():
    def fn(engine, p):
        app = App(engine)  # cubes within 5 m of the origin
        engine.spawn(Mesh(cube(1, 1, 1)), Material(StandardMaterial("#ffffff")),
                     Transform(pos=np.array([[3000.0, 0.0, 4000.0]], np.float32)), n=1)
        return app, [bench.hold(frames=31)]

    run = run_scenario(Scenario("far", fn, {}, {"smoke": 31, "full": None}), size=(64, 48))
    assert abs(run.summary["phases"]["main"]["origin_max_m"] - 5000.0) < 1.0
    assert len(run.frames["origin_m"]) == 31


def test_settle_time_is_the_last_settle_of_the_phase():
    from manifoldx.streaming import Streamer
    from test_streaming import build_cell

    def fn(engine, p):
        s = Streamer(engine, cell=8.0, build=build_cell)
        at = {"x": 4.0}

        @engine.system
        def walk(q: mx.Query[Transform], dt: float):
            s.focus(at["x"], 4.0)

        def go(app):
            at["x"] = 4.0 + 8 * 40

        return None, [bench.hold(until=bench.settled(s), max_s=30), bench.mark("b"), bench.hold(frames=3),
                      bench.each_frame(go, frames=1), bench.hold(until=bench.settled(s), max_s=30)]

    run = run_scenario(Scenario("resettle", fn, {}, {"smoke": 400, "full": None}), size=(64, 48))
    b = run.summary["phases"]["b"]
    moved = next(e["t"] for e in run.events if e["kind"] == "step" and e["index"] == 3)
    start = next(e["t"] for e in run.events if e["kind"] == "mark" and e["name"] == "b")
    assert b["streamer"]["settled_s"] > moved - start  # after the move, not the settled frames before it


def test_a_cleanup_that_raises_still_stops_recording(monkeypatch):
    import os

    from manifoldx import counters

    def broken():
        raise ValueError("a job file went bad")

    monkeypatch.setattr(counters, "jobs", broken)
    sc, _ = make([bench.hold(frames=2)])
    run = run_scenario(sc, size=(64, 48))
    assert run.summary["phases"]["main"]["frames"] == 2
    assert not counters.recording() and counters.JOBS_ENV not in os.environ


# --- step 3: timing the comparison relies on -----------------------------------------------------


def test_bookkeeping_stays_out_of_the_next_frame(monkeypatch):
    from manifoldx.bench import runner

    def slow(engine):
        time.sleep(0.08)
        return 0.0

    monkeypatch.setattr(runner, "origin_distance", slow)  # sampled at frame 0, then every 30
    sc, _ = make([bench.hold(frames=4)])
    run = run_scenario(sc, size=(64, 48))
    assert run.frames["bench_ms"][0] >= 80
    assert run.frames["wall_ms"][1] < 60


def test_a_streamer_made_in_a_step_is_tracked():
    from manifoldx.streaming import Streamer
    from test_streaming import build_cell

    def fn(engine, p):
        made = {}

        def walk(app):  # the streamer is made by the step, after the scenario returned
            if "s" not in made:
                made["s"] = Streamer(engine, cell=8.0, build=build_cell, workers="thread")
            made["s"].focus(4.0, 4.0)

        return None, [bench.mark("a"), bench.each_frame(walk, until=lambda: "s" in made and made["s"].settled)]

    run = run_scenario(Scenario("late", fn, {}, {"smoke": 400, "full": None}), size=(64, 48))
    assert run.error is None
    assert run.summary["phases"]["a"]["streamer"]["settled_s"] is not None


def test_each_entry_of_a_phase_is_timed_from_its_own_start():
    from manifoldx.bench.probes import stream_summary

    nodes = [{"t": 5.2, "ev": "drawn", "level": 0}]
    ends, settled = [0.5, 1.0, 5.5, 6.0], [False, True, False, True]
    s = stream_summary(nodes, [(0.0, 1.0), (5.0, 6.5)], ends, settled, np.ones(4, bool))
    assert s["first_drawn_s"] == 0.2
    assert s["settled_s"] == 1.0  # the last entry's settle, from 5.0
    assert [e["start_s"] for e in s["entries"]] == [0.0, 5.0]
