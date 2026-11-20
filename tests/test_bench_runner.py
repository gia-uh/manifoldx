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
    sc, _ = make([bench.hold(frames=4)])
    run = run_scenario(sc, size=(64, 48))
    ph = run.summary["phases"]["main"]
    assert ph["draw_p50_ms"] > 0 and ph["cpu_p50_ms"] >= 0
    assert len(run.frames["draw_ms"]) == len(run.frames["cpu_ms"]) == 4


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
