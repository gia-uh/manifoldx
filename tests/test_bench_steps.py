"""Scenarios, app steps and the engine's steps, driven frame by frame on a headless engine."""

import json
import types

import numpy as np
import pytest

from manifoldx import bench
from manifoldx.bench import micro
from manifoldx.bench.scenario import STEPS, Ctx, Params, StepCall, StepTimeout, iterate, load_json, step
from manifoldx.bench.steps import camera, each_frame, emit, hold, mark, shot, zoom
from manifoldx.bench.steps import set as set_state


class FakeRun:
    def __init__(self):
        self.marks, self.shots = [], []

    def mark(self, name):
        self.marks.append(name)

    def shot(self, name):
        self.shots.append(name)


@pytest.fixture
def engine():
    e = micro.headless_engine(16)
    micro.start(e)
    return e


def make_ctx(engine, app=None, index=0):
    ctx = Ctx(engine, app if app is not None else types.SimpleNamespace(count=0), FakeRun())
    ctx.index = index
    return ctx


def drain(gen, engine):
    """Render one frame per yield; return how many."""
    n = 0
    for _ in gen:
        engine._draw_frame()
        engine._render_canvas.draw()
        n += 1
    return n


def test_hold_frames_renders_that_many(engine):
    assert drain(iterate(hold(frames=5), make_ctx(engine)), engine) == 5


def test_hold_times_out_with_the_step_named(engine):
    with pytest.raises(StepTimeout, match=r"hold.*#3.*0\.2 s"):
        drain(iterate(hold(until=lambda: False, max_s=0.2), make_ctx(engine, index=3)), engine)


def test_hold_until_already_true_takes_no_frame(engine):
    assert drain(iterate(hold(until=lambda: True), make_ctx(engine)), engine) == 0


def test_camera_far_away_moves_the_origin(engine):
    drain(iterate(camera((900_000.0, 50.0, 0.0), look_at=(900_100.0, 0.0, 0.0)), make_ctx(engine)), engine)
    ox, oz = engine.origin
    assert abs(900_000.0 - ox) < 2000
    assert np.hypot(engine.camera.position[0], engine.camera.position[2]) < 2000


def test_camera_near_the_origin_leaves_it(engine):
    drain(iterate(camera((10.0, 5.0, 10.0), look_at=(0.0, 0.0, 0.0)), make_ctx(engine)), engine)
    assert engine.origin == (0.0, 0.0)
    assert np.allclose(engine.camera.position, [10.0, 5.0, 10.0])


def test_zoom_in_steps_holds_each_height_until_the_condition(engine):
    calls = {"n": 0}

    def ready():
        calls["n"] += 1
        return calls["n"] % 2 == 0  # true every other call: one frame of waiting per height

    heights = []
    gen = iterate(zoom(1000.0, 10.0, steps=4, hold=ready), make_ctx(engine))
    for _ in gen:
        heights.append(round(float(engine.camera.position[1]), 3))
        engine._draw_frame()
    assert sorted(set(heights), reverse=True) == [1000.0, 215.443, 46.416, 10.0]


def test_zoom_needs_seconds_or_steps(engine):
    with pytest.raises(ValueError, match="seconds or steps"):
        list(iterate(zoom(100.0, 10.0), make_ctx(engine)))


def test_set_merges_a_patch_into_state(engine):
    engine.state["sky"] = {"weather": "clear", "stars": 60}
    drain(iterate(set_state({"sky": {"weather": "storm"}}), make_ctx(engine)), engine)
    assert engine.state["sky"] == {"weather": "storm", "stars": 60}


def test_emit_reaches_a_handler(engine):
    got = []

    @engine.on("ping")
    def ping(payload):
        got.append(payload)

    drain(iterate(emit("ping", {"n": 1}), make_ctx(engine)), engine)
    engine._draw_frame()  # events are delivered at the start of the next frame
    assert got == [{"n": 1}]


def test_mark_and_shot_go_to_the_run(engine):
    ctx = make_ctx(engine)
    drain(iterate(mark("zoom"), ctx), engine)
    drain(iterate(shot("street"), ctx), engine)
    assert ctx.run.marks == ["zoom"] and ctx.run.shots == ["street"]


def test_each_frame_calls_the_function_with_the_app(engine):
    app = types.SimpleNamespace(count=0)

    def tick(a):
        a.count += 1

    assert drain(iterate(each_frame(tick, frames=4), make_ctx(engine, app)), engine) == 4
    assert app.count == 4


def test_an_app_step_gets_the_app_and_can_span_frames(engine):
    @step("wiggle")
    def wiggle(app, n):
        for _ in range(n):
            app.count += 1
            yield

    app = types.SimpleNamespace(count=0)
    assert drain(iterate(StepCall("wiggle", (3,), {}), make_ctx(engine, app)), engine) == 3
    assert app.count == 3


def test_a_plain_app_step_takes_no_frame(engine):
    @step("poke")
    def poke(app):
        app.count += 10

    app = types.SimpleNamespace(count=0)
    assert drain(iterate(StepCall("poke", (), {}), make_ctx(engine, app)), engine) == 0
    assert app.count == 10


def test_an_unknown_step_is_named(engine):
    with pytest.raises(ValueError, match="no step named 'teleport'"):
        list(iterate(StepCall("teleport", (), {}), make_ctx(engine)))


def test_json_scenario_calls_steps_by_name(tmp_path, monkeypatch):
    mod = tmp_path / "jsonapp.py"
    mod.write_text("def make(engine, p):\n    return {'tau': p.tau}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    path = tmp_path / "s.json"
    path.write_text(json.dumps({
        "name": "from_json", "app": "jsonapp:make", "params": {"tau": 16},
        "steps": [{"mark": "a"}, {"hold": {"frames": 2}}, {"emit": ["ping", {"n": 1}]}],
    }))
    sc = load_json(path)
    app, steps = sc.fn(None, Params(sc.params, {"tau": "24"}))
    assert sc.name == "from_json" and app == {"tau": 24}
    assert steps == [StepCall("mark", ("a",), {}), StepCall("hold", (), {"frames": 2}),
                     StepCall("emit", ("ping", {"n": 1}), {})]


def test_params_parse_overrides_to_the_default_type():
    p = Params({"tau": 16, "traffic": True, "ratio": 0.5, "name": "x"},
               {"tau": "24", "traffic": "false", "ratio": "2", "name": "y"})
    assert (p.tau, p.traffic, p.ratio, p.name) == (24, False, 2.0, "y")


def test_an_unknown_param_is_refused():
    with pytest.raises(ValueError, match="no param 'tua'"):
        Params({"tau": 16}, {"tua": "1"})


def test_a_comma_list_param_says_sweeps_come_later():
    with pytest.raises(ValueError, match="sweep"):
        Params({"tau": 16}, {"tau": "12,16"})


def test_no_engine_step_knows_about_vehicles_or_maps():
    assert not {"drive", "teleport", "latlon", "place"} & STEPS.keys()


# --- step 2: game time and checked paths ---------------------------------------------------------


def drain_t(gen, engine, ctx):
    """drain, advancing ctx.t by ctx.dt after each frame as the runner does."""
    n = 0
    for _ in gen:
        engine._draw_frame()
        engine._render_canvas.draw()
        ctx.t += ctx.dt
        n += 1
    return n


def test_hold_seconds_counts_game_time_under_a_fixed_step(engine):
    ctx = make_ctx(engine)
    ctx.dt = 1 / 30
    assert ctx.t == 0.0
    assert drain_t(iterate(bench.hold(seconds=1.0), ctx), engine, ctx) == 30


def test_zoom_seconds_counts_game_time(engine):
    ctx = make_ctx(engine)
    ctx.dt = 1 / 10
    assert drain_t(iterate(bench.zoom(100.0, 10.0, seconds=1.0), ctx), engine, ctx) == 10


@pytest.mark.parametrize("call", [
    lambda: bench.fly((0, 10, 0), (100, 10, 0), speed=0),
    lambda: bench.fly((0, 10, 0), (0, 10, 0), speed=5),
    lambda: bench.zoom(0.0, 50.0, steps=3),
    lambda: bench.zoom(100.0, -1.0, steps=3),
])
def test_bad_paths_are_refused(call):
    with pytest.raises(ValueError):
        call()
