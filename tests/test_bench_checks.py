"""Checks: budgets and functions over a run, evaluated after it."""

from manifoldx import bench
from manifoldx.bench import checks
from manifoldx.bench.runner import run_scenario
from manifoldx.bench.scenario import SCENARIOS
from test_bench_runner import make


def test_lookup_reads_a_key_or_a_probe_and_key():
    ph = {"p95_ms": 12.0, "origin_max_m": 30.0, "counters": {"draws": {"mean": 4.0, "max": 5}},
          "streamer": {"settled_s": 1.5}, "systems_ms": {"m.spin": 2.0}, "app": {"cars": {"last": 3, "mean": 2.5}}}
    assert checks.lookup(ph, "p95_ms") == 12.0 and checks.lookup(ph, "frames_p95_ms") == 12.0
    assert checks.lookup(ph, "origin_max_m") == 30.0 and checks.lookup(ph, "counters_draws") == 4.0
    assert checks.lookup(ph, "streamer_settled_s") == 1.5 and checks.lookup(ph, "systems_m.spin") == 2.0
    assert checks.lookup(ph, "app_cars") == 2.5 and checks.lookup(ph, "nothing_here") is None


def test_a_budget_names_its_phase_and_frame():
    sc, _ = make([bench.mark("a"), bench.hold(frames=4)])
    run = run_scenario(sc, size=(64, 48), budgets=[checks.Check("fast", {"p95_ms": 0.001})])
    (c,) = run.summary["checks"]
    assert not c["ok"] and c["phase"] == "a" and c["budget"] == 0.001 and c["value"] > 0.001
    assert isinstance(c["frame"], int) and "p95_ms" in c["message"] and "a" in c["message"]


def test_a_budget_that_holds_passes_in_every_phase():
    sc, _ = make([bench.hold(frames=2), bench.mark("b"), bench.hold(frames=2)])
    run = run_scenario(sc, size=(64, 48), budgets=[checks.Check("draws", {"counters_draws": 1000})])
    assert [(c["phase"], c["ok"]) for c in run.summary["checks"]] == [("main", True), ("b", True)]


def test_a_function_check_sees_the_run():
    seen = {}

    def fn(run):
        seen["frames"] = len(run.frames["wall_ms"])
        return run.phases["main"]["frames"] == 3

    sc, _ = make([bench.hold(frames=3)])
    run = run_scenario(sc, size=(64, 48), budgets=[checks.Check("three", {}, fn=fn)])
    assert run.summary["checks"][0]["ok"] and seen["frames"] == 3


def test_a_gpu_budget_skips_on_software(monkeypatch):
    sc, _ = make([bench.hold(frames=2)])
    run = run_scenario(sc, size=(64, 48), budgets=[checks.Check("gpu", {"p95_ms": 0.001}, on="gpu")])
    c = run.summary["checks"][0]
    if run.summary["host"]["software"]:
        assert c["skipped"] and c["ok"] and "software" in c["message"]
    else:
        assert not c["skipped"] and not c["ok"]


def test_checks_run_after_the_engine_is_closed():
    closed = {}
    sc, holder = make([bench.hold(frames=2)])

    def fn(run):
        closed["app"] = holder["app"].closed
        return True

    run_scenario(sc, size=(64, 48), budgets=[checks.Check("after", {}, fn=fn)])
    assert closed["app"] is True


def test_the_decorator_gives_the_scenario_a_check():
    @bench.scenario("checked_tmp", sizes={"smoke": 1})
    def checked_tmp(engine, p):
        return None, []

    try:
        checked_tmp.check(origin_max_m=20_000, phase="far")
        (c,) = SCENARIOS["checked_tmp"].checks
        assert c.budgets == {"origin_max_m": 20_000} and c.phase == "far"
    finally:
        del SCENARIOS["checked_tmp"]


def test_an_unknown_budget_fails_with_the_names_it_has():
    sc, _ = make([bench.hold(frames=2)])
    run = run_scenario(sc, size=(64, 48), budgets=[checks.Check("typo", {"p59_ms": 10})])
    c = run.summary["checks"][0]
    assert not c["ok"] and "p59_ms" in c["message"] and "p95_ms" in c["message"]


def test_a_json_scenario_declares_checks(tmp_path):
    import json

    from manifoldx.bench.scenario import load_json

    path = tmp_path / "s.json"
    path.write_text(json.dumps({"name": "j", "app": "jsonapp:make", "steps": [],
                                "checks": [{"budgets": {"frames_p95_ms": 40}, "phase": "far", "on": "gpu"}]}))
    (c,) = load_json(path).checks
    assert (c.name, c.budgets, c.phase, c.on) == ("frames_p95_ms", {"frames_p95_ms": 40}, "far", "gpu")
