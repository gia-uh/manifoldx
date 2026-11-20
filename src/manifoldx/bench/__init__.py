"""Benchmark, profile and check any manifoldx app: scenarios, probes, run folders and micro cases.

`manifoldx bench` is the command; this package is what an app's scenarios and micro cases import:

    from manifoldx import bench

    @bench.scenario("zoom", params={"tau": 16})
    def zoom(engine, p):
        app = make_app(engine, tau=p.tau)
        return app, [bench.camera((0, 2000, 0), (0, 0, 0)), bench.hold(until=bench.settled(app.streamer))]
"""

from manifoldx.bench import micro
from manifoldx.counters import metric, span, worker_span
from manifoldx.bench.micro import case
from manifoldx.bench.runner import Run, run_scenario
from manifoldx.bench.scenario import Ctx, Params, Scenario, StepCall, StepTimeout, load_json, scenario, step
from manifoldx.bench.steps import camera, drawn, each_frame, emit, fly, hold, mark, set, settled, shot, zoom

__all__ = [
    "Ctx", "Params", "Run", "Scenario", "StepCall", "StepTimeout", "camera", "case", "drawn", "each_frame", "emit", "fly",
    "hold", "load_json", "mark", "metric", "micro", "run_scenario", "scenario", "set", "settled", "shot", "span", "step",
    "worker_span", "zoom",
]
