# Measuring a frame

A city that streams in as the camera falls toward it feels slow, and nobody
can say where the time goes. The frame counter says forty milliseconds. Is that
the systems, the render, the wait for the GPU, or the streamer building cells
in the background? Did last week's change make it better or worse? A script
that times a few frames answers one of those questions once and is thrown
away. A scenario answers them every time it runs.

`manifoldx bench` runs scenarios: an app built the way it always is, plus a
list of steps that move the camera, wait for something and take pictures. Each
run records every frame, splits it by phase, and writes a folder that a person
reads as a table and an agent reads as JSON.

## A scenario

The quad streaming example draws a plain a hundred kilometres wide, streamed in
quadtree nodes. Its `build(engine, fly=False)` sets up the plain and returns the
streamer without the system that flies the camera, so a scenario can fly it
instead. manifoldx's own scenarios live in `benchmarks/scenarios.py`:

```python
from manifoldx import bench

AT = (51_200.0, 51_200.0)


@bench.scenario("quad_zoom", sizes={"smoke": 20, "full": None})
def quad_zoom(engine, p):
    """The quad streaming plain from 20 km up down to 50 m, each height held until it has settled."""
    from examples.quad_streaming import build

    streamer = build(engine, fly=False)

    @engine.system
    def follow(query, dt):
        ox, oz = engine.origin
        x, y, z = (float(v) for v in engine.camera.position)
        streamer.focus(x + ox, z + oz, y)

    ready = bench.settled(streamer)
    return streamer, [
        bench.camera((AT[0], 20_000.0, AT[1] - 6000.0), (AT[0], 0.0, AT[1])),
        bench.hold(until=ready, max_s=60),
        bench.mark("zoom"),
        bench.zoom(20_000.0, 50.0, at=AT, steps=6, hold=ready, max_s=60),
        bench.shot("street"),
    ]
```

The function gets the engine and its `Params`, builds the app and returns it
with its steps. Each step is a `StepCall`, a name and its arguments; the runner
looks the name up when the step runs. `bench.settled(streamer)` is true once
the streamer has drawn every node the camera needs, which is
`QuadStreamer.settled` (or `Streamer.settled`). `hold` keeps rendering until it
is, and gives up with a `StepTimeout` after `max_s` seconds, naming the step.
`mark` starts a phase: everything before it is `main`, everything after it is
`zoom`, and every number is reported per phase.

The project's `pyproject.toml` says where the scenarios are:

```toml
[tool.manifoldx.bench]
scenarios = "benchmarks.scenarios"
```

## Running it

```bash
uv run manifoldx bench list
uv run manifoldx bench run quad_zoom --size 960x540
```

```text
quad_zoom  loop=serial  size=960x540  Quadro M2000M  load 5.01–5.09
phase          frames   p50 ms   p95 ms   max ms  cpu p50  slowest systems
main                4    334.9    885.1    932.8    332.9  follow 26.915
zoom               69     40.0     77.0    109.1     38.6  follow 26.163
run folder: bench/runs/2026-11-20T14-18-56-quad_zoom-89fc
```

The four frames of `main` are the first view of the plain: each one waited for
nodes 25 km wide to upload. The zoom itself runs at 40 ms a frame, and the
`follow` system, which is the streamer choosing its nodes, takes 26 of them.
The load average goes in the first line because every number on a shared
machine depends on it.

The folder holds `run.json` (the summary), `frames.npz` (every frame's wall
time, CPU time, phase, split and systems), `events.jsonl` (each step and mark
with its time) and `shots/street.png`. The summary is also appended as one line
to `bench/history.jsonl`. `manifoldx bench show <folder>` prints it again;
`--json` on `run` or `show` prints the summary alone, for an agent. Progress
goes to stderr, one line a second.

## The engine's steps and an app's own

The engine's steps know nothing about what an app draws: `camera`, `fly` and
`zoom` move the camera in world coordinates (moving the engine's floating
origin when the camera goes more than 2 km from it), `hold` and `each_frame`
take frames, `set` and `emit` are the agent loop's verbs, and `mark` and `shot`
record. An engine step runs with a `Ctx`, which holds the engine, the app and
the run.

Anything else is a step the app registers. A driving game adds `drive`; a
viewer adds `open_model`:

```python
@bench.step("spin_up")
def spin_up(app, rpm):
    for _ in range(60):
        app.rotor.speed += rpm / 60
        yield                      # one frame per yield
```

A step that is a generator takes one frame per `yield`; a plain function takes
none. A scenario calls it like any other, `spin_up(300)`, and a JSON scenario
calls it by name:

```json
{"name": "spin", "app": "rotor_app:make", "params": {"rpm": 300},
 "steps": [{"spin_up": [300]}, {"hold": {"frames": 30}}, {"shot": "rotor"}]}
```

`manifoldx bench run spin.json` runs it. `manifoldx bench list` shows both
kinds of step.

## Loops, steps and sizes

`--loop pipelined` runs the next frame's systems while this one renders, on the
free-threaded build; the `overlap` column of the split says how much of the
frame they shared. `--loop check` adds `MANIFOLDX_PIPELINE=check`, which raises
when a system writes something the render reads. `--step 1/30` gives systems a
fixed dt instead of the wall clock, so two runs see the same simulation.
`--param n=200` overrides a param, parsed to the default's type. `--smoke`
caps every waiting step at the scenario's smoke size, which is what the tests
run.

```bash
uv run manifoldx bench run meshes_spin --loop pipelined --param n=5000
```

A run ends with its numbers even when it fails: a step that raises, a hold that
times out or Ctrl-C still write the folder, with the error in `run.json`, and
`run` exits 1. The app's `close()` and its streamers' worker pools are closed
on the way out. The `Run` that `run_scenario` returns holds the same summary,
the per-frame arrays, the events and the shots, for code that drives scenarios
itself.

## Micro cases

A frame is the sum of many smaller costs. A micro case times one of them at
several sizes: a function of `n` registered with `@bench.case`, returning
milliseconds. Each case is a `Case`, and each timing a `Result`.

```python
from manifoldx import bench
from manifoldx.bench.micro import measure


@bench.case("modeling", "smooth", sizes=[5, 6, 7], smoke=1)
def smooth(level):
    """Smooth a subdivided sphere."""
    m = sphere(level)
    return measure(lambda: m.smooth())
```

```bash
uv run manifoldx bench micro benchmarks.suites.modeling --case smooth
uv run manifoldx bench micro --against ../manifoldx-main --case sprites
```

With no modules, `micro` runs the ones listed under `micro` in the project's
`Config`. `--against DIR` runs the same cases on the manifoldx in another
directory, such as a copy of the main branch made with your version control,
and prints a Markdown table of both. manifoldx itself never asks a version
control system for anything.
