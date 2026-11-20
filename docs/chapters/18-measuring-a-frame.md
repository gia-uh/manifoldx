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

An app can also name `folders` under its project, such as a cache its workers
fill; each run sizes them at its start and its end.

## Running it

```bash
uv run manifoldx bench list
uv run manifoldx bench run quad_zoom --size 960x540
```

```text
quad_zoom  loop=serial  size=960x540  Quadro M2000M  load 5.95–5.95
phase          frames   p50 ms   p95 ms   max ms draw p50  cpu p50  slowest systems
main                4    359.8    851.5    886.6    357.3    358.9  follow 29.784
              per frame: draws 69.5, bytes_uploaded 851768, material_reads 0.5, settled 2.65 s
zoom               71     46.3     92.4    117.6     44.7     80.1  follow 28.706
              per frame: draws 998.62, bytes_uploaded 437408, material_reads 1, settled 3.10 s
run folder: bench/runs/2026-11-20T15-36-42-quad_zoom-0b65
```

The four frames of `main` are the first view of the plain: each one waited for
nodes 25 km wide to upload, 850 KB a frame. The zoom runs at 46 ms a frame; the
`follow` system, which is the streamer choosing its nodes, takes 29 of them,
and a frame draws about a thousand nodes. `draw` is the engine's frame on the
main thread; `cpu` is the CPU time of the whole process over the frame, every
thread included, so a `cpu` above `draw` is work done beside the frame, here the
streamer's builds. The streamer settled 3.1 s after the zoom started. The load
average goes in the first line because every number on a shared machine
depends on it.

The folder holds `run.json` (the summary), `frames.npz` (every frame's numbers,
one array per probe), `events.jsonl` (steps, marks, streamer nodes and worker
jobs, with their times) and `shots/street.png`. The summary is also appended as
one line to `bench/history.jsonl`. `manifoldx bench show <folder>` prints it
again; `--json` on `run` or `show` prints the summary alone, for an agent.
Progress goes to stderr, one line a second, with the step it is on
(`step 4/5`).

## What a run records

Every run records the same probes; nothing has to be turned on.

- **frames**: each frame's wall time, the engine's frame (`draw`) and the
  process's CPU time, with p50, p95, p99, max, and how many frames took more
  than 1.5 and 3 times the median. What the runner itself does between frames
  (sampling memory and the origin, copying a shot) is timed apart as
  `bench_ms` in `frames.npz` and left out of the next frame's wall time.
- **split** and **systems**: the frame's parts (update, acquire, render,
  overlap) and each system's milliseconds.
- **counters**: what the engine did in the frame, counted where it does it:
  `draws`, `bundles_recorded`, `material_reads`, `material_uploads`, `matrices`
  recomputed, `bytes_uploaded`, entities `spawned` and `destroyed`. The engine
  reports them through `manifoldx.counters`, whose calls do nothing outside a
  run. A run on the pipelined loop marks them `counters_exact: false`: a render
  that overlaps the next update counts with whichever frame ends first.
- **streamer**: each node a `Streamer` or `QuadStreamer` submitted, built,
  landed on the GPU, drew, undrew, cancelled or dropped, by level, with the
  time to the first node drawn and the time until the streamers settled. A
  streamer made by a step counts as well as one made by the scenario. A phase
  entered twice (the same `mark` again) times each stretch from its own start
  and lists them under `entries`.
- **workers**: each build's milliseconds, measured inside the worker, by kind,
  and how many processes ran them.
- **memory**: resident memory and the GPU memory allocated through wgpu, once
  a second; garbage collections and their milliseconds; the app's `folders`.
- **origin**: the farthest anything is drawn from the floating origin, every
  30 frames. On `quad_zoom` it is 52 km: the coarse nodes keep their far
  corners.
- **host**: CPUs, load, the adapter and whether it is a software rasterizer,
  the Python version and whether the build is free-threaded.

`show` picks one probe, of one phase or of each:

```bash
uv run manifoldx bench show <folder> --probe counters --phase zoom
uv run manifoldx bench show <folder> --probe streamer
uv run manifoldx bench show <folder> --probe memory
```

## What an app adds

The engine cannot know which part of a system is slow, or how many cars the
game has. Three calls let the app say so, and do nothing outside a run:

```python
from manifoldx import bench


@engine.system
def drive(query, dt):
    with bench.span("walls"):          # reported as drive/walls, under spans_ms
        collide_with_walls()
    bench.metric("cars", len(cars))    # the summary's app dict: last, mean, max


def build_block(key):                  # runs in the app's own worker pool
    with bench.worker_span("block", key):
        return polygonize(key)
```

`worker_span` works in threads and in processes: a run names a folder in the
environment variable `MANIFOLDX_BENCH_JOBS`, each process writes its jobs there,
and the run reads them and removes the folder at its end. manifoldx's own
streamers wrap their builds the same way. `meshes_spin` opens a `busy` span and
publishes a `cubes` metric; `stream_walk` walks the terrain of
`examples/stream_walk.py`, whose cells are built in a worker process.

The counts of one frame are a `Frame`: its counts, its spans and its metrics.
Code that drives the engine itself can call `counters.start()`,
`counters.take()` once a frame and `counters.stop()`.

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
fixed dt instead of the wall clock, so two runs see the same simulation;
`hold(seconds=…)`, `zoom(seconds=…)` and `fly` count that game time, so they
take the same frames on any machine, while every `max_s` stays real seconds.
`--param n=200` overrides a param, parsed to the default's type. `--smoke`
caps every waiting step at the scenario's smoke size, which is what the tests
run.

```bash
uv run manifoldx bench run meshes_spin --loop pipelined --param n=5000
```

A run ends with its numbers even when it fails: a step that raises, a hold that
times out or Ctrl-C still write the folder, with the error in `run.json`, and
`run` exits 1. The app's `close()` and its streamers' worker pools are closed
on the way out, wherever the app kept them. The `Run` that `run_scenario` returns holds the same summary,
the per-frame arrays, the events and the shots, for code that drives scenarios
itself.

## Comparing runs

One run says how long a frame takes; the question is usually whether it got
faster. `--param` with several values sweeps it, and every combination of the
swept params is one side:

```bash
uv run manifoldx bench run meshes_spin --param system_ms=1,6 --repeat 3
```

A run with more than one side is a set. Each side runs `--repeat` times, every
run in a fresh process, and the order alternates each round (A B, B A, A B), so
a load that comes and goes on the machine lands on every side. The set folder
holds `set.json` (the sides as `Side` records, and the runs in the order they
ran) and each side's run folders. At the end, the run prints the comparison.

`bench compare` makes the same comparison from any runs: two run folders, a
side's folder, or a whole set. The first side is the baseline.

```bash
uv run manifoldx bench compare bench/runs/2026-11-20T16-49-55-meshes_spin-set-1c01
uv run manifoldx bench compare bench/runs/A bench/runs/B --out /tmp/a-vs-b
```

It writes `compare.md`, a Markdown table with one row per phase and number:
each side's value (the median over its runs) with its spread in brackets (the
lowest and highest run, or, with one run, the 25th to 75th percentile of the
phase's frame times), and each side's ratio to the baseline. A timing row gets
the note "needs a quiet re-run" when a ratio is within 1.3x either way, and
"taken under load" when a run saw a load average above half the CPUs. Counts,
memory and the app's metrics get no note: they do not depend on the load. Each
shot that every side took is laid side by side in `sheets/<name>.png`, in the
order of the sides.

The same scenario runs on another copy of the app with `--against DIR`, or on
another manifoldx with `--manifoldx SPEC`:

```bash
uv run manifoldx bench run quad_zoom --against ../myapp-main --repeat 3
uv run manifoldx bench run quad_zoom --manifoldx ../manifoldx           # a directory: installed editable
uv run manifoldx bench run quad_zoom --manifoldx manifold-gfx==0.9.0    # anything uv run --with takes
```

With `--against` the other copy is the baseline; with `--manifoldx` this copy
on its own manifoldx is, and the ratios are of the other manifoldx to it.
manifoldx never makes the other copy: make it with your version control, or
unpack a release. It must be a project uv can run, with
manifoldx as a dependency, and its manifoldx must have `bench run --json`. Both
flags run the other side with `uv run`; without uv they refuse and say so, and
everything else on this page works without it. The two do not combine in one
run.

Each (project, spec) pair gets a project environment under one cache folder
(`~/.cache/manifoldx/bench-envs`, or `MANIFOLDX_BENCH_CACHE`), reused by the
next run, so nothing is written into the other copy. A set says how much the
cache grew, counting only what removing it would free: uv links most files from
its own cache, so an environment of a few hundred MB holds a few MB of its own. Past 2 GB (`MANIFOLDX_BENCH_CACHE_GB`) the environments used least
recently go, and `manifoldx bench clean` removes them all. The layer that
`--with` adds lives in uv's own cache, which `uv cache prune` cleans.

## Micro cases

A frame is the sum of many smaller costs. A micro case times one of them at
several sizes: a function of `n` registered with `@bench.case`, returning
milliseconds. Each case is a `Case`, and each timing a `Result`.

```python
from manifoldx import bench
from manifoldx.bench.micro import measure
from manifoldx.modeling import Mesh


@bench.case("modeling", "smooth", sizes=[5, 6, 7], smoke=1)
def smooth(level):
    """Smooth a subdivided sphere."""
    m = Mesh.icosphere(subdivisions=level)
    return measure(lambda: m.smooth(iterations=1))
```

```bash
uv run manifoldx bench micro benchmarks.suites.modeling --case smooth
uv run manifoldx bench micro --against ../manifoldx-main --case sprites
```

With no modules, `micro` runs the ones listed under `micro` in the project's
`Config`. `--against DIR` runs the same cases on the manifoldx in another
directory, such as a copy of the main branch made with your version control,
and prints a Markdown table of both. The two copies take turns over
`--repeat` rounds (3 by default), each in a fresh process; the table gives each
side's median with its spread over the rounds, and the same notes as `bench
compare`. `--profile` profiles the last round of each side only, so the timed
rounds run without the profiler. `micro` exits 1 when a case fails on either
side. manifoldx itself never asks a version control system for anything.
