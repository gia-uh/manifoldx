---
when: making something faster, checking whether a change made something slower, or proving a performance change in a pull request
triggers: ["performance", "benchmark", "slow", "faster", "speed-up", "profile", "per frame", "scaling", "large scene", "perf"]
---

# Benchmarking and proving a performance change

## When to use this

Reach for this before and after any change meant to make manifoldx faster, and when a review asks whether a change costs time. The tools are in `benchmarks/` (not shipped in the wheel); `tests/test_benchmarks.py` runs every case at a tiny size so they keep working.

## Running

Two tools, one command. A scenario runs a whole app through steps and records every frame; a micro case times one function at several sizes.

```bash
uv run manifoldx bench list                                   # scenarios, engine steps, app steps
uv run manifoldx bench run quad_zoom --size 960x540           # a table per phase and a run folder
uv run manifoldx bench run meshes_spin --loop pipelined --json   # the summary alone, for an agent
uv run manifoldx bench show <run folder> --json
uv run manifoldx bench micro benchmarks.suites.frame --case sprites --sizes 1000 100000
uv run manifoldx bench micro benchmarks.suites.frame --case sprites --profile
uv run python -m benchmarks frame --case sprites               # the same micro cases by suite name
```

Scenarios are in `benchmarks/scenarios.py`, micro suites in `benchmarks/suites/`, both listed under `[tool.manifoldx.bench]` in `pyproject.toml`. Chapter 18 of the book explains steps, phases, loops (`--loop serial|pipelined|check`), `--step 1/30`, `--param`, `--smoke` and the run folder (`run.json`, `frames.npz`, `events.jsonl`, `shots/`). A run that fails still writes its folder and exits 1; progress goes to stderr once a second with the step it is on (`step 3/5`), so a monitor can count it.

`--sizes` replaces every selected case's sizes, so select cases of one kind with it: a modeling size is an icosphere subdivision level, and 1000 there runs out of memory.

## Which probe answers which question

Every run records all of them; `manifoldx bench show <folder> --probe P [--phase NAME]` prints one.

| Question | Probe | Where to look |
|---|---|---|
| Where does the frame go: systems, render, waiting for the GPU? | `split`, `systems` | `split_ms`, `systems_ms` per phase |
| Which part of one system is slow? | `spans` | wrap the part in `bench.span("name")`; `spans_ms` shows `system/name` |
| How many draws, material reads, bundles re-recorded or bytes uploaded a frame? | `counters` | mean and max per phase; `counters` in `frames.npz` per frame |
| When did the streamer draw the first detail, and when did it settle? | `streamer` | `first_drawn_s`, `settled_s`, node events by level |
| How long does a build take inside the worker? | `workers` | jobs, p50 and max ms by kind; an app's own pools use `bench.worker_span` |
| How much memory, and how much time in garbage collection? | `memory` | `rss_mb`, `gpu_mb`, `gc`, `folders_mb`; per phase `rss_max_mb`, `gc_ms` |
| Is anything drawn far from the floating origin? | `origin` | `origin_max_m` per phase |
| A number only the app knows (cars, traffic ms)? | `app` | `bench.metric(name, value)`; the summary's `app` dict |
| Was the machine loaded, is this a software rasterizer, a free-threaded build? | `host` | `load_*`, `software`, `free_threaded` |

`cpu_ms` is the process's CPU time over a frame, every thread included; `draw_ms` is the engine's frame on the main thread. A `cpu` well above `draw` is work beside the frame, such as a worker thread's builds.

## Comparing with another copy

manifoldx never calls a version control system. To compare with main, make a copy of main first, then point `--against` at it:

```bash
git worktree add ../manifoldx-main origin/main        # or any copy of another manifoldx project
uv run manifoldx bench micro benchmarks.suites.frame --case sprites --against ../manifoldx-main
uv run manifoldx bench run quad_zoom --against ../manifoldx-main --repeat 3
git worktree remove ../manifoldx-main
uv run manifoldx bench clean                          # the environments --against left in the cache
```

`micro --against` runs the cases in fresh processes, on the other copy's `src/` and on this one, taking turns over `--repeat` rounds (3 by default), and prints a Markdown table of both, ready for a pull request. The other copy needs no `manifoldx.bench`: the harness comes from this one. It exits 1 when a case fails on either side.

`run --against DIR` runs a scenario on another copy of an app (manifoldx is one: its scenarios are in `benchmarks/scenarios.py`) through `uv run --project DIR`, and `run --manifoldx SPEC` runs this copy on another manifoldx through `uv run --with`. The other copy needs uv and a manifoldx with `bench run --json`. Its environment is kept in `~/.cache/manifoldx/bench-envs` (capped at 2 GB, `MANIFOLDX_BENCH_CACHE_GB`) until `bench clean`; check the free disk before the first run on a full machine. `--param k=a,b` sweeps a param the same way, with no uv. Each of these runs every side in fresh processes, alternating the order each round, into one set folder, and ends with `bench compare`'s table; `bench compare A B` makes it from any two runs, sides or a set, with a contact sheet of the shots both took.

## Proving a change in a pull request

1. Before touching the code, add or extend a case that exercises the path, scaling N over at least three sizes up to the largest a real scene uses (100k entities for the frame suite), or a scenario when the cost is a whole frame of an app. Put the scene setup in `benchmarks/scenes.py`.
2. Pin the behaviour with tests first (they must pass on the old code), then change the code.
3. Run the case with `--against` a copy of main on a quiet machine and paste the table into the PR under a `## Timing` heading, with the machine line (`cpus`, GPU or software). For a scenario, run it on both copies and give both phase tables.
4. Say in the PR what the table cannot show, such as GPU time on a real GPU.

## Reading the numbers

- `measure` reports the best of 5 runs; `frame_ms` the median of 5 frames after 2 warm-up frames, at a 64×64 target so the rasterizer adds little. Both are CPU time.
- On a shared or GPU-less machine, run-to-run noise is about ±10 %; a speed-up under 1.3× needs a second run before it counts. With a load average near the CPU count, single cases swing from 0.4× to 1.7× either way. `--against`, sweeps and `compare` alternate the sides over rounds so the load lands on both, report each side's spread, and mark a timing row "needs a quiet re-run" when its ratio is within 1.3× either way and "taken under load" when a run saw a load above half the CPUs. Re-run those rows on a quiet machine before quoting them.
- Without a GPU, wgpu runs on a software Vulkan driver: CPU-side costs are real, GPU-side costs (raster, uploads, readbacks) are not those of a real GPU. Say so when a fix is about GPU transfers.
- `matrix_recomputes` on frame cases counts `TransformCache._recompute_matrices` calls in one frame.
- `web web-crowd` exports an app of `n` boxes in one batch with a query system moving all of them, opens it in headless Chromium (SwiftShader) and reports the median of 60 ticks' `__mx.frameMs.work`: the transpiled systems, the flush and the batches' sync, not the render. `calls` is the last frame's draw calls (1, the one batch). It needs `uv sync --group browser` and a Chromium Playwright can launch; without them the case reports `RuntimeError: playwright not installed` or `chromium not installed`. It reports and gates nothing.

## Writing a case

```python
from manifoldx.bench import case
from manifoldx.bench.micro import measure

@case("modeling", "my_op", sizes=[5, 6, 7], smoke=1)
def my_op(level):
    """One line saying what is timed (shown by --list)."""
    m = ...                      # setup, not timed
    return measure(lambda: m.my_op())   # ms; or {"ms": ..., "faces": ...} to print extra numbers
```

`smoke` is the size the test suite runs: keep it under a tenth of a second. A case that needs the GPU calls `harness.start(engine)` or `engine._ensure_offscreen(1)`; on a machine without an adapter the test suite skips it.
