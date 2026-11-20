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

Scenarios are in `benchmarks/scenarios.py`, micro suites in `benchmarks/suites/`, both listed under `[tool.manifoldx.bench]` in `pyproject.toml`. Chapter 18 of the book explains steps, phases, loops (`--loop serial|pipelined|check`), `--step 1/30`, `--param`, `--smoke` and the run folder (`run.json`, `frames.npz`, `events.jsonl`, `shots/`). A run that fails still writes its folder and exits 1; progress goes to stderr once a second, so a monitor can count it.

## Comparing with another copy

manifoldx never calls a version control system. To compare with main, make a copy of main first, then point `--against` at it:

```bash
git worktree add ../manifoldx-main origin/main        # or any copy of another manifoldx project
uv run manifoldx bench micro benchmarks.suites.frame --case sprites --against ../manifoldx-main
git worktree remove ../manifoldx-main
```

The cases run twice in fresh processes, on the other copy's `src/` and on this one, and a Markdown table of both is printed, ready for a pull request. The other copy needs no `manifoldx.bench`: the harness comes from this one.

## Proving a change in a pull request

1. Before touching the code, add or extend a case that exercises the path, scaling N over at least three sizes up to the largest a real scene uses (100k entities for the frame suite), or a scenario when the cost is a whole frame of an app. Put the scene setup in `benchmarks/scenes.py`.
2. Pin the behaviour with tests first (they must pass on the old code), then change the code.
3. Run the case with `--against` a copy of main on a quiet machine and paste the table into the PR under a `## Timing` heading, with the machine line (`cpus`, GPU or software). For a scenario, run it on both copies and give both phase tables.
4. Say in the PR what the table cannot show, such as GPU time on a real GPU.

## Reading the numbers

- `measure` reports the best of 5 runs; `frame_ms` the median of 5 frames after 2 warm-up frames, at a 64×64 target so the rasterizer adds little. Both are CPU time.
- On a shared or GPU-less machine, run-to-run noise is about ±10 %; a speed-up under 1.3× needs a second run before it counts.
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
