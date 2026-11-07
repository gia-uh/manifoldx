---
when: making something faster, checking whether a change made something slower, or proving a performance change in a pull request
triggers: ["performance", "benchmark", "slow", "faster", "speed-up", "profile", "per frame", "scaling", "large scene", "perf"]
---

# Benchmarking and proving a performance change

## When to use this

Reach for this before and after any change meant to make manifoldx faster, and when a review asks whether a change costs time. The tools are in `benchmarks/` (not shipped in the wheel); `tests/test_benchmarks.py` runs every case at a tiny size so they keep working.

## Running

```bash
uv run python -m benchmarks --list                          # suites, cases, default sizes
uv run python -m benchmarks frame                           # one suite, every case, default sizes
uv run python -m benchmarks frame --case sprites --sizes 1000 100000
uv run python -m benchmarks frame --case sprites --profile  # adds a cProfile top-15 per case
uv run python -m benchmarks frame --case sprites --against main
```

Suites: `frame` (whole scenes, CPU per `Engine._draw_frame`), `ecs`, `compute`, `skinning`, `modeling`, `loaders`, `web` (the exported page in headless Chromium, see below). `--sizes` replaces every selected case's sizes, so select cases of one kind with it: a modeling size is a subdivision level, and 1000 there runs out of memory.

`--against REF` extracts `src/` of that git ref to a temp folder and runs the selected cases twice in fresh processes, on REF's code and on this checkout's, then prints a Markdown table with the speed-up. It refuses a run that imported manifoldx from anywhere else. The suites themselves come from this checkout, so a new case shows `n/a` or `error` on the old code, which is expected for a new API.

## Proving a change in a pull request

1. Before touching the code, add or extend a case that exercises the path, scaling N over at least three sizes up to the largest a real scene uses (100k entities for the frame suite). Put the scene setup in `benchmarks/scenes.py` when another case can reuse it.
2. Pin the behaviour with tests first (they must pass on the old code), then change the code.
3. Run `uv run python -m benchmarks <suite> --case <name> --against main` on a quiet machine and paste the table into the PR under a `## Timing` heading, with the machine line (`cpus`, GPU or software rasterizer).
4. Say in the PR what the table cannot show, such as GPU time on a real GPU.

## Reading the numbers

- `measure` reports the best of 5 runs; `frame_ms` the median of 5 frames after 2 warm-up frames, at a 64×64 target so the rasterizer adds little. Both are CPU time.
- On a shared or GPU-less machine, run-to-run noise is about ±10 %; a speed-up under 1.3× needs a second run before it counts.
- Without a GPU, wgpu runs on a software Vulkan driver: CPU-side costs are real, GPU-side costs (raster, uploads, readbacks) are not those of a real GPU. Say so when a fix is about GPU transfers.
- `matrix_recomputes` on frame cases counts `TransformCache._recompute_matrices` calls in one frame.
- `web web-crowd` exports an app of `n` boxes in one batch with a query system moving all of them, opens it in headless Chromium (SwiftShader) and reports the median of 60 ticks' `__mx.frameMs.work`: the transpiled systems, the flush and the batches' sync, not the render. `calls` is the last frame's draw calls (1, the one batch). It needs `uv sync --group browser` and a Chromium Playwright can launch; without them the case reports `RuntimeError: playwright not installed` or `chromium not installed`. It reports and gates nothing.

## Writing a case

```python
from benchmarks.harness import case, measure

@case("modeling", "my_op", sizes=[5, 6, 7], smoke=1)
def my_op(level):
    """One line saying what is timed (shown by --list)."""
    m = ...                      # setup, not timed
    return measure(lambda: m.my_op())   # ms; or {"ms": ..., "faces": ...} to print extra numbers
```

`smoke` is the size the test suite runs: keep it under a tenth of a second. A case that needs the GPU calls `harness.start(engine)` or `engine._ensure_offscreen(1)`; on a machine without an adapter the test suite skips it.
