---
when: rerunning or reviewing S-a, the spike that asks whether a slow async system moves the frame
triggers: ["rust core", "S-a", "frame stability", "async system", "GIL", "GC pause", "free-threaded"]
---

# S-a: a slow async system does not move the frame on 3.14t

## The question

The Rust core was justified first by a frame that stays flat while an async system takes
300 ms. This spike measures that claim before any of the core exists, and compares the
Rust loop with the current engine in its pipelined mode.

## Method

`benchmarks/spikes/s_a_frame.py`, three loops, three modes, three runs of six seconds at
60 Hz on the reference machine (4 vCPU Xeon, free-threaded CPython 3.14.4, load average
1.2 at the start):

- **rust**: `crates/spike-s-a`, a paced loop in Rust that releases the GIL, calls one no-op
  Python sync callback per frame, and runs async callbacks on Rust threads.
- **rust-nosync**: the same loop with no Python callback at all, the shape of a frame lane
  made only of kernels and native systems.
- **current**: the current engine headless with `_pipeline_on = True`, the path `run()` takes
  on a free-threaded build, with one no-op system.

Modes:

- **base**: no async work.
- **async**: a Python job at 2 Hz that burns 300 ms of CPU and allocates 50 MB, no `sleep`.
  The point is to hold the interpreter, not to yield it.
- **async+gc**: the same job plus a forced collection of 400k cycles every second, a
  stop-the-world pause of 125–160 ms.

`over33` is the sum over the three runs (1080 frames).

## Results

| loop | mode | p50 ms | p99 ms | max ms | frames >33 ms | p99 / base |
|---|---|---|---|---|---|---|
| rust | base | 16.67 | 16.68 | 27.1 | 0 | 1.00 |
| rust | async | 16.67 | 16.69 | 19.4 | 0 | 1.00 |
| rust | async+gc | 16.67 | 51.75 | 79.8 | 42 | 3.10 |
| rust-nosync | base | 16.67 | 16.93 | 19.3 | 0 | 1.00 |
| rust-nosync | async | 16.67 | 16.71 | 19.1 | 0 | 0.99 |
| rust-nosync | async+gc | 16.67 | 17.00 | 20.1 | 0 | 1.00 |
| current | base | 16.67 | 16.77 | 18.1 | 0 | 1.00 |
| current | async | 16.67 | 17.31 | 19.3 | 0 | 1.03 |
| current | async+gc | 16.67 | 95.93 | 165.5 | 48 | 5.72 |

Full data: `benchmarks/spikes/s_a_results.json`.

## What it means

**The async hypothesis is rejected.** A CPU-bound Python system at 2 Hz, 300 ms per call,
moves neither loop: p99 is 1.00× on Rust and 1.03× on the current engine, with zero frames
over 33 ms in either. Free-threaded CPython plus pipelined frames already give the frame
independence from async Python that the Rust loop was meant to provide. S-a is a **no** by
its own gate: the current engine does not show the penalty the spike was looking for.

**GC is the remaining stall, and it is not the loop's to fix.** A stop-the-world pause of
125–160 ms costs frames in both. With a no-op Python sync callback, the Rust loop has a
lower p99 (51.75 ms against 95.93 ms) and a slightly lower count of late frames (42 against
48 of 1080), but it still loses them: the frame waits for the callback, and the callback
cannot run during the pause. Only **rust-nosync** keeps zero late frames, because the frame
never touches Python. That is the real condition: a Python-free frame lane, not a Rust
loop with Python systems in it.

**Consequence for the migration.** On the free-threaded build the loop does not justify
Rust. The case now rests on S-b (native irregular work) and on the IR evaluator, exactly
the steelman this spike was built to test. The stop rule still holds: if S-b also fails,
the Rust core stops and the kernel language continues on the current engine.

## Caveats

- The current engine is driven headless through `_draw_frame` with pipelining forced, not
  through a GLFW window; the frame work is the same, the pacing is the harness's.
- One no-op system, one async job, a 64×64 offscreen target. More async jobs compete for
  cores, not for the GIL; Rust does not add cores.
- The reference is the free-threaded build. On a GIL build the same async job would hold
  the interpreter, and neither a Rust loop with Python sync systems nor the current engine
  would be immune; only a Python-free frame lane would.
- Six-second runs with three repeats; the GC pause is synthetic and larger than a typical
  incremental collection in 3.14.

## Rerun

```bash
cd repos/manifoldx
uv run python benchmarks/spikes/s_a_frame.py --build
uv run python benchmarks/spikes/s_a_frame.py --seconds 6 --repeats 3
```

The harness writes `benchmarks/spikes/s_a_results.json` and prints the table above.
