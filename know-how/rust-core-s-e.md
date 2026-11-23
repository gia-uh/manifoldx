---
when: rerunning or reviewing S-e, the nbody and boids examples with a native step
triggers: ["rust core", "S-e", "nbody", "boids", "kernels", "numpy", "native step"]
---

# S-e: the nbody and boids examples are 37–219× faster natively

## The question

Do the two simulation examples improve when their hot loop runs in Rust instead
of numpy, at the sizes they ship with and at larger ones?

## Method

`benchmarks/spikes/s_nbody_boids.py`, with the native steps in
`crates/spike-nbody-boids` (pyo3 + numpy + rayon):

- **nbody**: the example's all-pairs gravity, same `max(dist, softening)` and the
  same velocity/position update. The numpy side calls the example's
  `mx.physics.gravity`; the native side computes the forces on the fly and
  parallelizes over bodies, so it never allocates an N×N×3 tensor.
- **boids**: the example's rules and constants. The numpy side is the example's
  all-pairs code; the native side uses a uniform grid at the perception radius
  (4 m, 27 cells) and parallelizes the force pass.
- Ten steps after two warm-ups, median per step, 4 vCPUs, no rendering.
- Correctness: one step from the same arrays; the largest difference is 1.2e-7
  for both, with predators far away in the boids case so only the boid math is
  compared.

## Results

| workload | N | numpy ms | native ms | speed-up | note |
|---|---:|---:|---:|---:|---|
| nbody | 500 | 32.34 | 0.88 | **36.7×** | example default |
| nbody | 1,000 | 110.04 | 2.71 | **40.6×** | |
| nbody | 2,000 | 440.84 | 10.49 | **42.0×** | |
| nbody | 5,000 | — | 51.78 | — | numpy skipped (memory) |
| nbody | 10,000 | — | 239.88 | — | numpy skipped (memory) |
| boids | 300 | 17.63 | 0.32 | **54.7×** | example default |
| boids | 1,000 | 209.92 | 1.09 | **192.1×** | |
| boids | 2,000 | 754.69 | 3.45 | **218.7×** | |
| boids | 5,000 | — | 12.91 | — | numpy skipped (memory) |
| boids | 10,000 | — | 53.75 | — | numpy skipped (memory) |

Full data: `benchmarks/spikes/s_nbody_boids_results.json`.

## What it means

**The nbody example is already over budget in numpy.** At its default 500 bodies
a step costs 32.34 ms, almost two 60 Hz frames, before any rendering. The native
step is 0.88 ms, 36.7× faster, and the same all-pairs math: the win is not an
algorithm change, it is not allocating the N×N×3 diff and distance tensors and
letting four cores do the loop. At 2,000 bodies numpy is 441 ms and native 10.5 ms;
at 10,000 the native step is 240 ms and numpy would need gigabytes for the
temporaries.

**The boids win is part algorithm, part native.** The native side uses a grid, so
it only visits neighbours within 4 m; the numpy example visits every pair. That is
why the speed-up grows with N (54.7× at 300, 219× at 2,000). Even at the default
300, where the grid is tiny, the step drops from 17.63 ms to 0.32 ms: the numpy
example is over the 16.7 ms frame before rendering.

**This is the workload the core's native systems and kernels are for.** S-b2 found
only 2.8× against a well-vectorized cell builder; these examples find 37× and up,
because the per-entity work is real, the numpy code allocates large temporaries,
and the native side can parallelize. It does not rescue the frame loop or the
renderer, but it is a strong, concrete case for native kernels.

## Caveats

- No rendering or GPU upload in the timing; the examples' full frames add that.
- The numpy boids could also use a grid; the comparison is native grid against
  numpy all-pairs, so part of the 54–219× is algorithmic. The nbody comparison is
  the clean one: same algorithm, 37–42×.
- The native boid predator wander uses a deterministic xorshift, not numpy's
  `randn`; the correctness run puts predators far away and compares the boid math.
- Four vCPUs on a shared server; the pyo3 call is included in the native time.
  Rayon at 500 bodies still wins, but the gap narrows at small N.

## Rerun

```bash
cd repos/manifoldx
uv run python benchmarks/spikes/s_nbody_boids.py --build
uv run python benchmarks/spikes/s_nbody_boids.py
```

The harness writes `benchmarks/spikes/s_nbody_boids_results.json` and prints the
table.
