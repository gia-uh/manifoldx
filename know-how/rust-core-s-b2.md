---
when: rerunning or reviewing S-b2, the irregular cell that tests whether the native speed-up widens
triggers: ["rust core", "S-b2", "irregular", "cell", "BFS", "native builder", "numpy floor"]
---

# S-b2: irregularity moves the floor from 2.1× to 2.8×, not to 4×

## The question

S-b put a native cell builder at 25.4× the loop-shaped Python builder but only
2.1× the vectorized numpy one, on a regular grid. S-b2 asks whether a graph, BFS
distances and variable lots — the irregular work the migration is supposed to win
— widen that floor.

## Method

`benchmarks/spikes/s_b2_irregular.py`, the same two cdylibs and capsule from S-b,
a different cell:

- A 25 m street grid with a ninth of its edges removed by hash, so the graph has
  dead ends and disconnected corners.
- BFS distances from the centre over that graph.
- Four candidate lots per block, kept by the block's hash (parks every 13th
  block), so blocks carry one to four buildings of 9.5 m footprints.
- Building heights read the BFS distance at the block's corner plus a hash.
- Lamp rows are 16 bytes: x, z, kind, and the BFS distance.
- Roads are only the surviving edges.

Output: 70,896 building vertices, 106,344 indices, 17,496 road indices and 8,976
bytes of lamp rows for seed 7.

Three builders, 20 runs each after 3 warm-ups, median: the loop shape, the
vectorized numpy shape with the graph and BFS still in Python, and native through
the capsule. Bytes are compared on seeds 0, 7 and 12345. Memory is a two-worker
process pool against eight in-process native builds.

## Results

| check | result |
|---|---|
| bytes identical, seeds 0, 7, 12345 | yes, for the loop and the numpy builder |
| Python loop, p50 | 110.06 ms |
| Python numpy, p50 | 15.18 ms |
| Native, p50 (inside the call) | 5.46 ms (4.39 ms) |
| Speed-up vs the loop builder | **20.1×** |
| Speed-up vs the vectorized builder | **2.8×** |
| Process pool, peak RSS (2 workers) | 157.1 MB (parent 27.4, workers 64.9 each) |
| Native, peak RSS | 32.4 MB |
| Native / pool | **20.6%** |

Full data: `benchmarks/spikes/s_b2_results.json`.

## What it means

**The floor moved, but not to 4×.** The graph and the BFS made the vectorized
builder 1.3× slower relative to native than the regular grid (2.1× → 2.8×), which
is the direction the migration expected. It is not enough: a numpy author who
batches the geometry still gets within 2.8× of the native builder on this cell.
The graph and BFS are small next to the 70k vertices.

**The current builder is still 20× behind.** Against the loop shape the streamer
uses today, the native builder is 20× faster. That is the number a port buys
without rewriting the app's builder in numpy. The question the migration has to
answer is whether the apps in question will rewrite their builders in numpy or
keep the loop shape.

**Memory and the boundary do not move.** 20.6% of the two-worker pool, and the
same capsule contract as S-b: identical bytes, wrong versions fail at
registration, panics become exceptions. Those wins are independent of how
irregular the geometry is.

**If 4× against the best Python is the bar, the next workload is facades.** Walls
with bands, windows, doors and varied footprints are per-feature geometry with
variable counts; batching them in numpy is much harder than batching boxes. A
per-feature facade cell is the next honest test, and a bigger one than this.

## Caveats

- A synthetic graph and synthetic lots; a real province loads geographic data,
  which is I/O and not measured here.
- The numpy builder still builds the graph and runs BFS in Python; vectorizing
  those is possible but the BFS is sequential.
- Two pool workers, no warm world. With the real warm data the memory ratio only
  improves for Rust.
- Four vCPUs, free-threaded 3.14.4, load average 0.45. Native p95 had one 14 ms
  outlier; the p50 is stable across the runs.

## Rerun

```bash
cd repos/manifoldx
uv run python benchmarks/spikes/s_b2_irregular.py --build
uv run python benchmarks/spikes/s_b2_irregular.py
```

The harness writes `benchmarks/spikes/s_b2_results.json` and prints the table.
