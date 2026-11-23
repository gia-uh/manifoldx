---
when: rerunning or reviewing S-b, the spike that puts a game's cell builder in Rust behind a capsule
triggers: ["rust core", "S-b", "native builder", "capsule", "SDK", "cell", "streaming", "process pool"]
---

# S-b: a native cell builder passes, with a numpy-sized caveat

## The question

Can a game write its cell builder in Rust, register it from Python as a versioned
capsule, and get the same bytes faster and with less memory than the Python
builder and the process pool? This is the spike that decides the migration once
S-a removed the frame loop as the reason.

## Method

`benchmarks/spikes/s_b_builders.py`, two separately compiled cdylibs:

- `examples/native_builder/native`: the game's builder, against
  `crates/manifoldx-sdk`. It exports `manifoldx.builder.v1` as a `PyCapsule`, and a
  panicking builder for the failure tests.
- `crates/spike-s-b`: the engine side, a stand-in for the streamer. It opens the
  capsule once, checks the name, and calls the pointer with a collector that owns
  the output buffers.

The workload is a 1 km² cell: a 25 m street grid (41 lines each way), one box per
block with a deterministic height (every 11th block is a park), and a lamp row
every third intersection. It produces 35,200 vertices, 52,800 indices and 6,720
bytes of entity rows.

Three builders, 20 runs each after 3 warm-ups, median:

- **python loop**: the shape the current streamer uses, a Python loop appending
  lists, converted to numpy at the end.
- **python numpy**: the same cell vectorized, the floor of what numpy can do on a
  regular grid.
- **native**: through the capsule, including the collector copies and the return
  to Python bytes.

Memory: a `ProcessPoolExecutor` with two spawned workers building eight cells, the
shape of `streaming.py`'s `Builders`, against eight cells in one process through
the capsule. Peak RSS is `VmHWM` of the parent plus each worker.

Failure cases: a panicking builder, a capsule with a wrong version, nine
non-capsule inputs, 10,000 arbitrary Python objects, and 1,000 capsules with a
wrong name.

## Results

| check | result |
|---|---|
| bytes identical, seeds 0, 7, 12345 | yes: positions, normals, indices and rows |
| Python loop, p50 | 46.14 ms |
| Python numpy, p50 | 3.84 ms |
| Native, p50 (inside the call) | 1.81 ms (1.59 ms) |
| Speed-up vs the loop builder | **25.4×** |
| Speed-up vs the vectorized builder | **2.1×** |
| Process pool, peak RSS (2 workers) | 133.0 MB (parent 27.6, workers 52.7 each) |
| Native, peak RSS | 27.3 MB |
| Native / pool | **20.6%** |
| Panic | `RuntimeError: native builder failed (code 2): panic: boom…`, process alive, next build works |
| Wrong version | `ValueError: … not named manifoldx.builder.v1: compiled against another SDK version` |
| Non-capsules / fuzz objects / wrong capsules | 9/9, 10,000/10,000, 1,000/1,000 rejected |

Full data: `benchmarks/spikes/s_b_results.json`.

## What it means

**S-b passes its gate.** Against the builder the current streamer uses, the native
builder is 25× faster, uses a fifth of the process-pool memory, produces identical
bytes, and fails safely at the boundary. The capsule contract works: two wheels,
no shared Rust types, the engine owns the memory, a panic becomes a Python
exception.

**The caveat is the numpy floor.** A vectorized numpy builder does the same cell in
3.84 ms, so the native advantage over the best Python is 2.1×, not 25×. The grid is
regular enough to broadcast; a real cell builder is not. Road graphs, parcels,
spatial indexes, pathfinding and per-building variation are where numpy stops
being a spreadsheet and becomes a Python loop again. S-b measured the current
shape, not that irregular work; an S-b2 on a road graph with a BFS pass is the
honest next test before the migration is called justified.

**The memory win is durable.** The pool duplicates the interpreter, numpy and the
app's warm data per worker; the native path does not. The 20.6% here has no warm
world loaded. With the 2.5 GB a province loads today, the ratio only improves for
Rust, and the process count stops being a memory multiplier.

**The boundary is the part that is clearly better than today.** A versioned
capsule fails at registration with a readable error; a panic is caught; malformed
inputs are rejected; nothing crosses but `#[repr(C)]` structs and function
pointers. That is the SDK the migration was designed around.

## Caveats

- A regular grid is the best case for numpy and the worst case for the native
  speed argument. The 2.1× is a floor, not a forecast.
- Two pool workers, no warm world. The real pool has more workers and gigabytes
  per worker; the memory ratio would be smaller for Rust.
- CPython refuses to create a capsule with a null pointer, so the loader's null
  check is defensive and untested. The name is the whole guarantee: a capsule with
  the right name and a data pointer opens, and only a call would crash. This is
  the same trust numpy's capsules carry.
- The panic test prints the default Rust panic line to stderr; the process
  survives and the next build works.
- Four vCPUs, free-threaded 3.14.4, load average 0.6. Timings are medians of 20.

## Rerun

```bash
cd repos/manifoldx
uv run python benchmarks/spikes/s_b_builders.py --build
uv run python benchmarks/spikes/s_b_builders.py
```

The harness writes `benchmarks/spikes/s_b_results.json` and prints the table.
