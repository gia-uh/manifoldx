---
when: rerunning or reviewing S-d, the spike that decides over 100k rows by building IR only
triggers: ["rust core", "S-d", "IR", "selection", "commands", "100k", "rows read", "apply"]
---

# S-d: 100k rows decided by IR, no rows read, 0.99 ms to apply

## The question

Can a Python system decide over 100k entities by building a selection and
enqueuing commands, without reading a single row, while the frame stays at 60 Hz?

## Method

`benchmarks/spikes/s_d_ir.py` over `crates/spike-s-d`:

- A Rust store of 100,000 agents, eight f32 columns (x, z, vx, vz, speed,
  target_speed, ttl, energy).
- Python builds a selection with `world.select("speed", "<", 2.0)`, which stores
  a spec in Rust and returns an id. `world.set(sel, "target_speed", 8.0)` and
  `world.spawn(40)` enqueue commands. No method returns a row.
- A Rust loop paces 600 frames at 60 Hz, runs five kernels (position integration,
  speed toward target, ttl decay, vx from the speed error, energy decay), calls the
  Python system every 30 frames (2 Hz), and applies its commands at the frame
  boundary. Apply time is measured inside Rust.
- The naive path, for contrast: `world.read_speeds()` returns the column as a
  Python list and increments the row counter.

## Results

| measurement | value |
|---|---|
| Selection `speed < 2.0` | 20,000 of 100,000 rows |
| Python decision (select + set + spawn), p50 | **0.036 ms** |
| Naive `read_speeds` (100k floats to Python) | 12.46 ms |
| Commands applied per batch, p50 / p99 / max | **0.482 / 0.990 / 0.990 ms** |
| Frame period, p50 / p99 / max | 16.667 / **16.674** / 18.89 ms |
| Frames over 33 ms | 0 |
| Rows read into Python | **0** |
| Commands | 20 sets, 20 spawns, 32,360 rows affected |
| Entities after the run | 100,800 |

All three S-d gates pass: zero rows read, apply p99 under 1 ms, frame p99 under
16.7 ms.

## What it means

**The zero-row rule works.** The Python system's whole decision costs 0.036 ms and
touches no rows; reading the same column into Python costs 12.46 ms, 345× more,
and that is a single column. A system that reads rows at 2 Hz spends a quarter of
every frame in the read alone; the IR path spends 0.04 ms.

**The apply fits the frame.** A batch scans the 100k rows, matches 20,000 on the
first call and writes them: 0.48 ms median, 0.99 ms worst over 20 batches. The
frame keeps its 16.67 ms period with five kernels, the 2 Hz system and the apply,
and never goes over 33 ms. This is the first spike whose gate passes without a
caveat about the current engine already doing it.

**The selection is fresh at apply time.** The async system stores a spec, not an
index list; the scan happens when the command is applied, so rows that moved
between the decision and the frame are seen as they are. That is the design's
"commands carry their expression" rule, and it costs the 0.5–1 ms measured.

**What is not tested here.** The five kernels are fixed Rust loops, not the
IR→Rust CPU backend or WGSL; the selection is a single column predicate, not the
full `query().where()` IR; and the store has no secondary indexes, so every batch
scans 100k rows. A real store would add an index for hot predicates, which can
only lower the apply time.

## Caveats

- Synthetic rows and eight columns; a real agent row carries more, and the scan
  grows with the row width.
- The spawn adds 800 rows over the run (0.8%); a spawn-heavy system would grow the
  scan, which is exactly where an index or an archetype split matters.
- One 600-frame run on 4 vCPUs, free-threaded 3.14.4; the p99 is stable but the
  max frame was 18.89 ms.
- The apply time is measured inside Rust and includes the command dispatch, the
  scan and the writes; it excludes the Python call that enqueued them (0.036 ms).

## Rerun

```bash
cd repos/manifoldx
uv run python benchmarks/spikes/s_d_ir.py --build
uv run python benchmarks/spikes/s_d_ir.py
```

The harness writes `benchmarks/spikes/s_d_results.json` and prints the table.
