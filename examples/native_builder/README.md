# Native cell builder (S-b spike)

A game's cell builder in Rust, registered from Python as a versioned capsule, next
to the same builder in Python. This is the S-b spike from the Rust core spec; it is
not part of the engine's public API yet.

- `native/` — the game crate. Depends on `crates/manifoldx-sdk`, exports
  `build_cell` as a `PyCapsule` named `manifoldx.builder.v1`, plus a wrong-version
  capsule and a panicking one for the failure tests.
- `cell.py` — the regular 1 km² cell in Python: `build_cell` in the loop shape the
  current streamer uses, and `build_cell_numpy` as the vectorized floor.
- `irregular.py` — the S-b2 cell: a street graph with missing edges, BFS distances,
  variable lots, and the same loop and vectorized builders.
- `../../benchmarks/spikes/s_b_builders.py` — S-b: builds both cdylibs, compares the
  bytes, times them, measures the process-pool memory and runs the failure corpus.
- `../../benchmarks/spikes/s_b2_irregular.py` — S-b2: the same checks on the irregular
  cell, plus the numpy floor.

```bash
cd repos/manifoldx
uv run python benchmarks/spikes/s_b_builders.py --build
uv run python benchmarks/spikes/s_b_builders.py
uv run python benchmarks/spikes/s_b2_irregular.py
```

The cell: a 25 m street grid over 1 km², one box per block with a deterministic
height (every 11th block a park), and a lamp row every third intersection. The
results and their caveats are in `know-how/rust-core-s-b.md` and
`know-how/rust-core-s-b2.md`.
