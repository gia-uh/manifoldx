---
when: rerunning or reviewing S-c, the reload spike with a store that survives Python
triggers: ["rust core", "S-c", "reload", "hot reload", "store", "native incremental", "dev loop"]
---

# S-c: the store survives, but the desktop reload was already fast

## The question

Can a reload keep 10k rows and 200 mesh handles while the Python app module is
reimported and its systems swapped, in under a second, and can a native crate
rebuild incrementally in under five?

## Method

`benchmarks/spikes/s_c_reload.py`, three measurements:

- **prototype**: `crates/spike-s-c` holds 10k rows, 200 mesh handles and the frame
  counter in Rust. A Python app module is written, imported, and its `setup`
  returns the system list; the harness reimports the module, swaps the systems,
  and runs one frame. Ten reloads; counts, frame and RSS checked after each. The
  new module's system must run (`CALLS == 1`), proving the swap.
- **current**: the same shape of app (117 distinct meshes, 10k entities) on the
  current `HotReloader`, headless, which reruns the whole script and rebuilds the
  engine. Five reloads.
- **native**: touch the game crate's source, run `cargo build --release`, then
  load the new cdylib and open a builder capsule. Five builds.

## Results

| measurement | p50 | p95 |
|---|---:|---:|
| Prototype: save to first frame | **0.79 ms** | 1.05 ms |
| Current desktop reloader, 117 meshes / 10k entities | **83.8 ms** | 101.7 ms |
| Native incremental build | **840 ms** | 859 ms |
| Load the new cdylib and open the capsule | 4.1 ms | — |

The prototype kept 10,000 entities and 200 meshes through ten reloads, ran the
new module's system each time, and grew RSS by 0.88%. The current reloader's app
ended with 10,117 entities after each reload (117 meshes + 10k rows).

All the S-c gates pass: Python p50 < 1 s, p95 < 1.5 s, counts survive, RSS < 5%,
native p50 < 5 s.

## What it means

**The gate does not discriminate on the desktop.** The current `HotReloader`
already reloads this app in 84 ms, not the 3.1–3.5 s the spec attributed to it.
That number in #164 is `run --web`, and its Python phase is 3.07 s: startup 1.29 s
(rerunning the app's startup, loading 117 OBJs and 18 textures), reach 1.18 s
(walking the app's code again), and about 0.2 s of snapshot, transpile and write.
The tab swap is another 209 ms. The desktop reloader was never the problem.

**The prototype shows the mechanism, not the web win.** The store and the system
swap cost 0.79 ms; the scene is not rebuilt and the rows never leave Rust. That is
exactly what removes the 1.29 s startup and, if the core keeps the IR, the 1.18 s
reach from the web reload. But the prototype does not model the web export's
snapshot, transpile, write or the tab swap, and the app module is synthetic. The
end-to-end web number is still unmeasured; the remaining phases from #164 add up
to about 0.2 s, so the 1 s target looks reachable, not proven.

**The native half passes.** An 840 ms incremental build plus a 4 ms load is well
under the 5 s gate, and that is with pyo3 and the SDK already compiled. The design
rule that native code keeps no static state is what makes the swap possible; the
spike loads a new cdylib in a fresh process, not in the running one.

**For the migration, S-c is a pass with the same shape as S-a.** The current
engine already meets the desktop requirement. The store's value is the web reload
and apps whose startup is expensive; the prototype shows it can be sub-millisecond
but not that the web page gets there.

## Caveats

- The prototype's app module is small; a real five-module app imports more, and
  the core's IR rebuild is not modelled.
- The current app is synthetic: 117 cubes, not 117 OBJs and 18 textures. The real
  startup in #164 is slower because of file decoding, which the store would skip.
- The web reload was not measured here; #164's numbers are from a loaded 4-core
  VPS with lavapipe and headless SwiftShader.
- The native load was measured in a fresh process; the in-process capsule swap is
  a design detail, and the build dominates.

## Rerun

```bash
cd repos/manifoldx
uv run python benchmarks/spikes/s_c_reload.py
```

The harness writes `benchmarks/spikes/s_c_results.json` and prints the table.
