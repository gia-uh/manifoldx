# ManifoldX — Agent Instructions

## What is ManifoldX?

A pure-Python real-time 3D rendering engine on top of `wgpu`, with an Entity Component System (ECS) architecture. The goal is to let researchers run large-N simulations and visualize them in 3D from the same NumPy-based codebase — no graphics expertise required.

Published as `manifold-gfx` on PyPI. Beta / academic project — expect breaking changes.

Key concepts:
- **ECS with SoA layout.** Each component is a `(max_entities, total_cols)` numpy array; per-frame physics runs as vectorized numpy ops over entity arrays, no Python loops in the hot path.
- **Material = pipeline.** Each material type compiles its own WGSL shader and owns a per-batch uniform layout. The renderer batches entities by `(geometry_id, material_type[, subtype])` and issues one instanced draw call per batch.
- **Three render paths:** mesh (3D geometry + transforms), sprite (camera-facing point sprites with sphere imposters), label (camera-facing billboards textured from a shared atlas). All three coexist in one render pass.

## Quick Commands

```bash
make test           # full test suite (uses offscreen wgpu backend)
make lint           # ruff check
make format         # ruff format

# Run an example interactively
uv run python examples/<name>.py

# Run with hot reload: saving a source file reruns it in the same window (know-how/hot-reload.md)
uv run manifoldx run examples/<name>.py

# Benchmarks, before/after against main (know-how/benchmarking.md)
uv run python -m benchmarks frame --case sprites --against main

# Smoke-render an example to MP4 (good for visual regression)
uv run python examples/<name>.py --render --duration 2 --fps 30 --output /tmp/<name>.mp4
```

## Project Structure

- **Python**: 3.13+, managed via `uv`
- **Source**: `src/manifoldx/` — `engine.py`, `ecs.py`, `components.py`, `renderer.py`, `materials.py`, `camera.py`, `resources.py`, plus `viz/` for sci-viz primitives
- **Tests**: `tests/` — top-level for engine/ECS, `tests/viz/` for sci-viz primitives
- **Examples vs demos** — two kinds, split by folder:
  - `examples/` — **focused**: one file, one feature, minimal code (documentation). Named by the feature (`shadows.py`, `fields.py`, `booleans.py`), not `*_demo.py`. New single-feature showcases go here, minimal.
  - `demos/` — **show-off**: game-like / sim-like experiences combining many systems, possibly multi-file + assets (`terrain_stream.py`, `labyrinth_fps.py`, `sun_earth_moon.py`, `smoke.py`). New multi-system experiences go here.
  - Each folder has a `README.md` stating the convention. Long-term demo roadmap: `.knowledge/analysis/2026-07-28-demo-roadmap-lague-inspired.md`.
- **Plans**: `.knowledge/plans/` — TDD-structured implementation plans, each task split into "write failing test → confirm fail → implement → confirm pass → commit" steps
- **Changelog**: `CHANGELOG.md` — Keep-a-Changelog format, `[Unreleased]` accumulates between version bumps

## Key Conventions

- Colour space: material colours (`color=`) are sRGB, like texture texels; materials decode them with `resources.material_rgb`. Shaders write linear colour and the sRGB render target encodes it once; no shader applies `pow(color, 1/2.2)`.
- Use `uv run` for all Python invocations (`uv run pytest`, `uv run python examples/...`).
- Conventional commit messages: `feat(scope):`, `refactor(scope):`, `test(scope):`, `fix(scope):`, `docs(scope):`, `chore(scope):`.
- ECS components inherit from `manifoldx.components.Component`; dtype + shape are derived from class-level annotations (`Float`, `Vector3`, `Vector4` from `manifoldx.types`). The Engine auto-registers them on first spawn — no manual `engine.store.register_component(...)` boilerplate.
- Built-in `Transform` / `Mesh` / `Material` use explicit `register(store)` static methods — they pre-date the `Component` base and have richer `get_data` semantics that don't fit the simple-field pattern.
- Materials expose `_compile()` (returns WGSL string), `uniform_type()` (dict of field → wgsl type), optional `pipeline_subtype` (string, included in pipeline cache key).
- Tests that need a real wgpu device gate on `get_offscreen_canvas` and `pytest.skip` if the backend is unavailable, so the suite stays runnable on machines without GPU.
- Plans live in `.knowledge/plans/<YYYY-MM-DD>-<slug>.md`. Sub-projects (e.g. sci-viz primitives v1) span multiple plans (`plan-1-foundation.md`, `plan-2-text-rendering.md`, ...). Each task in a plan ends with a commit step; each sub-project ends with a CHANGELOG entry under `[Unreleased]`.

## Know-how

Procedure docs in `know-how/` encode how to do things in this repo. Match the current task against the *when to reach for it* line and load the matched doc(s) before acting. Each doc has its own `triggers` frontmatter for finer-grained matching.

- [authoring-an-ecs-component](know-how/authoring-an-ecs-component.md) — adding a new ECS component (PointCloud-style marker, ScalarValue-style scalar, Velocity-style vector, etc.) using the `Component` base class with annotation-driven dtype + shape.

- [benchmarking](know-how/benchmarking.md) — making something faster or proving a performance change in a PR: `benchmarks/` suites, `--against main` before/after tables.

- [hot-reload](know-how/hot-reload.md) — running an app with `manifoldx run`, keeping state across reloads in `engine.state`, the gotchas of rerunning app code, and keeping reloads fast.

- [web-export](know-how/web-export.md) — `manifoldx export --web`: what reaches the browser, the transpiled subset and its errors, annotating, and replaying an export in Node against the Python app.

(More docs will land here as patterns crystallize. Empty bullet groups belong here, not in `know-how/`.)

## Sub-projects in flight

- **Web target** — `manifoldx export --web` (#140): plan 1 (merged) transpiles the per-frame Python, snapshots its state and replays it in Node; plan 2 (merged) writes the browser page (three.js scene, HTML GUI, browser input); plan 3 on `feat/web-target-3` adds `manifoldx run --web`, the dev loop in a tab. Know-how: `know-how/web-export.md`. Spec: `.knowledge/analysis/2026-11-06-web-target-design.html`. Plans: `.knowledge/plans/2026-11-06-web-target-plan-1-transpiler-and-replay.html`, `.knowledge/plans/2026-11-06-web-target-plan-2-browser-page.html`, `.knowledge/plans/2026-11-07-web-target-plan-3-run-web.html`.
- **Humanoid v1** — landed. `manifoldx.humanoid` (proportions, poses, solver, mannequin), `Engine.render_frame` with an id pass, `Camera.project`, single gamma encoding, material colours decoded from sRGB. First client: mosaico's scene renderer. Spec: `.knowledge/analysis/2026-10-04-characters-and-previz-design.md`. Plan: `.knowledge/plans/2026-10-05-humanoid-v1.md`. Next: inverse kinematics (`reach`, `look_at`) and animation.
- **Geometric modeling v1** — landed (batches 0-4). `manifoldx.modeling` subpackage: immutable numpy `Mesh` value type baking to the geometry dict via `to_geometry()` (single seam, no renderer/ECS changes). Primitives (`icosphere`/`box`/`plane`/`cylinder`/`torus`), seeded `noise` (perlin/fbm), deformers (`displace`/`twist`/`bend`/`taper`), sculpt brushes (`Falloff` + `draw`/`inflate`/`pinch`/`flatten`/`smooth`, cached one-ring `adjacency`), topology (`subdivide` midpoint+loop / `extrude` / `decimate` grid-clustering), and BSP-tree CSG (`union`/`difference`/`intersection`, pure-numpy, no vendored dep). Demos: `examples/deformers.py`, `topology.py`, `booleans.py`, `ffd.py`, `fields.py`, `color_by.py` + demo `demos/terrain_stream.py`. Deferred: QEM decimate, Manifold-style exact-predicate CSG, `ffd`/`bend_along_curve`. Design: `.knowledge/analysis/2026-07-28-geometric-modeling-v1-design.md` + `2026-07-28-modeling-booleans-v1-design.md`. Plans: `.knowledge/plans/2026-07-28-geometric-modeling-v1-plan-{1,2,3}-*.md`.
- **Sci-viz primitives v1** — `manifoldx.viz` subpackage. Plans 1–4 landed: foundation (PointCloud + ColormapMaterial + sprite path), text rendering (TextLabel + LabelMaterial + atlas), axes/scale-bar/legend (AxisFrame + screen-anchored labels), and the Altair-style declarative shim (`mxv.points/mesh/axes/legend/scale_bar/lights` + `Channel` wrappers + `+`-composed `Chart`). Plan 5 (visual regression infrastructure) is not yet specced. Plans live at `.knowledge/plans/2026-05-*-sci-viz-primitives-v1-*.md`.
- **Event-driven engine v1** — landed. `engine.emit(...)` / `@engine.on(...)` event bus with sync + async handlers. Replaces legacy `@engine.startup`/`@engine.shutdown`/`@engine.update`.
- **Input layer v1** — landed on top of the event bus. Keyboard, mouse, wheel, resize events + Bevy-style `engine.input` polling state.
- **Volume rendering v1** — landed. DVR via fragment-shader raymarching, `Volume` component + `VolumeMaterial` with reusable colormap LUT and per-material 256-sample opacity LUT.
- **Compute systems Phase 2** — landed. Python→WGSL transpiler so kernel authors write annotated Python instead of WGSL strings. Phase 3 (multi-pass / cross-kernel deps) not yet specced.
- **GUI v1** — landed. `manifoldx.gui` with widgets (`Panel`/`Text`/`ValueDisplay`/`Button`/`Slider`/`Toggle`), stack + flex layout, theme + named-class styling, `RectMaterial` (WGSL signed-distance rounded rect), `LabelTextureAtlas`-backed glyph path, `_GuiBridge` (pointer routing + hit-test + drag capture), `engine.gui.pointer_over_gui` cooperative-consume flag, and a new `gui` render pass at the end of the render order. Demo at `examples/gui.py`. Plans at `.knowledge/plans/2026-05-11-gui-v1-plan-*.md`. Design at `.knowledge/analysis/2026-05-10-gui-v1-design.md`.
- **Textured PBR v1** — landed. `StandardMaterial` accepts an optional `albedo_map: TextureHandle` and samples it in the fragment shader; `manifoldx.textures.load_texture(engine, path)` decodes images via Pillow and uploads them as `Rgba8UnormSrgb`. `manifoldx.assets.obj.load_obj` (re-exported as `manifoldx.load_obj`) parses Wavefront OBJ for all four face-line forms. First bundled binary assets in the repo: a UV-mapped Newell teapot, a procedural blue-and-white-china albedo, and NASA Blue Marble / LRO Earth + Moon textures for the new `examples/textures.py` and `demos/sun_earth_moon.py`. The fixes section of CHANGELOG also notes a pre-existing batching bug surfaced and fixed during this work (`mesh_batches` key now includes `mat_id`). Plan: `.knowledge/plans/2026-06-09-textured-pbr-v1.md`. Design: `.knowledge/analysis/2026-06-09-textured-pbr-v1-design.md`. Follow-ups: normal / metallic-roughness / AO maps; textured `BasicMaterial` (so the demo's sun can carry a real surface texture); mipmaps; IBL.
