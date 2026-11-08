# ManifoldX

[![PyPI version](https://img.shields.io/pypi/v/manifold-gfx)](https://pypi.org/project/manifold-gfx/)
[![Python versions](https://img.shields.io/pypi/pyversions/manifold-gfx)](https://pypi.org/project/manifold-gfx/)
[![License](https://img.shields.io/pypi/l/manifold-gfx)](LICENSE)
[![Tests](https://github.com/gia-uh/manifoldx/actions/workflows/test.yml/badge.svg)](https://github.com/gia-uh/manifoldx/actions/workflows/test.yml)

A real-time 3D rendering engine built on pure [wgpu](https://github.com/gfx-rs/wgpu) with an Entity Component System (ECS) architecture. Written in Python with numpy for high-performance data handling.

> ⚠️ **Beta / Academic Project** — This is an experimental proof-of-concept exploring the extent to which Python can be used for high-performance graphics via wgpu. Not recommended for production use. Expect bugs, breaking changes, and missing features.

## Motivation

Domain researchers need to run large-scale simulations (10⁴-10⁶ entities) and visualize results in 3D. **Currently they can't do both in pure Python.**

| Tool                       | Simulation | Visualization          | Language      |
| -------------------------- | ---------- | ---------------------- | ------------- |
| Matplotlib/Plotly          | ✅          | 2D only, slow at scale | Pure Python   |
| VisPy                      | ❌          | OpenGL, steep curve    | Python + GLSL |
| PyGfx                      | ❌          | Excellent rendering    | Pure Python   |
| Game engines (PyGame)      | ✅          | 2D mostly              | Pure Python   |
| Specialized (OpenMM, SUMO) | ✅          | Data export only       | Fortran/C++   |

**The gap:** No Python tool combines data-driven simulation with accessible 3D visualization in one package.

### The Vision

ManifoldX gives researchers both:

1. **Simulation in pure NumPy** — vectorized operations over entity arrays, no Python loops in the hot path
2. **3D visualization without graphics knowledge** — spawn entities with meshes/materials, the engine handles GPU rendering

```python
# Write physics in pure numpy (data-driven, not OOP)
@engine.system
def nbody_physics(query, dt):
    # Magic happens here...
    forces = compute_gravity_all_pairs(query[Transform].pos.data)
    velocities += forces * dt
    query[Transform].pos += velocities * dt

# Engine handles GPU buffers, WGSL shaders, instanced draw calls
# You get 3D visualization of your simulation instantly
```

**Target domains:** Astrophysics (galaxy formation), molecular dynamics (protein folding), epidemiology (disease spread), crowd science (evacuation flows), traffic engineering (vehicle flow), weather (particle advection).

**Works in:** Jupyter notebooks, Quarto documents, Streamlit dashboards, standalone scripts.

> ⚠️ **Spoiler:** Python is surprisingly capable at real-time 3D. The N-body demo runs 500 bodies with N² gravity — 250,000 force pairs per frame in pure numpy.

### Why Not PyGfx?

[PyGfx](https://github.com/pygfx/pygfx) is an excellent rendering engine and has been a key source of inspiration — we've learned a lot from its WGSL patterns, material system, and API design. However, it's fundamentally a **scene graph** engine (object-oriented hierarchy of transforms), not a data-driven ECS. For large-scale simulations with vectorized physics, the OOP overhead of traversing a scene graph becomes a bottleneck. ManifoldX uses a flat SoA data layout where simulation logic operates directly on numpy arrays.

**Technically this is a game engine** — the ECS + instanced rendering + PBR pipeline is exactly what you'd use for a game. But that's not the focus. The goal is making 3D visualization accessible to researchers who don't know anything about graphics.

## Installation

```bash
pip install manifold-gfx           # Base (no rendering backends)
pip install manifold-gfx[desktop]  # + GLFW for desktop window
pip install manifold-gfx[offline]  # + imageio-ffmpeg for video
pip install manifold-gfx[all]      # Everything

# or with uv
uv add manifold-gfx
uv add manifold-gfx[desktop]       # Desktop rendering
uv add manifold-gfx[offline]      # Video rendering
```

**Requirements:**
- Python 3.13+
- GPU with WebGPU support (via wgpu backend)
  - Vulkan on Linux
  - Metal on macOS
  - D3D12 on Windows

**Note:** Install with `[desktop]` for interactive rendering, `[offline]` for video output, or `[all]` for both.

## The guide

The documentation is a book in [`docs/`](docs/) that follows one project, a
small town plaza, from a first frame to a finished scene. Every code block in
it runs against the library when the guide is built, so what it says is what
this version does. Each chapter's finished program is in
[`docs/examples/`](docs/examples/).

1. [A first frame](docs/chapters/01-a-first-frame.md): the engine, a camera, something to look at.
2. [People who walk](docs/chapters/02-people-who-walk.md): components, queries and systems.
3. [Reacting to the visitor](docs/chapters/03-reacting-to-the-visitor.md): events, input and commands.
4. [Light and surface](docs/chapters/04-light-and-surface.md): materials, lights and shadows.
5. [Sky, water and textures](docs/chapters/05-sky-water-and-textures.md): textures, water, atmosphere and image-based lighting.
6. [Modelling the buildings](docs/chapters/06-modelling-the-buildings.md): meshes built in code: booleans, deformers, fields.
7. [Bodies and poses](docs/chapters/07-bodies-and-poses.md): humanoid figures, skinning and the id pass.
8. [Measuring the plaza](docs/chapters/08-measuring-the-plaza.md): point clouds, labels, axes, charts and volumes.
9. [A control panel](docs/chapters/09-a-control-panel.md): widgets inside the window.
10. [Crowds on the GPU](docs/chapters/10-crowds-on-the-gpu.md): compute kernels written in Python.
11. [Assets in and out](docs/chapters/11-assets-in-and-out.md): OBJ, glTF and streaming.
12. [A still worth printing](docs/chapters/12-a-still-worth-printing.md): the path tracer.
13. [Working with an agent](docs/chapters/13-working-with-an-agent.md): hot reload, the dev channel and the web page.
14. [The plaza, end to end](docs/chapters/14-the-plaza-end-to-end.md): everything together.

Build it locally with `make docs` (the site lands in `docs/site/`).

## Examples and demos

Two kinds of runnable code: **`examples/`** are focused — one feature, minimal
code (documentation); **`demos/`** are show-off — game-like / sim-like
experiences combining many systems. See each folder's `README.md`.

### `examples/` — focused feature examples

| Example                  | Description                                                                |
| ------------------------ | -------------------------------------------------------------------------- |
| `hello_world.py`         | Minimal empty window                                                       |
| `cube.py`                | Rotating cube with Phong material                                          |
| `pbr.py`                 | 3×2 grid demonstrating PBR materials + 3 orbiting lights                   |
| `shadows.py` / `spotlight.py` | Directional-sun and spot-light shadow mapping                         |
| `ibl.py`                 | Image-based lighting (split-sum environment reflection)                    |
| `textures.py`            | Textured PBR: a UV-mapped teapot with an albedo map                        |
| `spheres.py`             | Many spheres with physics-like behavior                                    |
| `nbody.py` / `gas.py` / `boids.py` | Pure-numpy gravity / ideal-gas / flocking simulations            |
| `nbody_compute.py` / `gas_compute.py` / `point_cloud_compute.py` | Same, ported to GPU compute kernels       |
| `scatter_plot.py` / `point_cloud.py` / `axes.py` | Sci-viz `manifoldx.viz` primitives                        |
| `volume.py`              | Direct volume rendering of a 64³ Gaussian blob                            |
| `gui.py`                 | Immediate-ish GUI widgets (panels, sliders, toggles)                       |
| `fields.py` / `color_by.py` | Composable field algebra + per-vertex coloring (procedural terrain)     |
| `deformers.py` / `sculpt`* / `topology.py` / `booleans.py` / `ffd.py` | `manifoldx.modeling` operators       |
| `event_dolly.py` / `event_pulse.py` | Event bus (async dolly, sync spawn-on-event)                    |
| `input_orbit.py` / `input_fly.py` | Drag-to-orbit and WASD fly-cam via the input layer                |

### `demos/` — end-to-end showcases

| Demo | Description |
| --- | --- |
| `terrain_stream.py` | Infinite procedural-terrain flyby with shader fog; patches generated in a worker process and streamed into a recycled slot pool |
| `labyrinth_fps.py`  | First-person maze crawl with a shadow-casting flashlight |
| `sun_earth_moon.py` | Textured Earth + Moon orbiting a sun (bundled NASA imagery) |
| `smoke.py`          | Volumetric Perlin-FBM smoke with camera motion |

Run one:

```bash
python examples/nbody.py                 # interactive window
python demos/terrain_stream.py --render --duration 8 --output flyby.mp4
```

## Contributing

Contributions are welcome. Open an issue first for anything beyond a small fix;
the roadmap lives in the pinned vision issue. Agents working in this repository
start from [`AGENTS.md`](AGENTS.md).

```bash
git clone https://github.com/gia-uh/manifoldx.git
cd manifoldx
uv sync --extra offline --extra viz --extra svg --dev --group docs
make test        # the suite; render tests need a GPU adapter (lavapipe works)
make docs-check  # the guide: every block runs, examples match, every class covered
```

## License

MIT License — See LICENSE file.

## Credits

- [wgpu](https://github.com/gfx-rs/wgpu) — Pure Python WebGPU bindings
- [PyGfx](https://github.com/pygfx/pygfx) — Reference for WGSL shader patterns
- [rendercanvas](https://github.com/pygfx/rendercanvas) — Window management
