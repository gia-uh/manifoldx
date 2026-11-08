# ManifoldX

[![PyPI version](https://img.shields.io/pypi/v/manifold-gfx)](https://pypi.org/project/manifold-gfx/)
[![Python versions](https://img.shields.io/pypi/pyversions/manifold-gfx)](https://pypi.org/project/manifold-gfx/)
[![License](https://img.shields.io/pypi/l/manifold-gfx)](LICENSE)
[![Tests](https://github.com/apiad/manifoldx/actions/workflows/test.yml/badge.svg)](https://github.com/apiad/manifoldx/actions/workflows/test.yml)

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

## Quick Start

```python
import manifoldx as mx
import numpy as np

from manifoldx.components import Transform, Mesh, Material
from manifoldx.resources import StandardMaterial, PointLight, cube, sphere

# Create engine with default settings
engine = mx.Engine("My First Scene")

# Create a cube and sphere
cube_geo = cube(1, 1, 1)
sphere_geo = sphere(0.7, 32)

# Create PBR materials (roughness: 0-1, metallic: 0-1)
red_shiny = StandardMaterial(color="#ff3333", roughness=0.15, metallic=0.9)
blue_dull = StandardMaterial(color="#3366ff", roughness=0.8, metallic=0.0)

# Spawn entities
engine.spawn(
    Mesh(cube_geo),
    Material(red_shiny),
    Transform(pos=(-1.5, 0, 0)),
)

engine.spawn(
    Mesh(sphere_geo),
    Material(blue_dull),
    Transform(pos=(1.5, 0, 0)),
)

# Add an orbiting light
light = PointLight(color="#ffffff", intensity=15.0, position=(5, 5, 5))
engine.set_lights([light])

# Animate
@engine.system
def animate_lights(query: mx.Query[Transform], dt: float):
    t = engine.elapsed
    light.position = (
        5 * np.cos(t * 0.7),
        3 + np.sin(t * 0.5) * 2,
        5 * np.sin(t * 0.7),
    )

# Auto-fit camera to view the scene
engine.camera.fit(radius=5.0, azimuth=30, elevation=35)

# CLI handles run vs render automatically
engine.cli()
```

Save as `my_scene.py` and run:

```bash
python my_scene.py              # Interactive window
python my_scene.py --render     # Render 60s video → my_scene.mp4
python my_scene.py --render --fps 60 --duration 120 --output movie.mp4
```

## Built to work with an agent

`manifoldx run` reloads your app in the same window on every save, and serves a dev page and an HTTP API beside it, on 127.0.0.1. An agent (or you) can see the frame on screen, read the log the app writes with `engine.log`, read and patch `engine.state`, fire the app's events, and wait for a log line, while a person plays:

```bash
uv run manifoldx run --debug -m mygame.dev   # prints: dev page and API at http://127.0.0.1:8770
manifoldx look                          # .manifoldx/look.png: the frame on screen
manifoldx log --match "entered cave" --wait 300
manifoldx emit spawn '{"kind": "slime"}'
manifoldx metrics -n 30                 # --debug: fps, frame times, memory, gc per second
manifoldx set '{"player": {"hp": 1}}'
```

So you build a game or a simulation interactively: the agent changes the code, sees the result in the running window, injects situations to test it through the app's own events, and follows what the person does. `know-how/agent-loop.md` and `know-how/hot-reload.md` describe both.

## N-Body Simulation

A pure-numpy gravitational simulation running 500 bodies in real-time at a single draw call (instanced rendering).

**Physics:** All pairwise forces are computed with a single vectorized numpy expression — no Python loops in the hot path. For N bodies this means N² = 250,000 force computations per frame, each a 3-component vector.

```python
@engine.system
def nbody_physics(query: mx.Query[Transform], dt: float):
    global velocities
    pos = query[Transform].pos.data

    # All-pairs position differences (N, N, 3) — one numpy broadcast
    diff = pos[None, :] - pos[:, None]

    # Pairwise distances (N, N)
    dist = np.linalg.norm(diff, axis=2)
    dist = np.maximum(dist, SOFTENING)

    # Gravitational force magnitude for every pair
    force_mag = G * (masses[None, :] * masses[:, None]) / dist**2

    # Net force on each body: sum over all other bodies
    direction = diff / dist[:, :, None]
    forces = force_mag[:, :, None] * direction
    net_force = forces.sum(axis=1)

    # Integrate: F = ma → a = F/m
    velocities += (net_force / masses[:, None]) * dt
    query[Transform].pos += velocities * dt
```

> See `examples/nbody.py` for the full implementation with velocity damping and speed clamping.

### Ideal Gas Simulation

The `examples/gas.py` demo shows the other side: **no gravity, all collisions**. 500 particles bounce inside an invisible box with elastic collisions and wall reflections — a kinetic theory simulation in pure numpy.

**Collisions** find overlapping pairs with a vectorized comparison, filter with `np.where(np.triu(...))`, then resolve impulse with `np.add.at` for safe accumulation. **Wall reflections** are a single vectorized mask: `velocities[next_pos < wall] = np.abs(...)`.

> See `examples/gas.py` for the full implementation.

### Boids Flocking Simulation

The `examples/boids.py` demo shows **emergent behavior** from simple local rules: 300 boids with **separation**, **alignment**, **cohesion**, plus 4 wandering **predators** they flee from.

**Flocking rules** (all vectorized):
- **Separation** — boids repel neighbors weighted by 1/dist²
- **Alignment** — match average velocity of nearby boids
- **Cohesion** — steer toward center of mass of neighbors

**Predator avoidance** — boids detect predators within 10 units and flee with 20x the force of any flocking rule. Fleeing boids get a speed boost (15 vs 10).

**Spatial optimization** — uses squared distances to avoid `sqrt` in the hot path. All three rules computed via masked tensor sums over axis=1 of an `(N, N, 3)` difference tensor.

> See `examples/boids.py` for the full implementation.

### Three Demos, Three Vectorization Patterns

| Demo       | Entities | Physics Pattern                   | Operations/Frame               |
| ---------- | -------- | --------------------------------- | ------------------------------ |
| `nbody.py` | 500      | All-pairs gravity                 | N² = 250,000 force pairs       |
| `gas.py`   | 500      | Pair collisions + walls           | O(N²) pair checks + wall masks |
| `boids.py` | 300 + 4  | Neighbor flocking + predator-flee | (N,N,3) tensor sums + (N,P,3)  |

All three use **pure numpy** — zero Python loops in the hot path. The ECS overhead is ~microseconds/frame; the bottleneck is GPU fill-rate, not CPU physics.

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

## Features

### ECS Architecture
- **Structure of Arrays (SoA)** layout for each component
- Vectorized numpy operations for batch transforms
- Free-list for efficient entity reuse
- Component view with operator overloads (`+=`, `*=`, etc.)

### Rendering
- **Instanced drawing** — single draw call per (geometry, material) batch
- **Material-specific pipelines** — each material type compiles its own WGSL shader
- **Transform caching** — dirty-flag optimization to avoid recomputing matrices
- **Shared transform buffer** — all instance transforms uploaded once per frame

### Materials & Lighting
- **BasicMaterial** — unlit flat color with simple diffuse
- **StandardMaterial** — full PBR with GGX BRDF
  - Roughness/metallic workflow
  - Multiple point lights with inverse-square attenuation
  - Reinhard tonemapping + gamma correction
- **ColormapMaterial** — sprite path, 1D LUT lookup on a per-instance scalar; six built-in colormaps (viridis, magma, plasma, inferno, turbo, gray); optional Lambert shading
- **LabelMaterial** — camera-facing billboard text rasterized through a shared `LabelTextureAtlas`; world- or screen-anchored
- **AxisMaterial** — line-list material for axes; world- or screen-anchored
- **VolumeMaterial** — direct volume rendering with a colormap LUT + 256-sample opacity LUT
- **External lights** — passed to engine like camera (not in ECS)

### Camera
- Perspective projection (WebGPU NDC)
- Spherical coordinate orbit controls
- Fit/fit_bounds for automatic framing

### Geometries
- Cube, UV Sphere, Plane (with normals)
- `SPRITE_QUAD` for the camera-facing sprite path (sci-viz point clouds)
- `axis_line_x` / `axis_line_y` / `axis_line_z` for the line-list axis pipeline

### Sci-viz primitives (`manifoldx.viz`)
- **PointCloud + ColormapMaterial** — large-N particle clouds, per-particle scalar attribute mapped through a colormap LUT, per-particle world-space radius
- **TextLabel + LabelMaterial** — camera-facing billboard text via a PIL-rasterized `LabelTextureAtlas`
- **AxisFrame + AxisMaterial** — world- or screen-anchored axes; screen anchoring gives scale bars without a dedicated component
- **Colormap legends** — `LabelTextureAtlas.register_colormap_legend(...)` rasterizes any colormap into the atlas, rendered through the screen-anchored label path

### Event-driven engine
- Pub/sub event bus parallel to the frame loop: `engine.emit(name, payload)` and `@engine.on(name)`
- Sync handlers run inline at the head of the next frame; async handlers progress between draw callbacks
- Async waiters: `await engine.tick()`, `await engine.delay(seconds)`, `await engine.elapsed_at(target)`, plus `await engine.run_blocking(fn, ...)` for blocking work
- Lifecycle decorators (`startup`, `shutdown`, `frame`) ride the same bus

### Input layer
- Typed event dataclasses on the bus: `KeyEvent`, `PointerEvent`, `WheelEvent`, `ResizeEvent`
- Bevy-style polling state at `engine.input`: `is_pressed(key)` / `just_pressed(key)` / `just_released(key)`, plus `mouse_pos`, `mouse_delta`, `wheel_delta`, `viewport_size`
- Polling state is updated immediately on every event; `just_*` and delta accumulators are this-frame-bound
- Key names: letters are lowercase whatever Shift says (Shift is in `modifiers`), the space bar is `"Space"`
- `engine.input.capture_pointer(True)` hides and locks the cursor for mouse look (glfw windows); `mouse_delta` keeps counting

### GPU compute systems
- Author kernels as a typed-Python `def main(self, i)` body — no inline WGSL strings required
- Python → WGSL transpiler emits the bind-group header, helper functions, and the `@compute` wrapper
- `engine.compute(cls)` validates synchronously: shader-compile errors and Python-source-level transpiler errors surface at registration time
- Mypy-clean kernel bodies: vec3/vec4 are real classes with arithmetic dunders; math builtins carry PEP-484 signatures

### Volume rendering
- `Volume` ECS component (handle into a per-engine `VolumeRegistry`); `engine.register_volume(numpy_array)` for CPU upload
- Front-to-back composite raymarching in a fragment shader, with depth-test on / depth-write off and pre-multiplied alpha blending
- Reusable colormap LUT + per-material 256-sample opacity LUT (built from piecewise stops or a numpy array)

### Path tracing (`manifoldx.trace`)
- A progressive path tracer for any scene written to glTF: `Scene.from_glb(path, overrides)`, `Tracer(scene, w, h)`, `tracer.step(n)`, `tracer.image()`
- WGSL compute kernels through wgpu-py (no hardware ray tracing); the BVH is built in numpy with binned SAH
- Lambert + GGX, thin and solid glass, emissive surfaces, base-colour textures and vertex colours; a Preetham sky with a sun disk, HDR maps, punctual lights; next-event estimation with MIS; an edge-aware à-trous denoiser
- `manifoldx trace scene.glb -o out.png --size 1024x640 --spp 64`; on an RTX 3060 a 4k-triangle interior comes out clean in under a second

### Hot reload

`manifoldx run app.py` (or `manifoldx run -m package.module`) runs an app and reruns it in the same window each time one of its source files changes. The scene is rebuilt from the new code. Keep in `engine.state` whatever should survive a reload, as plain data:

```python
engine.state.setdefault("angle", 0.0)

@engine.system
def spin(query: Query[Transform], dt: float):
    engine.state.angle += dt
    query[Transform].rot = Transform.rotation(x=0, y=engine.state.angle, z=0)
```

A broken edit leaves the running version on screen; an error while running pauses the app until the next save.

What reloads, what to keep in `engine.state`, the gotchas and how to keep reloads fast: [know-how/hot-reload.md](know-how/hot-reload.md).

### Video Rendering

Render simulations to video files for sharing or CI using the built-in CLI:

```bash
python examples/nbody.py --render
python examples/nbody.py --render --fps 60 --duration 120 --output movie.mp4
```

**Options:**
- `--render` — Render to video instead of showing window
- `--fps N` — Frames per second (default: 30)
- `--duration N` — Video duration in seconds (default: 60)
- `--output FILE` — Output filename (default: `<script>.mp4`)
- `--quality Q` — Quality: low, medium, high (default: high)

**Programmatic API:**
```python
engine.render(
    output="simulation.mp4",
    fps=30,
    duration=60,
    quality="high",
)
```

**Note:** Video rendering requires the `offline` extra: `pip install manifold-gfx[offline]`

## Architecture Highlights

The ECS uses numpy arrays for all component data. When you call `query[Transform].pos += velocity * dt`, it's a single vectorized numpy operation spanning thousands of entities.

**Real-world examples:** The N-body demo (`examples/nbody.py`) simulates 500 bodies with 250,000 pairwise gravitational force computations per frame. The ideal gas demo (`examples/gas.py`) runs 500 particles with elastic collisions and wall reflections. Both are pure numpy with zero Python loops.

## Limitations (Known)

- ❌ No shadows
- ❌ No general texture support (label atlas is the only `texture_2d_array` path)
- ❌ No environment/IBL mapping
- ❌ Single material params per draw call (not per-instance)
- ❌ Only point lights in PBR shader
- ❌ Limited to ~100k entities

## Future Ideas

This is an academic/experimental project. Ideas for future development:

1. **Per-instance material data** — Storage buffer for varying roughness/metallic per instance in a single draw
2. **Shadow mapping** — Shadow pass + PCF sampling
3. **Texture maps** — Diffuse, normal, roughness textures via storage buffers
4. **Spot/Directional lights** — Extend PBR shader
5. **Environment mapping** — IBL with prefiltered radiance
6. **Skinned animation** — Bone transforms in vertex shader
7. **Post-processing** — Bloom, DOF, TAA
8. **Deferred rendering** — Forward+ / clustered lighting for many lights

## Contributing

Contributions welcome! This is an educational project — all skill levels encouraged.

**Areas needing work:**
- Bug fixes and stability improvements
- Additional geometry types (torus, cylinder, etc.)
- More material types (toon, unlit with texture)
- Shadow implementation
- Performance profiling and optimization

**Getting started:**

```bash
# Clone and set up
git clone https://github.com/apiad/manifoldx.git
cd manifoldx
pip install -e ".[dev]"

# Run tests
make test

# Run an example
python -m examples.cube
```

## Testing

```bash
# Run all tests (uses offline backend for headless CI)
make test

# Run specific test file
python -m pytest tests/test_ecs.py -v
```

Current test coverage: **457 tests** covering ECS operations, components, materials, rendering passes (mesh, sprite, label, axis, volume), camera, sci-viz primitives, the event bus, input layer, GPU compute kernels, the WGSL transpiler, and video output.

## License

MIT License — See LICENSE file.

## Credits

- [wgpu](https://github.com/gfx-rs/wgpu) — Pure Python WebGPU bindings
- [PyGfx](https://github.com/pygfx/pygfx) — Reference for WGSL shader patterns
- [rendercanvas](https://github.com/pygfx/rendercanvas) — Window management
