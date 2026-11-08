# Crowds on the GPU

It is the night of the festival. Twenty thousand people stand in the plaza and
in the streets that open onto it, and the job is one line of physics: each
person keeps their own room, so every person has to look at every other one and
step away from the ones too close. Twenty thousand times twenty thousand is
four hundred million comparisons for a single frame, and numpy on the CPU no
longer gets through them before the next one is due.

A vectorised numpy pass manages it, but on one core, and the draw waits for
it. This job is the same small calculation in every thread, which is the
shape a graphics card is built for. ManifoldX takes it as a `Compute` class: the
body is annotated Python, and the engine compiles it to WGSL, the shader
language the GPU runs.

## The smallest kernel

The plaza is built the way chapter 1 built it: three hundred walkers scattered
across the square, the paving, and the basin in the middle.

```python {run}
import numpy as np

import manifoldx as mx
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.compute import Compute, ComputeShaderCompileError, ReadsWrites, Uniform
from manifoldx.compute.shader import u32, vec3
from manifoldx.resources import DirectionalLight, PhongMaterial, cube, plane
from PIL import Image

WALKERS = 300


class Velocity(Component):
    vector: vec3


engine = mx.Engine("stroll", width=480, height=270)
rng = np.random.default_rng(4)
pos = np.zeros((WALKERS, 3), dtype=np.float32)
pos[:, 0] = rng.uniform(-18.0, 18.0, WALKERS)
pos[:, 2] = rng.uniform(-12.0, 12.0, WALKERS)
pos[:, 1] = 0.85
vel = np.zeros((WALKERS, 3), dtype=np.float32)
vel[:, 0] = rng.uniform(1.0, 1.4, WALKERS)
height = rng.uniform(0.92, 1.08, WALKERS).astype(np.float32)
scale = np.stack([height, height, height], axis=1)
engine.spawn(
    Mesh(cube(0.38, 1.7, 0.38)),
    Material(PhongMaterial("#e2683c")),
    Transform(pos=pos, scale=scale),
    Velocity(vector=vel),
    n=WALKERS,
)
engine.spawn(
    Mesh(plane(40, 40)),
    Material(PhongMaterial("#b8ac96")),
    Transform(pos=(0, 0, 0), rot=Transform.rotation(x=-np.pi / 2)),
)
engine.spawn(
    Mesh(cube(4, 1, 4)),
    Material(PhongMaterial("#8a9bb0")),
    Transform(pos=(0, 0.5, 0)),
)
```

The new part is the kernel. A `Compute` subclass says which component arrays it
works on, and `main` is the body of one thread; `i` is the row this thread owns.
The `Velocity` component stores a velocity per person in a `vec3`, the shader's
three-float vector: inside a kernel it has arithmetic of its own, and it is the
same `vec3` you annotate a component field with, which is why a position and a
velocity can simply be added.

```python {run continue}
class StrollKernel(Compute):
    transforms: ReadsWrites[Transform]
    velocities: ReadsWrites[Velocity]

    dt: Uniform[float] = "frame_dt"
    n: Uniform[float] = WALKERS

    workgroup_size = 64
    dispatch = WALKERS

    def main(self, i: int) -> None:
        if i >= u32(self.n):
            return
        pos: vec3 = self.transforms[i].pos
        vel: vec3 = self.velocities[i].vector
        self.transforms[i].pos = pos + vel * self.dt


wgsl = StrollKernel().compile()
print("\n".join(wgsl.splitlines()[:4]))

engine.compute(StrollKernel)
engine.camera.set_pose(position=(12, 12, 22), target=(0, 0.8, 0))
engine.set_sun(DirectionalLight(color="#fff2dd", intensity=3.0, direction=(-0.35, -1, -0.4)))
Image.fromarray(engine.render_frame(supersample=1))
```

The annotations above `main` are the wiring. The line
`transforms: ReadsWrites[Transform]` says the kernel reads and writes the
`Transform` component, and the engine binds that component's array — the same
array the renderer draws from — as a storage buffer. `velocities` is the same for the
`Velocity` component. `dt: Uniform[float] = "frame_dt"` is a scalar the engine
refills every frame; `n` is a `Uniform` too, but it holds a constant. A binding
the kernel only reads is written `Reads` instead.

`engine.compute` draws nothing. It hands the class to the engine's
`ComputeRunner`, which already checks the shader text, builds the pipeline the
first time a frame needs it, and dispatches it once per frame — after the CPU
systems and before the draw, so what you see is the state the kernel wrote.
`render_frame` runs one such frame.

Two numbers set how the work is launched. `workgroup_size = 64` groups the
threads into blocks of 64, the size drivers prefer. `dispatch = WALKERS` says
how many threads to launch; the symbols `"entity_count"` and `"max_entities"`
are also accepted. This world also holds paving and the fountain, so the kernel
names the walkers' own count instead, and `n` carries the same number into the
shader. The count rounds up to a whole number of blocks, so the last block is
partly empty; `if i >= u32(self.n)` sends those extra threads home before they
touch a row. And the walkers were spawned first, so they are rows 0 to 299 of
both arrays, with the paving and the basin after them.

`compile` is what turns the class into WGSL, and `transpile_compute` is the
walker behind it. It keeps a `TypeEnv`, a table of the type of every name in
scope — an `f32` here, a `vec3` there, the `u32` of a loop counter — and spells
each operation from it. The print above shows what the bindings became:

```text
struct Uniforms { dt: f32, n: f32, };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> transforms: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocities: array<f32>;
```

One uniform block holds every `Uniform` the class declares, and the components
follow as flat arrays of floats, which is exactly how the ECS keeps them. A
`Transform` row is ten floats — position, rotation, scale — so the shader reads
`transforms[i * 10u + 0u]` and its neighbours for the position, and writes a new
`vec3` back as three separate stores.

The transpiler refuses what WGSL would refuse, at the moment you register the
kernel and before any frame is drawn. An integer and a float do not mix in WGSL,
so `1 + self.dt` is an error rather than a guess:

```python {run}
from manifoldx.components import Transform
from manifoldx.compute import Compute, ComputeShaderCompileError, ReadsWrites, Uniform
from manifoldx.compute.shader import vec3


class Broken(Compute):
    transforms: ReadsWrites[Transform]
    dt: Uniform[float] = "frame_dt"

    def main(self, i: int) -> None:
        step: float = 1 + self.dt
        self.transforms[i].pos = self.transforms[i].pos + vec3(step, 0.0, 0.0)


try:
    Broken().compile()
except ComputeShaderCompileError as err:
    print(err.category, "-", err.message)
```

`ComputeShaderCompileError` carries the category of the mistake and a message
that says what to do about it; at `engine.compute` the same error stops the
program before the first frame, so a typo never becomes a black one.

The frame from the stroll is one thirtieth of a second after the spawn, and
each walker has moved the three to five centimetres its velocity asks for.
Nothing in the picture came from Python: the positions on screen are what the
shader wrote into the array.

## The crowd

The stroll kernel had one thread and nothing to look at. The festival needs the
loop: every person against every other. The velocity is still three numbers, but
a person also has a personal space — the room they insist on — so the component
is a `vec4`, and the fourth component is not padding. `vec4` is the shader's
four-float vector: `.x`, `.y`, `.z`, `.w` read its parts, and the transpiler
writes it back as four stores.

The crowd is built first, in ordinary numpy: a wobbly disc of points, none
inside the basin, each person with a reach between 0.45 and 0.7 metres, tall or
short, in one of four colours. The renderer batches by material, so the four
colours cost four instanced draws rather than twenty thousand. The crowd is
spawned before the paving, so its rows are 0 to 19 999 and the three scenery
entities sit above them, where the kernel never looks.

```python {run export=examples/gpu_crowd.py}
"""Festival crowd: twenty thousand people steering away from each other on the GPU."""

import numpy as np

import manifoldx as mx
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.compute import Compute, ReadsWrites, Uniform
from manifoldx.compute.shader import dot, length, sqrt, u32, vec3, vec4
from manifoldx.resources import DirectionalLight, PhongMaterial, cube, plane
from PIL import Image

CROWD = 20_000


class Velocity(Component):
    """Velocity in xyz; in w, the room this person keeps around them."""

    motion: vec4


engine = mx.Engine("Festival crowd", width=480, height=270)

rng = np.random.default_rng(11)
angle = rng.uniform(0, 2 * np.pi, CROWD)
edge = 20.0 * (1.0 + 0.12 * np.sin(3 * angle) + 0.07 * np.sin(7 * angle))
radius = np.sqrt(rng.uniform(3.0**2, edge**2, CROWD))
pos = np.zeros((CROWD, 3), dtype=np.float32)
pos[:, 0] = np.cos(angle) * radius
pos[:, 2] = np.sin(angle) * radius
pos[:, 1] = 0.85
motion = np.zeros((CROWD, 4), dtype=np.float32)
motion[:, 3] = rng.uniform(0.45, 0.7, CROWD)
height = rng.uniform(0.92, 1.08, CROWD).astype(np.float32)
scale = np.stack([height, height, height], axis=1)

colours = ["#e2683c", "#3c7de2", "#d9b64a", "#3fb37a"]
share = CROWD // len(colours)
watched = []
for k, colour in enumerate(colours):
    lo = k * share
    hi = CROWD if k == len(colours) - 1 else (k + 1) * share
    watched.append(
        engine.spawn(
            Mesh(cube(0.38, 1.7, 0.38)),
            Material(PhongMaterial(colour)),
            Transform(pos=pos[lo:hi], scale=scale[lo:hi]),
            Velocity(motion=motion[lo:hi]),
            n=hi - lo,
        )
    )

engine.spawn(
    Mesh(plane(64, 64)),
    Material(PhongMaterial("#757d68")),
    Transform(pos=(0, -0.01, 0), rot=Transform.rotation(x=-np.pi / 2)),
)
engine.spawn(
    Mesh(plane(40, 40)),
    Material(PhongMaterial("#b8ac96")),
    Transform(pos=(0, 0, 0), rot=Transform.rotation(x=-np.pi / 2)),
)
engine.spawn(
    Mesh(cube(4, 1, 4)),
    Material(PhongMaterial("#8a9bb0")),
    Transform(pos=(0, 0.5, 0)),
)


```

The kernel is where the job lives.

```python {run continue export=examples/gpu_crowd.py}
class CrowdKernel(Compute):
    """Move every person away from their neighbours, one thread per person."""

    transforms: ReadsWrites[Transform]
    people: ReadsWrites[Velocity]

    edge: Uniform[float] = 31.0
    fountain: Uniform[float] = 3.0
    strength: Uniform[float] = 8.0
    max_speed: Uniform[float] = 1.4
    dt: Uniform[float] = "frame_dt"
    n: Uniform[float] = CROWD

    workgroup_size = 64
    dispatch = CROWD

    def main(self, i: int) -> None:
        if i >= u32(self.n):
            return
        pos: vec3 = self.transforms[i].pos
        motion: vec4 = self.people[i].motion
        reach: float = motion.w

        push: vec3 = vec3(0.0, 0.0, 0.0)
        for j in range(u32(self.n)):
            if i == j:
                continue
            away: vec3 = pos - self.transforms[j].pos
            d2: float = dot(away, away)
            if d2 >= reach * reach:
                continue
            if d2 < 1e-6:
                continue
            d: float = sqrt(d2)
            push += away / d * (1.0 - d / reach)

        flat: vec3 = vec3(pos.x, 0.0, pos.z)
        rad: float = length(flat)
        wall: vec3 = vec3(0.0, 0.0, 0.0)
        if rad > self.edge:
            wall = flat / rad * ((self.edge - rad) * 4.0)

        vel: vec3 = vec3(motion.x, motion.y, motion.z)
        vel = vel + (push * self.strength + wall) * self.dt
        speed: float = length(vel)
        if speed > self.max_speed:
            vel = vel / speed * self.max_speed

        self.people[i].motion = vec4(vel.x, vel.y, vel.z, motion.w)
        next: vec3 = pos + vel * self.dt
        nflat: vec3 = vec3(next.x, 0.0, next.z)
        nrad: float = length(nflat)
        if nrad < self.fountain and nrad > 0.001:
            next = vec3(nflat.x / nrad * self.fountain, next.y, nflat.z / nrad * self.fountain)
        self.transforms[i].pos = next


engine.compute(CrowdKernel)
engine.camera.set_pose(position=(0, 42, 18), target=(0, 0.6, 0))
engine.set_sun(DirectionalLight(color="#fff2dd", intensity=3.0, direction=(-0.35, -1, -0.4)))

before = [w.transform.pos.copy() for w in watched]
frame = engine.render_frame(supersample=1)
moved = [np.linalg.norm(w.transform.pos - p) for w, p in zip(watched, before)]
print("four people moved, in metres:", np.round(moved, 3))
Image.fromarray(frame)
```

The inner loop is the whole job. `dot`, `sqrt` and `length` are shader
builtins, and the transpiler maps them straight to WGSL. `away` points from the
neighbour to us and `d2` is its squared length: squared, because a square root
only matters once a neighbour is close enough to matter, and `continue` skips
the rest. For each
neighbour inside `reach` we add a push along `away` of size `1 - d / reach`, so
someone just across the line pushes a little and someone on top of us pushes a
lot. `for j in range(u32(self.n))` is the one loop form the transpiler accepts,
and `u32(...)` converts the uniform's `f32` into the index type the array needs.
Every thread runs its own copy of this loop against the same buffer: twenty
thousand threads reading twenty thousand rows each, which is the four hundred
million of the opening.

The rest is the square. Beyond 31 metres a soft wall turns the crowd back onto
the paving — twenty thousand people at festival density need more room than the
square has, and the streets around it take the rest — and at the basin the new
position is clamped to the rim, so nobody wades into the fountain.

The print is the kernel's receipt: four people whose handles the spawn returned,
and how far each moved in the first step, between three and seventeen
millimetres. That is the mirror at work — `ComputeRunner` uploads the rows to the
GPU and copies the written rows back into the same numpy array before the draw,
so a kernel's result is visible both on screen and to Python.

The frame is the crowd one step into the festival. From above, it fills the
square and spills into the streets around it, and the basin is the hole in the
middle. At four people to the square metre the
square holds about six thousand; the other fourteen thousand stand in the roads.
That is geometry, not a limit of the kernel.

That is the finished program, and it is in
[`docs/examples/`](https://github.com/gia-uh/manifoldx/tree/main/docs/examples)
like the other chapters'. To watch the crowd in a window instead of a still,
install the `desktop` extra and end the program with `engine.run()`.

```python
engine.run()
```

One honest note about the machine. When no graphics card is present, ManifoldX
renders on a software Vulkan driver, lavapipe, and these pages were built that
way. There the shader runs on the CPU like everything else, and a frame of the
crowd takes about a second; on a real card the same four hundred million pairs
would be a small fraction of that, but the number here would not show it, so the
guide does not print one. What the frame does show is that the work left Python:
the loop is a shader, the arrays are the same ones the ECS draws, and nothing in
the hot path is a Python `for`.

The next chapter loads the plaza's models from disk and hands the scene to other
programs.
