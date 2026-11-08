# People who walk

Two hundred people are crossing the plaza. Each is somewhere between the west
gate and the fountain, heading for the east arcade at their own pace, and every
frame each one has to take a step without walking through a neighbour. At twenty
frames a second there is no time for a Python loop per person.

## A person is a row

A thing in the world is an *entity*: a number, a row in the store. What the
entity knows lives in *components*, and a component is a fixed number of floats
per entity. `Component` is the base class that turns a class body of annotations
into one. `Float` takes a column, `Vector3` takes three, `Vector4` takes four.
Declaring the fields is the whole definition: a subclass reads its annotations
when the class is created and lays the storage out in that order.

```python {run export=examples/walkers.py}
import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Colors, Component, Layer, Material, Mesh, Transform
from manifoldx.physics import all_pairs
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane, sphere
from manifoldx.types import Color, Float, Vector3

N = 200
FLOOR = (-0.70710678, 0.0, 0.0, 0.70710678)  # a quarter turn about x: lay the plane flat
COATS = [Colors.CYAN, Colors.YELLOW, Colors.MAGENTA, Colors.WHITE]


class Walker(Component):
    """What one person knows: where they are going and how fast."""

    velocity: Vector3
    destination: Vector3
    pace: Float = 1.2


```

The engine keeps that data in an `EntityStore`: one numpy array per component,
with a row for every entity slot and the fields packed side by side. This is
why a crowd is cheap. The 200 velocities are one array, not 200 little objects,
and the store can hand a whole slice of them to numpy in one call. The store is
sized once, at `Engine(max_entities=...)`; it starts at a hundred thousand rows
and reuses the slots of entities that are destroyed.

`engine.spawn` puts entities into the store. Given `n=`, it fills `n` rows in one
call; a value that is a single point or a single number is broadcast across
them, and an array with one row per entity is written as it is. The call returns
an `EntityHandle` to the first of the new entities, which we will use for one
person in a moment. Each person also gets `Layer(layer=1)`, a render tag that
keeps the crowd separable from the plaza; we come back to it.

```python {run continue export=examples/walkers.py}
engine = mx.Engine("plaza", width=480, height=270)
engine.spawn(
    Mesh(plane(40, 40)),
    Material(StandardMaterial(color="#c8b89a", roughness=0.9)),
    Transform(rot=FLOOR),
)
engine.spawn(
    Mesh(cube(4, 1, 4)),
    Material(StandardMaterial(color="#8a9bb0", roughness=0.6)),
    Transform(pos=(0, 0.5, 0)),
)
engine.camera.set_pose(position=(0, 10, 15), target=(0, 0.8, -6))
engine.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.4, -1, -0.3)))


```

Before the crowd exists, we need to decide where everyone is going. Spreading
200 goals along the arcade so they do not pile on the same spot is a one-time
selection: it should not sit in the frame path. `@engine.background` decorates a
pure function — plain data in, plain data out, no engine calls inside — and every
call runs it on a worker thread and returns a `Task` instead of the value.
`task.wait()` blocks until it is there; `task.ready` and `task.result` check
without waiting, and `task.on_ready(fn)` hands the value to `fn` on the next
frame instead. In a window the frame loop keeps drawing while the worker runs;
in this still we wait once, then spawn.

The selection itself uses `all_pairs`, which returns the distance grid of the
points as a `Pairs`: `diff[i, j]` is the vector from point `i` to point `j`,
`dist` its length, and `dist_safe` the same length with a floor so a division
never divides by zero. The loop keeps the point farthest from everything chosen
so far, which spreads 200 goals evenly along a line.

```python {run continue export=examples/walkers.py}
@engine.background
def arcade(seed: int, n: int) -> np.ndarray:
    """n destinations along the east arcade, as far apart as we can make them."""
    rng = np.random.default_rng(seed)
    candidates = np.column_stack(
        [
            np.full(n * 4, 18.0),
            np.full(n * 4, 0.85),
            rng.uniform(-16.0, -4.0, n * 4),
        ]
    ).astype(np.float32)
    pairs = all_pairs(candidates)
    chosen = [0]
    nearest = pairs.dist[0].copy()
    for _ in range(n - 1):
        k = int(np.argmax(nearest))
        chosen.append(k)
        nearest = np.minimum(nearest, pairs.dist[k])
    return candidates[chosen]


destinations = arcade(7, N).wait()

rng = np.random.default_rng(11)
start = np.column_stack(
    [
        rng.uniform(-19.0, 8.0, N),
        np.full(N, 0.85),
        rng.uniform(-16.0, -4.0, N),
    ]
).astype(np.float32)
pace = rng.uniform(0.9, 1.6, N).astype(np.float32)

body = Mesh(cube(0.4, 1.7, 0.25))
walkers = []
for i, coat in enumerate(COATS):
    group = slice(i * 50, (i + 1) * 50)
    walkers.append(
        engine.spawn(
            body,
            Material(StandardMaterial(color=coat, roughness=0.8)),
            Transform(pos=start[group]),
            Walker(
                velocity=np.zeros((50, 3), np.float32),
                destination=destinations[group],
                pace=pace[group],
            ),
            Layer(layer=1),
            n=50,
        )
    )

view = engine.store.get_component_view(["Transform", "Walker"])
print(len(view), "walkers,", len(engine.store.get_component_view([])), "entities alive")
Image.fromarray(engine.render_frame(supersample=1))


```

Two hundred people stand where they were spawned, each already holding a
destination. `get_component_view` is the question "which rows have these
components?". It gives a `ComponentView`, the same object a system receives: the
alive entities that have every named component, in ascending index order.
`view[Walker]` is a `ComponentAccessor`, and `.velocity` is a view into the three
columns of that field. `.data` materialises them. Everything is still one numpy
array underneath:

```python {run continue}
print(Walker._layout)
print("velocity", view[Walker].velocity.data.shape)
print("paces", view[Walker].pace.data[:5, 0])
print("first rows", [int(i) for i in view][:3])
```

`Walker._layout` is the column map the annotation built: velocity at 0, the
destination at 3, and the one-column pace at 6. The pace is stored as one column
per person, so `pace.data` is `(200, 1)`; the system takes `[:, 0]` of it to get
200 numbers.

## The system that walks them

A system is a function the engine calls once per frame, with a view of the
entities that have the components it asks for. The request is the annotation:
`Query[Transform, Walker]` means "every entity with both". At registration the
decorator wraps the function in a `System`, which remembers the function and the
component names, and appends it to the engine's `SystemRegistry`; the registry
runs its systems in the order they were registered, and `engine.remove_system`
takes one out again.

Inside the system, reads are current and writes are queued.
`query[Transform].pos += step` turns into a command, and the engine applies the
commands after every system has run. That keeps a frame's data consistent while
it is being computed, but it also means two systems that add to the same field
in one frame do not stack: each reads the same starting array and the later
command wins. Steering and pushing apart are therefore one system here.

A person wants to travel at their own pace along the vector to their
destination; a fraction of `dt` turns the velocity toward that, so they speed up
and slow down instead of snapping. The separation term uses `Pairs` again, this
time on the people. For every pair closer than 0.9 m, the overlap times the unit
vector away from the neighbour is a push; summed over the row, it is the push
away from everyone too close at once. It goes into the wanted velocity, so a
crowd jostles rather than teleports.

The second system turns each person to face their destination. A rotation is a
`Vector4`, the quaternion `(x, y, z, w)`; `Transform.rotation(y=...)` builds one
per row from an array of angles. It writes only the four rotation columns of
`Transform`, a separate command from the position columns the first system
writes, so both land.

```python {run continue export=examples/walkers.py}
@engine.system
def walk(query: mx.Query[Transform, Walker], dt: float):
    pos = query[Transform].pos.data
    vel = query[Walker].velocity.data
    dest = query[Walker].destination.data
    pace = query[Walker].pace.data[:, 0]

    to = dest - pos
    dist = np.linalg.norm(to, axis=1, keepdims=True)
    wanted = to / np.maximum(dist, 1e-6) * pace[:, None]
    wanted[dist[:, 0] < 0.6] = 0.0  # close enough: stop where you are

    pairs = all_pairs(pos, softening=0.01)
    overlap = np.maximum(0.9 - pairs.dist, 0.0)
    np.fill_diagonal(overlap, 0.0)
    push = (-pairs.diff / pairs.dist_safe[..., None] * overlap[..., None]).sum(axis=1)
    wanted += push * 1.5

    new_vel = vel + (wanted - vel) * np.minimum(1.0, 4.0 * dt)
    query[Walker].velocity += new_vel - vel
    query[Transform].pos += new_vel * dt


@engine.system
def face(query: mx.Query[Transform, Walker], dt: float):
    to = query[Walker].destination.data - query[Transform].pos.data
    query[Transform].rot = Transform.rotation(y=np.arctan2(to[:, 0], to[:, 2]))


engine.set_fixed_timestep(0.05)
for _ in range(40):
    frame = engine.render_frame(supersample=1)
print("mean x after 2 s:", round(float(view[Transform].pos.data[:, 0].mean()), 2))
Image.fromarray(frame)
```

`render_frame` runs the systems once and draws the result, so forty calls are
two seconds of walking. The mean x moves east, but the still is more useful than
the number: some people have started, some are already half way, and the ones
behind the fountain pass around each other.

## One person in particular

`spawn` returned an `EntityHandle` for each group, pointing at the first entity
of the fifty. A handle is for the single entity you keep: `handle.transform.pos`
reads and writes that person's position, `handle.set_material` changes what they
wear, `handle.set_geometry` swaps their mesh, and `handle.destroy` removes them.
Here it singles out a guide in dark blue and moves them to the fountain steps.

That is also how colour works in this engine: an entity has no colour of its
own, and what you see is the `Material` it holds. The renderer batches entities
by material — everyone in the same coat is one draw call. Four coats are four
batches for 200 people; one person with a material of their own adds a fifth.
`Colors` is the short list of named hex strings we spawned from, and `Color` is
the value behind them: `Color(Colors.CYAN).r` reads a channel, `to_linear` and
`to_srgb` convert between the two meanings of a colour. A material takes the hex
string, the three sRGB floats or the `Color` itself.

```python {run continue export=examples/walkers.py}
leader = walkers[0]
leader.set_material(StandardMaterial(color="#0b7285", roughness=0.4))
leader.transform.pos = np.array([-5.0, 0.85, 1.8], np.float32)
print("the guide is entity", leader.index)

for _ in range(20):
    frame = engine.render_frame(supersample=1)
Image.fromarray(frame)
```

## Two hundred goals

The crowd is heading somewhere, and it helps to see where. A small post at each
destination, coloured with a dimmed version of that group's coat, turns the
destinations into a diagram. The posts hold no `Layer`, so they sit on layer 0
with the paving and the basin.

`Layer` is a component like any other, a single `Float`; an entity without one
sits on layer 0, and the people were spawned on layer 1. A camera draws only the
layers in its `layers` set, and `None`, the default, means every layer. The plaza
camera is set to `{0, 1}`, which is the whole scene.

```python {run continue export=examples/walkers.py}
marker = Mesh(sphere(0.3, 8))
for i, coat in enumerate(COATS):
    c = Color(coat)
    group = slice(i * 50, (i + 1) * 50)
    engine.spawn(
        marker,
        Material(StandardMaterial(color=(0.8 * c.r, 0.8 * c.g, 0.8 * c.b), roughness=0.9)),
        Transform(pos=destinations[group]),
        n=50,
    )
engine.camera.layers = {0, 1}
print(len(engine.store.get_component_view(["Layer"])), "people on layer 1")
```

Set the camera to `{1}` and the paving, the fountain and the posts go away: the
picture is the crowd alone, a way to watch the flow without the scenery.

```python {run continue}
engine.camera.layers = {1}
frame = engine.render_frame(supersample=1)
engine.camera.layers = {0, 1}
Image.fromarray(frame)
```

Two hundred people on their own: no paving, no fountain, no posts. Nothing in
the scene changed between these two frames; only the camera's layer set did.

## The finished program

Six more seconds of walking, then a count of who is still on their way:

```python {run continue export=examples/walkers.py}
for _ in range(120):
    frame = engine.render_frame(supersample=1)

pos = view[Transform].pos.data
left = np.linalg.norm(view[Walker].destination.data - pos, axis=1)
pairs = all_pairs(pos)
off = ~np.eye(N, dtype=bool)
print(
    "walking:",
    int((left > 0.7).sum()),
    "closest pair:",
    round(float(pairs.dist[off].min()), 2),
)
Image.fromarray(frame)
```

The blocks above are the whole program, tangled into
`docs/examples/walkers.py`. It runs headless and ends on the plaza after nine
seconds of crossing; the `print` reports who has arrived. To watch it live,
end the program with `engine.run()`, which opens a window and calls the same
systems on every screen refresh until you close it.

```python
engine.run()
```

At 200 people the drawing dominates: the steering and the distance pass cost a
few milliseconds together, and the distance pass grows with the square of the
crowd. That is the one part of this chapter that does not scale by adding rows,
and the reason chapter 10 moves the crowd onto the GPU. Until then the shape
holds: the store is a handful of arrays, the systems two functions over those
arrays, and when the people get skeletons their joint matrices go in a
`JointStore`, one block per posed entity, which chapter 7 uses. The next
chapter gives the visitor a way to step into the plaza.
