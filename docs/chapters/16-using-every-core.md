# Using every core

The plaza's morning walk has one system moving every walker. It is fast enough
for forty people, and it is the wrong shape: a walker heading north and one
heading east do the same arithmetic on different rows, and one Python thread
does both. The work splits by where a walker is going. So does the code, and on
free-threaded Python the four systems can run at the same time.

ManifoldX 0.11 requires CPython 3.14t, the free-threaded build: the interpreter
has no global interpreter lock, so two systems that touch different rows of the
store really run on two cores. On a normal build the same program still works,
one system after another, and warns once the first time a parallel system is
registered.

## Four directions, four systems

A system marked `parallel=True` joins the group of consecutive systems around
it. Here the walkers are spawned in four sets, each carrying its own marker
component, and each system queries `Transform` plus its marker: the northbound
system sees only northbound walkers, and writes only their rows.

```python {run export=examples/plaza_parallel.py}
import os

import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, cube


class Northbound(Component):
    pass


class Southbound(Component):
    pass


class Eastbound(Component):
    pass


class Westbound(Component):
    pass


WALKERS = 60


def build():
    """The plaza with four sets of walkers, each set with its own system."""
    rng = np.random.default_rng(11)
    engine = mx.Engine("parallel plaza", width=480, height=270)
    engine.background_color = (0.12, 0.14, 0.2)
    engine.camera.fit(24.0, center=(0.0, 1.0, 0.0), azimuth=45, elevation=34)
    engine.set_sun(DirectionalLight(color="#fff1d6", intensity=2.6, direction=(-0.4, -1.0, -0.5)))

    engine.spawn(
        Mesh(cube(40, 0.2, 40)),
        Material(StandardMaterial("#6b6a63", roughness=0.95)),
        Transform(pos=(0.0, -0.1, 0.0)),
    )
    engine.spawn(
        Mesh(cube(3, 1.2, 3)),
        Material(StandardMaterial("#8fa3b8", roughness=0.4)),
        Transform(pos=(0.0, 0.6, 0.0)),
    )

    for marker, color in [
        (Northbound, "#c8a06a"),
        (Southbound, "#a6735a"),
        (Eastbound, "#7f9d6a"),
        (Westbound, "#6e7f9d"),
    ]:
        positions = np.column_stack([
            rng.uniform(-16.0, 16.0, WALKERS),
            np.full(WALKERS, 0.55),
            rng.uniform(-16.0, 16.0, WALKERS),
        ]).astype(np.float32)
        engine.spawn(
            marker(),
            Mesh(cube(0.35, 1.1, 0.35)),
            Material(StandardMaterial(color)),
            Transform(pos=positions),
            n=WALKERS,
        )

    @engine.system(parallel=True)
    def north(q: mx.Query[Transform, Northbound], dt: float):
        q[Transform].position += np.array([0.0, 0.0, -3.0 * dt], np.float32)

    @engine.system(parallel=True)
    def south(q: mx.Query[Transform, Southbound], dt: float):
        q[Transform].position += np.array([0.0, 0.0, 3.0 * dt], np.float32)

    @engine.system(parallel=True)
    def east(q: mx.Query[Transform, Eastbound], dt: float):
        q[Transform].position += np.array([3.0 * dt, 0.0, 0.0], np.float32)

    @engine.system(parallel=True)
    def west(q: mx.Query[Transform, Westbound], dt: float):
        q[Transform].position += np.array([-3.0 * dt, 0.0, 0.0], np.float32)

    @engine.system
    def count_the_edge(q: mx.Query[Transform], dt: float):
        position = q[Transform].position.data
        engine.edge = int((np.abs(position[:, [0, 2]]) > 15.0).any(axis=1).sum())

    return engine


engine = build()
for _ in range(6):
    frame = engine.render_frame(supersample=1)

print("at the edge:", engine.edge)
print("per system, ms:", {name.rsplit(".", 1)[-1]: round(1000 * seconds, 2)
                          for name, seconds in engine.system_split.items()})
print("group wall, ms:", [round(1000 * seconds, 2) for seconds in engine.group_split])
print("systems total, ms:", round(1000 * sum(engine.system_split.values()), 2))
```

The four walkers' systems are next to each other, so they form one group; the
fifth has no `parallel=True`, so it is a barrier. It starts only after every
northbound, southbound, eastbound and westbound walker has moved, and the next
frame's group starts only after it has returned. That is the whole contract:
consecutive parallel systems run together, and a serial system is a wall between
two groups.

Each system writes through its query as usual. What is different is where those
writes go: every system in a group gets a command buffer of its own, and when
all of them have finished their buffers are appended to the frame's in
**registration order**, not in the order the systems happened to finish. A
frame's store is therefore the store a serial run of the same systems produces,
whatever the thread timing.

The group's wall time is the number to compare with the sum of the group's
systems' times. Four systems of about a millisecond each should finish a group
in about a millisecond, not four. `engine.system_split` resets every frame and
tells you which system to split next; the five slowest are also added to the
once-a-second line of `manifoldx run --debug` and `manifoldx state`, as
`systems name ms ...`. The benchmark suite runs the same shape headless:

```bash
uv run python -m benchmarks frame --case parallel_systems --sizes 1 4 8 --against main
```

## What a parallel system may do

A query is safe because query reads come from the start of the frame and query
writes are commands: neither system can see the other's half-done work. The
same reasoning does not cover everything a system can reach:

- **Yes:** read and write through its query, call `engine.emit`, and call
  `engine.destroy`. Destroy only appends a command, so it lands in the system's
  buffer with the writes.
- **No:** call `engine.spawn`. Spawn writes the entity store at once, and the
  other systems are reading it; the call raises a `RuntimeError` naming the
  system, and tells you to spawn from a serial system.
- **No:** write a shared Python object without a lock, or rely on another
  system of the same group having run. A group's systems are ordered with
  respect to the frame, not to each other.

The last rule is easy to break by accident with a module-level list or counter.
If a number changes when you did not expect it to, run the group serially
before hunting further:

```python {run continue export=examples/plaza_parallel.py}
parallel, serial = build(), build()
for _ in range(12):
    parallel.render_frame(supersample=1)

os.environ["MANIFOLDX_PARALLEL"] = "serial"
for _ in range(12):
    serial.render_frame(supersample=1)
del os.environ["MANIFOLDX_PARALLEL"]

same = parallel.store._components["Transform"].tobytes() == serial.store._components["Transform"].tobytes()
print("same store after 12 frames:", same)
parallel.close()
serial.close()
```

With `MANIFOLDX_PARALLEL=serial` the group still gives each system a command
buffer of its own and still flushes them in registration order, but it runs
them one after another on the calling thread. If the store matches the parallel
run, thread timing did not change the result, and a difference from one run to
the next means the systems share something they should not. A build with the
GIL on, or a group of one system, takes the same path — which is why the same
program runs unchanged on a normal CPython. `engine.group_split` lists each
group's wall seconds in run order, so the same printout shows what serial mode
costs.

## The picture

The final frame is the whole plaza again, with every walker where its own system
put it:

```python {run continue export=examples/plaza_parallel.py}
Image.fromarray(frame)
```

One last thing to know before marking a system parallel: the work has to be big
enough to pay for the trip to another core. The frame pool has one thread per
core and a group waits for all of its systems, so a system that takes
microseconds costs more in dispatch than it saves. The plaza's walkers move
thousands of rows in one vectorised expression; that is worth a thread. A
system that updates one entity is not.
