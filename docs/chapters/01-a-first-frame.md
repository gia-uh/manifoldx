# A first frame

The plaza starts as a plan on paper: a paved square forty metres on a side,
and a fountain basin in the middle, four metres wide and one metre tall. Before
anything moves, we want to see it from where a visitor would stand.

## The engine

An `Engine` owns the window, the camera and the world; you put things in the
world and it draws them on the GPU.

```python {run export=examples/first_frame.py}
import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane
from PIL import Image

engine = mx.Engine("plaza", width=480, height=270)
```

A thing in the world is an *entity*, and it is made of components. Three are
enough to see something. `Mesh` says what shape it has, `Material` how its
surface looks, and `Transform` where it stands. `cube` and `plane` return
geometry the engine uploads once; `StandardMaterial` is a physically based
surface with a colour and a roughness.

```python {run continue export=examples/first_frame.py}
FLOOR = Transform.rotation(x=-np.pi / 2)  # a quarter turn about x: lay the plane flat

engine.spawn(
    Mesh(plane(40, 40)),
    Material(StandardMaterial(color="#c8b89a", roughness=0.9)),
    Transform(pos=(0, 0, 0), rot=FLOOR),
)
engine.spawn(
    Mesh(cube(4, 1, 4)),
    Material(StandardMaterial(color="#8a9bb0", roughness=0.6)),
    Transform(pos=(0, 0.5, 0)),
)
```

`plane` lies in the x-y plane, facing +z, so the floor needs a quarter turn to
lie flat. A `Transform` stores its rotation as a quaternion `(x, y, z, w)`;
`Transform.rotation` builds one from angles in radians about x, y and z, so
`FLOOR` is a quarter turn about the x axis.

The `Camera` comes with the engine. `set_pose` puts it at the visitor's eye
and points it at the fountain; a `DirectionalLight` is the sun.

```python {run continue export=examples/first_frame.py}
engine.camera.set_pose(position=(14, 9, 14), target=(0, 0, 0))
engine.set_sun(DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.4, -1, -0.3)))
```

## Seeing it

`render_frame` draws one frame offscreen and returns it as a numpy array of
shape `(height, width, 3)`, so it works on a server, in a notebook and in this
guide. Pillow, imported at the top, turns the array into an image.

```python {run continue export=examples/first_frame.py}
frame = engine.render_frame()
print(frame.shape, frame.dtype)
Image.fromarray(frame)
```

The near faces hide the far ones, the floor runs to the horizon, and the sun
lights the top of the basin brighter than its sides. Our code only said what
is where; the engine sorted the faces and shaded the pixels.

To look at the plaza in a window instead, install the `desktop` extra and end
the program with `engine.run()`, which opens the window and draws a frame
every time the screen refreshes, until you close it.

```python
engine.run()
```

The basin is one entity among two. The next chapter fills the plaza with
people, hundreds of them, and moves them every frame without a Python loop.
