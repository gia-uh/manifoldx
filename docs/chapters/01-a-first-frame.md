# A first frame

The plaza starts as a plan on paper: a paved square forty metres on a side,
and a fountain basin in the middle, four metres wide and one metre tall. Before
anything moves, we want to see it from where a visitor would stand.

## Drawing it by hand

With only numpy and Pillow we can already make a picture. Put the camera
somewhere, turn each corner of the basin into a pixel with a perspective
divide, and draw the edges.

```python {run}
import numpy as np
from PIL import Image, ImageDraw

# The basin: a box 4 m wide, 1 m tall, centred on the origin. y points up.
corners = np.array(
    [[x, y, z] for x in (-2, 2) for y in (0, 1) for z in (-2, 2)], dtype=float
)
edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
         if np.sum(corners[a] != corners[b]) == 1]

eye = np.array([14.0, 9.0, 14.0])
forward = -eye / np.linalg.norm(eye)
right = np.cross(forward, [0, 1, 0])
right /= np.linalg.norm(right)
up = np.cross(right, forward)

def to_pixel(p, w=480, h=270, focal=600):
    d = p - eye
    x, y, z = d @ right, d @ up, d @ forward
    return (w / 2 + focal * x / z, h / 2 - focal * y / z)

img = Image.new("RGB", (480, 270), "white")
draw = ImageDraw.Draw(img)
for a, b in edges:
    draw.line([to_pixel(corners[a]), to_pixel(corners[b])], fill="black")
img
```

It is a fountain, more or less. But everything a picture of a place needs is
missing. The back edges show through the front faces, because nothing knows
which surface is nearer. There is no ground, no light and no colour; adding a
floor means another loop of edges, and filling faces means sorting them by
depth ourselves. And this is one box. The plaza will have hundreds of objects,
and each frame of a simulation will move them.

## The engine

ManifoldX does that work on the GPU. An `Engine` owns the window, the camera
and the world; you put things in the world and it draws them.

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
lights the top of the basin brighter than its sides. Nothing in our code
sorted a face or shaded a pixel.

To look at the plaza in a window instead, install the `desktop` extra and end
the program with `engine.run()`, which opens the window and draws a frame
every time the screen refreshes, until you close it.

```python
engine.run()
```

The basin is one entity among two. The next chapter fills the plaza with
people, hundreds of them, and moves them every frame without a Python loop.
