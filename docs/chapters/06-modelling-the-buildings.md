# Modelling the buildings

The plaza has a floor and a fountain, but no town around it. The north side needs
a town hall: a stone facade with two arched doors and a round window, a dome
behind it, paving that has settled for a hundred years, and countryside rising
past the edge of the square. A box primitive can be a wall, but a doorway is a
hole, a dome is half a sphere, and the ground is a surface no one drew. Each part
needs a different modelling operation.

## A wall, as a value

`manifoldx.modeling` works on a `Mesh` that lives on the CPU. It holds two numpy
arrays: `positions`, one row per vertex, and `faces`, one row of three vertex
indices per triangle. It is a frozen value. Every operator returns a new `Mesh`
and leaves the old one alone, so a pipeline reads as a chain of expressions
rather than a sequence of mutations. `box`, `plane`, `cylinder`, `icosphere` and
`torus` build the first one.

The other `Mesh` in the library is a component: the handle the engine uploads to
the GPU. The two never mix. We import the value as `GeoMesh`, and hand its
`to_geometry()` dict to the component, which is the one seam between the two.

```python {run}
from PIL import Image

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling import Mesh as GeoMesh
from manifoldx.resources import DirectionalLight, StandardMaterial

engine = mx.Engine("plaza", width=480, height=270)
engine.background_color = (0.62, 0.72, 0.84)
engine.set_sun(DirectionalLight(color="#fff3dd", intensity=3.2, direction=(-0.45, -0.85, -0.3)))
engine.enable_shadows(resolution=1024, bias=0.004, pcf_radius=2)

wall = GeoMesh.box(16, 10, 0.6).translate((0, 5, -20))
print(wall.positions.shape, wall.faces.shape)
print({name: (array.shape, array.dtype.name) for name, array in wall.to_geometry().items()})

engine.spawn(
    Mesh(wall.to_geometry()),
    Material(StandardMaterial(color="#cfc3a6", roughness=0.9)),
    Transform(),
)
engine.camera.set_pose(position=(7, 6, 4), target=(0, 5, -20))
Image.fromarray(engine.render_frame(supersample=1))
```

Eight vertices, twelve triangles, and a geometry dict whose `indices` are the
same faces flattened. `to_geometry()` computes vertex normals when the value has
none, so the wall is lit even though we never called `recompute_normals`. The
`from_geometry` inverse lets a mesh loaded from a file enter the same operators.

The picture is plain: a blank wall, sixteen metres wide and ten tall, standing
at the north edge of the square. It needs openings.

## Doors cut out

A doorway is not a box, because its top is a half-circle. It is a box with a
cylinder laid on its side, and a hole in a wall is one solid subtracted from
another: `difference` does exactly that. The cutters are merged first with
`union`, so one subtraction removes both doors and the round window.

The wall also gets a cornice. `extrude` raises a selected face region and stitches
the side walls, but it raises each boundary vertex along its *vertex* normal, not
along the face normal. A wall only 0.6 m thick has broad front and back faces, so
the corner normals lean sideways; the raised copy therefore slides out over the
plaza far more than it rises. On this wall it becomes a cornice with a deep
overhang. That is also why the extrusion comes before the boolean: a boolean
returns every triangle with its own copy of its corners, so a later `extrude`
would treat each triangle as a separate region and tear.

```python {run}
import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling import Mesh as GeoMesh
from manifoldx.resources import DirectionalLight, StandardMaterial

engine = mx.Engine("plaza", width=480, height=270)
engine.background_color = (0.62, 0.72, 0.84)
engine.set_sun(DirectionalLight(color="#fff3dd", intensity=3.2, direction=(-0.45, -0.85, -0.3)))
engine.enable_shadows(resolution=1024, bias=0.004, pcf_radius=2)


def along_z(mesh):
    """Turn a mesh a quarter turn about x, so a y-axis cylinder runs along z."""
    x, y, z = mesh.positions.T
    return mesh.with_positions(np.stack([x, -z, y], axis=1))


wall = GeoMesh.box(16, 10, 0.6).translate((0, 5, 0))
top = wall.positions[wall.faces].mean(axis=1)[:, 1] > 9.99
cornice = wall.extrude(top, 0.7)
print(cornice.positions.shape, cornice.faces.shape)

door = GeoMesh.box(1.9, 2.7, 2.0).translate((-3.6, 1.35, 0))
arch = along_z(GeoMesh.cylinder(radius=0.95, height=2.0, segments=24))
doorway = door.union(arch.translate((-3.6, 2.7, 0)))
window = along_z(GeoMesh.cylinder(radius=0.95, height=2.0, segments=24))
cutter = doorway.union(doorway.translate((7.2, 0, 0))).union(window.translate((0, 7.1, 0)))
facade = cornice.difference(cutter).translate((0, 0, -20.0))
print(facade.positions.shape, facade.faces.shape)

engine.spawn(
    Mesh(facade.to_geometry()),
    Material(StandardMaterial(color="#cfc3a6", roughness=0.9)),
    Transform(),
)
body = GeoMesh.box(16, 10, 8).translate((0, 5, -25.0))
engine.spawn(
    Mesh(body.to_geometry()),
    Material(StandardMaterial(color="#c9bda0", roughness=0.9)),
    Transform(),
)
engine.camera.set_pose(position=(7, 6, 4), target=(0, 6, -20))
Image.fromarray(engine.render_frame(supersample=1))
```

`along_z` is our own four lines: the value type can move points with `translate`
and `with_positions`, but it has no rotate operator, so a quarter turn about x is
a matrix written out. The cylinder is built along y and must lie along z to bore
through the wall.

The cornice goes from 12 vertices and 20 faces to a facade of 2,216 vertices and
994 faces. The cutters cost more than the wall: a BSP boolean is pure Python, so
its cost follows the triangle counts of both solids. Keep cutters coarse; a
24-sided cylinder is plenty for a door arch. The same `union`, `difference` and
`intersection` build every solid here, and `intersection` is what makes the dome
in a moment.

Behind the doors the body of the hall stands five metres back, so the openings
read as dark recesses.

## A dome, cut and worn

A dome is the top half of a sphere. `icosphere` gives a geodesic sphere; one
`subdivide` with the `"loop"` scheme smooths it and quadruples its triangles,
which is the point of the scheme — the default `"midpoint"` only splits each triangle
without moving the old vertices.

The copper is old and dented. A sculpt brush is a radial selection with a
strength: `Falloff` is the region, with weight 1 at its centre and 0 at its rim,
and `inflate` pushes the selected vertices along their normals. A negative
strength pushes them inward. We build one by hand to inspect its weights, then
give its centre and radius to the brush. The brush is wide and shallow, a
two-metre radius and a third of a metre deep, so the copper reads as dented
rather than pinched. `smooth` then relaxes the dent with the
one-ring neighbourhood from `adjacency()`, which returns a `VertexAdjacency`: two
arrays in CSR form, so vertex *i*'s neighbours are
`neighbors[offsets[i]:offsets[i + 1]]`. `draw`, `pinch` and `flatten` are the
other brushes, and all of them take a centre, a radius and a profile.

The brushes act on the sphere *before* the boolean. After a boolean, every
triangle owns its corners, so `adjacency()` sees each triangle as an island and
`smooth` would pull its three vertices toward its own centre instead of blending
neighbours.

```python {run}
import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling import Falloff, Mesh as GeoMesh
from manifoldx.resources import DirectionalLight, StandardMaterial

engine = mx.Engine("plaza", width=480, height=270)
engine.background_color = (0.62, 0.72, 0.84)
engine.set_sun(DirectionalLight(color="#fff3dd", intensity=3.2, direction=(-0.45, -0.85, -0.3)))
engine.enable_shadows(resolution=1024, bias=0.004, pcf_radius=2)

sphere = GeoMesh.icosphere(subdivisions=1, radius=4.2).subdivide(iterations=1, scheme="loop")
dent = Falloff((3.0, 2.4, 1.6), 2.0)
print(dent.weights(np.array([[3.0, 2.4, 1.6]])), (dent.weights(sphere.positions) > 0).sum())
adjacency = sphere.adjacency()
print(type(adjacency).__name__, adjacency.offsets.shape, adjacency.neighbors.shape)

sphere = sphere.inflate(center=dent.center, radius=dent.radius, strength=-0.35)
sphere = sphere.smooth(iterations=3, strength=0.5, center=dent.center, radius=dent.radius + 0.8)

keeper = GeoMesh.box(20, 20, 20).translate((0, 10, 0))
dome = sphere.intersection(keeper)
print(dome.positions.shape, dome.faces.shape)
dome = dome.decimate(grid=128)
print(dome.positions.shape, dome.faces.shape)

lattice = dome.ffd(resolution=(3, 3, 3))
for i in range(4):
    for k in range(4):
        lattice.move(i, 3, k, (0.0, 0.25, 0.0))
dome = lattice.apply().translate((0, 13.0, -24.5))

drum = GeoMesh.cylinder(radius=4.05, height=3.0, segments=40).translate((0, 11.5, -24.5))
engine.spawn(
    Mesh(drum.to_geometry()),
    Material(StandardMaterial(color="#c9bda0", roughness=0.9)),
    Transform(),
)
engine.spawn(
    Mesh(dome.to_geometry()),
    Material(StandardMaterial(color="#7d9b8c", roughness=0.4, metallic=0.25)),
    Transform(),
)
engine.camera.set_pose(position=(6, 16.5, -17), target=(0, 14.8, -24.5))
Image.fromarray(engine.render_frame(supersample=1))
```

The `intersection` with a large box translated up by ten metres keeps the half
above y = 0. The sphere has 320 triangles; the boolean returns 217 and 589
vertices, because every surviving triangle keeps its own corners. That flat
shading is visible on a curved surface, so `decimate` finishes the job:
`decimate(grid=128)` snaps vertices to a lattice and averages each occupied cell.
A coarse grid simplifies the shape; a grid this fine only merges the coincident
copies, taking 589 vertices to 119 for the same 217 faces, and the dome shades as
one smooth surface.

`ffd` returns an `FFD`: a Bezier control lattice embedding the mesh, here
4 × 4 × 4 points over its bounding box. `move` offsets one control point and
returns the lattice, and `apply` evaluates it. We lift the top row by a quarter
of a metre, which gives the hemisphere a gentle crown while it stays a dome.
With no point moved, the lattice reproduces the original mesh within rounding
error.

## Old paving

The paving is a plane with 128 segments a side. Two `Field`s displace it: one
slow `fbm` for the settlement of centuries and one fine one for wear. A `Field`
wraps a function from points to a scalar and is itself callable, so
`displace(field, amount)` samples it at every vertex and moves the vertex along
its normal. Fields compose with arithmetic and methods, `+`, `*`, `remap`,
`clamp`, `power`, `warp` and the rest, which is how a terrain is written as an
expression instead of a loop. `displace` is the deformer that takes a field;
`twist`, `bend` and `taper` are the others, and each needs only an angle or a
factor because it acts on the whole mesh.

The stone pattern is another field. `worley` returns the cellular distance
`f2f1`, small on the joint between two cells and larger inside a stone, and a
`Gradient` maps that scalar to colour by interpolating between stops. `color_by`
does the two steps at once: it samples the field at every vertex and stores the
result with `with_colors`, and a `StandardMaterial` with `vertex_colors=True`
draws it. The field's `freq` sets the cells: at 0.6 a cell is about 1.7 m across,
eight times the plane's 0.2 m vertex spacing, because the pattern lives only in
vertex colours and a cell narrower than a few vertices dissolves into noise.

```python {run}
from PIL import Image

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling import Gradient, Mesh as GeoMesh, fields
from manifoldx.resources import DirectionalLight, StandardMaterial

engine = mx.Engine("plaza", width=480, height=270)
engine.background_color = (0.62, 0.72, 0.84)
engine.set_sun(DirectionalLight(color="#fff3dd", intensity=3.2, direction=(-0.45, -0.85, -0.3)))
engine.enable_shadows(resolution=1024, bias=0.004, pcf_radius=2)

settle = fields.fbm(seed=4, freq=0.12, octaves=3) * 0.05
wear = fields.fbm(seed=11, freq=0.9, octaves=3) * 0.03
paving = GeoMesh.plane(width=40, depth=40, segments=200).displace(settle + wear, amount=1.0)
print(paving.positions.shape, paving.faces.shape)

stones = fields.worley(seed=3, freq=0.6, feature="f2f1")
paving = paving.color_by(
    stones,
    Gradient([(0.00, "#5f584c"), (0.12, "#8d8472"), (0.35, "#c3b7a0"), (0.85, "#d8cdb5")]),
)
engine.spawn(
    Mesh(paving.to_geometry()),
    Material(StandardMaterial(color="#ffffff", roughness=0.95, vertex_colors=True)),
    Transform(),
)
engine.camera.set_pose(position=(5, 5, 5), target=(4, 0, 3.5))
Image.fromarray(engine.render_frame(supersample=1))
```

The plane is 40 m on a side, so it is exactly the square of the plaza, and it
lies in the x-z plane with its normal up: `displace` along the normal is the
vertical settling we want. The gradient clamps outside its stops, so a scalar
below 0.0 or above 0.8 takes the end colour instead of wrapping.

Most sources carry a small symbolic syntax tree beside their callable.
`field_to_wgsl` walks that tree and emits a WGSL function, so the same field can
move vertices in a shader. `fbm` transpiles; `worley` does not, because it is
built from a hand-written callable and has no tree at all. In 0.10 that raises
the message of the missing tree rather than naming the source.

```python {run}
from manifoldx.modeling import FieldNotTranspilable, field_to_wgsl, fields

settle = fields.fbm(seed=4, freq=0.12, octaves=3) * 0.05
stones = fields.worley(seed=3, freq=0.6, feature="f2f1")
print(field_to_wgsl(settle, name="settle").splitlines()[-2])
try:
    field_to_wgsl(stones, name="stones")
except FieldNotTranspilable as error:
    print(type(error).__name__, ":", error)
```

The emitted shader is cheap value noise, not the CPU's permutation-table Perlin;
what survives the translation is the composition, not the exact value. Keep the
CPU field when the shape must be identical.

## The land beyond

The countryside is the same idea at a larger scale, and a `Field` is where the
whole expression lives. `ridged` gives sharp mountain ridges, `fbm` adds rough
ground, arithmetic mixes them, and `warp` bends the coordinates with two more
fields so the ridges meander instead of running straight. `distance()` is the
radius from the origin, and `remap` turns it into a mask: zero over the plaza,
one far away. Multiplying by the mask and raising it with `power` keeps the
square flat and sharpens the far peaks.

`decimate` then reduces the displaced grid before it becomes a GPU buffer; a
terrain does not need every one of 18,000 triangles. The colour is one more
field: `coord("y")` reads the height after displacement, and a `Gradient` sends
it from dark grass through dry upland to pale rock.

```python {run}
from PIL import Image

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling import Gradient, Mesh as GeoMesh, fields
from manifoldx.resources import DirectionalLight, StandardMaterial

engine = mx.Engine("plaza", width=480, height=270)
engine.background_color = (0.62, 0.72, 0.84)
engine.set_sun(DirectionalLight(color="#fff3dd", intensity=3.2, direction=(-0.45, -0.85, -0.3)))
engine.enable_shadows(resolution=1024, bias=0.004, pcf_radius=2)

relief = (
    fields.ridged(seed=5, freq=0.035, octaves=4) * 0.6
    + fields.fbm(seed=7, freq=0.18, octaves=3) * 0.4
).warp(8.0, fx=fields.fbm(seed=2, freq=0.05), fz=fields.fbm(seed=9, freq=0.05))
hill = fields.distance().remap(24.0, 64.0, 0.0, 1.0).clamp(0.0, 1.0)
relief = (relief * hill).power(1.4)

terrain = GeoMesh.plane(width=160, depth=160, segments=96).displace(relief, amount=11.0)
print(terrain.positions.shape, terrain.faces.shape)
terrain = terrain.decimate(grid=58)
print(terrain.positions.shape, terrain.faces.shape)

elevation = fields.coord("y").remap(-0.2, 8.0, 0.0, 1.0)
terrain = terrain.color_by(
    elevation,
    Gradient(
        [
            (0.00, "#5b6942"),
            (0.16, "#6b7749"),
            (0.40, "#7d7655"),
            (0.68, "#8b8370"),
            (0.90, "#c5bfb0"),
        ]
    ),
)
engine.spawn(
    Mesh(terrain.to_geometry()),
    Material(StandardMaterial(color="#ffffff", roughness=0.95, vertex_colors=True)),
    Transform(),
)
engine.camera.set_pose(position=(55, 60, 55), target=(0, 0, 0))
Image.fromarray(engine.render_frame(supersample=1))
```

The relief reaches 6.4 m, and the mask keeps it at zero within 24 m of the
fountain, so the square sits in a bowl of hills. The grid decimation takes 9,409
vertices and 18,432 triangles down to 4,091 and 7,941. At this size the far hills
are faceted and the near ground is not, which is the trade a coarse grid makes.

There are more sources than the ones here: `perlin` for one octave, `billow` for
puffy dunes, `constant` for a flat value. Every field here is deterministic from
its seed, so the plaza renders the same on every run.

## The finished program

The pieces are built the same way in one program: paving and terrain first, then
the hall, then the dome. Nothing new is introduced; the same operators are
composed in one scene.

```python {run export=examples/buildings.py}
import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling import Gradient, Mesh as GeoMesh, fields
from manifoldx.resources import DirectionalLight, StandardMaterial

engine = mx.Engine("plaza", width=480, height=270)
engine.background_color = (0.62, 0.72, 0.84)
engine.set_sun(DirectionalLight(color="#fff3dd", intensity=3.2, direction=(-0.45, -0.85, -0.3)))
engine.enable_shadows(resolution=1024, bias=0.004, pcf_radius=2)


def along_z(mesh):
    """Turn a mesh a quarter turn about x, so a y-axis cylinder runs along z."""
    x, y, z = mesh.positions.T
    return mesh.with_positions(np.stack([x, -z, y], axis=1))


# The paving: two noise fields settle it, and a cellular field shades its stones.
settle = fields.fbm(seed=4, freq=0.12, octaves=3) * 0.05
wear = fields.fbm(seed=11, freq=0.9, octaves=3) * 0.03
paving = GeoMesh.plane(width=40, depth=40, segments=200).displace(settle + wear, amount=1.0)
stones = fields.worley(seed=3, freq=0.6, feature="f2f1")
paving = paving.color_by(
    stones,
    Gradient([(0.00, "#5f584c"), (0.12, "#8d8472"), (0.35, "#c3b7a0"), (0.85, "#d8cdb5")]),
)
engine.spawn(
    Mesh(paving.to_geometry()),
    Material(StandardMaterial(color="#ffffff", roughness=0.95, vertex_colors=True)),
    Transform(),
)

# The countryside: ridges and roughness, held flat over the plaza.
relief = (
    fields.ridged(seed=5, freq=0.035, octaves=4) * 0.6
    + fields.fbm(seed=7, freq=0.18, octaves=3) * 0.4
).warp(8.0, fx=fields.fbm(seed=2, freq=0.05), fz=fields.fbm(seed=9, freq=0.05))
hill = fields.distance().remap(24.0, 64.0, 0.0, 1.0).clamp(0.0, 1.0)
relief = (relief * hill).power(1.4)
terrain = GeoMesh.plane(width=160, depth=160, segments=96).displace(relief, amount=11.0)
terrain = terrain.decimate(grid=58)
elevation = fields.coord("y").remap(-0.2, 8.0, 0.0, 1.0)
terrain = terrain.color_by(
    elevation,
    Gradient(
        [
            (0.00, "#5b6942"),
            (0.16, "#6b7749"),
            (0.40, "#7d7655"),
            (0.68, "#8b8370"),
            (0.90, "#c5bfb0"),
        ]
    ),
)
engine.spawn(
    Mesh(terrain.to_geometry()),
    Material(StandardMaterial(color="#ffffff", roughness=0.95, vertex_colors=True)),
    Transform(pos=(0, -0.4, 0)),
)
```

```python {run continue export=examples/buildings.py}
# The facade: a wall with a cornice, two arched doors and a round window.
wall = GeoMesh.box(16, 10, 0.6).translate((0, 5, 0))
top = wall.positions[wall.faces].mean(axis=1)[:, 1] > 9.99
cornice = wall.extrude(top, 0.7)

door = GeoMesh.box(1.9, 2.7, 2.0).translate((-3.6, 1.35, 0))
arch = along_z(GeoMesh.cylinder(radius=0.95, height=2.0, segments=24))
doorway = door.union(arch.translate((-3.6, 2.7, 0)))
window = along_z(GeoMesh.cylinder(radius=0.95, height=2.0, segments=24))
cutter = doorway.union(doorway.translate((7.2, 0, 0))).union(window.translate((0, 7.1, 0)))
facade = cornice.difference(cutter).translate((0, 0, -20.0))
engine.spawn(
    Mesh(facade.to_geometry()),
    Material(StandardMaterial(color="#cfc3a6", roughness=0.9)),
    Transform(),
)

# The hall behind it, and the drum the dome sits on.
body = GeoMesh.box(16, 10, 8).translate((0, 5, -25.0))
drum = GeoMesh.cylinder(radius=4.05, height=3.0, segments=40).translate((0, 11.5, -24.5))
engine.spawn(
    Mesh(body.to_geometry()),
    Material(StandardMaterial(color="#c9bda0", roughness=0.9)),
    Transform(),
)
engine.spawn(
    Mesh(drum.to_geometry()),
    Material(StandardMaterial(color="#c9bda0", roughness=0.9)),
    Transform(),
)
```

```python {run continue export=examples/buildings.py}
# The dome: a dented, smoothed sphere, cut in half, welded and crowned.
sphere = GeoMesh.icosphere(subdivisions=1, radius=4.2).subdivide(iterations=1, scheme="loop")
sphere = sphere.inflate(center=(3.0, 2.4, 1.6), radius=2.0, strength=-0.35)
sphere = sphere.smooth(iterations=3, strength=0.5, center=(3.0, 2.4, 1.6), radius=2.8)
dome = sphere.intersection(GeoMesh.box(20, 20, 20).translate((0, 10, 0)))
dome = dome.decimate(grid=128)
lattice = dome.ffd(resolution=(3, 3, 3))
for i in range(4):
    for k in range(4):
        lattice.move(i, 3, k, (0.0, 0.25, 0.0))
dome = lattice.apply().translate((0, 13.0, -24.5))
engine.spawn(
    Mesh(dome.to_geometry()),
    Material(StandardMaterial(color="#7d9b8c", roughness=0.4, metallic=0.25)),
    Transform(),
)

engine.camera.set_pose(position=(7, 6, 4), target=(0, 6, -20))
frame = engine.render_frame(supersample=1)
print(frame.shape, frame.dtype)
Image.fromarray(frame)
```

The facade faces the square, the hall gives it depth, and the dome rises above
the roof on its drum. The paving starts at the foot of the wall and runs to the
edge of the frame; the hills carry the eye past it. Every one of those shapes is
a `Mesh` value that was built, cut, worn or displaced in code, and the engine
only ever saw the geometry dicts at the end.

To walk around it in a window instead, install the `desktop` extra and finish the
program with `engine.run()`; `render_frame` is what keeps this chapter readable
on a machine without a screen.

## A car at the kerb

Some objects come in families: every car has wheels, a hood, a windscreen, doors
and a hatch, and two cars differ in measurements, not in kind. For those,
`manifoldx.modeling.assets` builds the object from named parameters. `CarParams`
holds a car's measurements, the ones a spec sheet or a photo gives: length,
width, height, wheelbase, overhang, wheel radius, the heights of the nose and the
tail, where the hood meets the windscreen and where the roof ends, plus a few
choices such as the headlamp style or a spoiler. `build_car` turns them into a
`Car`, a list of `Part`s.

The body is a loft of cross-sections along the car, each one a fixed template
scaled by smooth curves through those measurements, and the wheel arches are
raised into it, so no boolean is involved. Glass, pillars and doors are tags on
the loft's faces in its own coordinates, so they follow any shape. Lamps, the
grille and the plates are outlines projected onto the skin. Doors, hood and
hatch are separate parts on hinges, and the wheels turn about their hubs:
`Car.pose` gives each part's transform, `spawn_car` puts every piece in the
engine and `place_car` moves and poses them, here with the doors and the hatch
open and the front wheels turned.

```python {run export=examples/procedural_car.py}
from PIL import Image

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling.assets.car import CarParams, build_car, place_car, spawn_car
from manifoldx.resources import DirectionalLight, StandardMaterial, plane

engine = mx.Engine("car", width=480, height=270)
engine.background_color = (0.62, 0.72, 0.84)
engine.set_sun(DirectionalLight(color="#fff3dd", intensity=3.0, direction=(-0.5, -0.8, -0.35)))
engine.enable_shadows(resolution=1024, bias=0.004, pcf_radius=2)
engine.spawn(Mesh(plane(30, 30)), Material(StandardMaterial(color="#bdbdbd", roughness=0.95)),
             Transform(rot=(-0.70710678, 0.0, 0.0, 0.70710678)))

car = build_car(CarParams(length=4.03, width=1.81, height=1.57, wheelbase=2.66, headlamp="almond"))
print([part.name for part in car.parts])
print(car.part("door_fr").kind, car.part("wheel_fl").steer)

handles = spawn_car(engine, car)
place_car(handles, car, at=(0, 0, 0), yaw=0.0, doors=1.0, hatch=0.6, steer=0.35)

engine.camera.set_pose(position=(3.6, 1.8, 3.4), target=(0.0, 0.7, 0.0))
Image.fromarray(engine.render_frame(supersample=1))
```

Every number in `CarParams` is a dial: a longer wheelbase moves the rear axle and
its arch, a lower `tail_top` rakes the hatch, `headlamp="round"` swaps the lamp
outline. Detail comes later, from smoothing and subdivision on the same parts.

## Assets from an image model

Not every part wants a builder with twenty dials. A wheel face, a grille, a
lamp or a dashboard is a drawing: flat regions of colour on white, which an
image model can produce from a sentence. `manifoldx.image` and
`manifoldx.modeling.trace` turn such a drawing into a `Shape`, a stack of named
`Layer` values that the modelling tools can stretch, mirror and extrude.

`read_png` decodes an 8-bit, non-interlaced PNG — RGB, RGBA, grey, grey+alpha
or palette — into a numpy array, and `write_png` writes RGB or RGBA back, so a
drawing and a render can travel without an imaging library. `resize` resamples
an image bilinearly, or with `nearest=True` when the pixels are labels and
interpolating them would invent colours. `pad_to_aspect` centres a drawing on a
canvas of a named ratio from `ASPECTS`, returning the padded image and the
offset of the original inside it; that is how a view reaches an image model at
the ratio it expects, and how its answer is cropped back.

Tracing starts from a `palette`, a mapping from material name to RGB, and
`classify` labels every pixel with the index of its nearest palette colour.
`dilate` and `erode` grow and shrink a mask with a square or disc element,
`open_mask` and `close_mask` compose them to remove specks or fill pinholes, and
`components` numbers the connected blobs of a mask (4- or 8-connected).
`contours` returns the outer boundary of every component as a closed polygon in
pixel coordinates, counter-clockwise with y up; with `holes=True` it also
returns the holes, wound the other way. `simplify` applies Douglas-Peucker to a
closed polygon, so a traced circle drops from hundreds of points to the few
that carry its shape. `cut_by_rays` removes a thin strip along rays from a
centre, which opens the round windows of a wheel face into the outside and
leaves pieces without holes; without that, a region with a window cannot be
extruded as a solid face.

`trace_layers` is the whole pipeline: classify the image, take the materials in
`order` (largest region first), contour and simplify each, turn a region's holes
into layers of the material underneath, and map the drawing's frame — its
bounding box, or one given — to `-1 .. 1` on its longer side, centred and y up.
The result is a `Shape` in units of the part's half-size, with the outer side
at `+u`. `redraw` rasterises a `Shape` back to an image, even-odd filling every
polygon layer by layer, and `iou` compares two masks, so a trace is checked
against its own drawing before it is used.

A `Shape` is plain data. `to_json` and `from_json` round-trip it through text,
and `save` and `load` do the same through a file, so traced artwork lives beside
the code rather than inside it. `bounds` gives its `(u_min, v_min, u_max,
v_max)` box. `mirror` flips it across `u = 0` and keeps every polygon
counter-clockwise — the first thing to do to a drawing seen from behind.
`mirror_half` keeps one side of a symmetric drawing and reflects it, which
repairs an image model's small asymmetries; `stretch_middle` lengthens a shape
by a ratio while holding the outer fraction at each end exactly as drawn, so
slats and handles keep their proportions; `fit_box` scales and centres it to
fill a box.

```python
import numpy as np

from manifoldx.image import pad_to_aspect, read_png
from manifoldx.modeling.shape import stretch_middle
from manifoldx.modeling.trace import iou, redraw, trace_layers

palette = {"white": (255, 255, 255), "steel": (70, 100, 140), "brass": (190, 150, 60)}
drawing = read_png("wheel.png")
canvas, (x0, y0) = pad_to_aspect(drawing, "1:1", fill=255)
wheel = trace_layers(drawing, palette, ["steel", "brass"], background="white")
wheel = stretch_middle(wheel, 1.1, ends=0.2)
check = redraw(wheel, (drawing.shape[1], drawing.shape[0]), palette)
print(iou(np.any(check != 255, axis=-1), np.any(drawing != 255, axis=-1)))
wheel.save("wheel.json")
```

A traced shape becomes a solid, or a set of views becomes a whole part.
`extrude` sweeps a closed outline in the XY plane along z into a prism, its
two caps triangulated by ear clipping, which also handles a concave outline
such as a grille. When there is a front drawing and a side drawing instead,
`two_view_solid` lofts between them: both masks share their rows, each of `n`
heights gives an x interval from the front mask and a z interval from the
left-side mask, and the section is a superellipse inscribed in that rectangle,
with `power=2` an ellipse and larger values squarer. `height_range`,
`x_range` and `z_range` say which metres the masks cover — row 0 is the top,
the front mask's first column is `x_min`, and the side mask's first column is
`z_max` because the object faces the front-view camera, so its side image's
columns run toward `-z`. A section spans the first to the last drawn pixel of
a row, not its widest unbroken run: a part drawn on top of another leaves a
hole inside the silhouette, and a hole must not pinch the solid. Heights where
either mask is empty are skipped. `silhouette` projects a mesh back into a
view's mask, so the solid can be compared with the drawing it came from.

A textured part needs somewhere to put the picture. `view_atlas` takes parts
named in a mapping, sends every face to the view among `views` whose direction
is closest to its normal, and projects its vertices orthographically in that
view. The views become `Island` values in an `Atlas`: each island holds its
rectangle in pixels, the view's frame in metres, and the atlas holds the
metres per pixel it used. All islands share that scale unless `texel` asks for
another, so texel density is even across the part. The `Atlas` also carries a
split mesh per part, with a vertex duplicated where its faces land in different
islands, so every vertex has one UV. `template` draws one island flat: it
rasterises the island's triangles with a depth buffer, nearest face winning,
and returns the image and an integer map of the material index, white outside.
That image is what an image model is asked to paint.

`paste_island` brings the answer back: it resizes the model's picture to the
padded canvas, crops the island out at the offset `pad_to_aspect` returned and
writes it into the atlas rectangle. `kept_outline` compares the non-white
pixels of the answer with the template as an IoU, so a model that moved or
lost the outline is caught before it is used. `clean` repairs a painted
island: a white hole left inside a region becomes that region's flat colour,
and `tone` scales the whole island until the median of one region reaches the
colour asked for. `bleed` grows each region's colours into its background by a
few pixels, which keeps a seam filter from mixing in white. `normal_from_albedo`
derives a tangent-space normal map from the picture itself: a thin line darker
than its surroundings becomes a groove, and its smoothed relief differentiates
into tilted normals, flat blue where the picture is flat. `flip_normal_x`
inverts the red channel for an island mirrored across its seam.

```python
from manifoldx.modeling.paint import bleed, clean, kept_outline, normal_from_albedo, paste_island
from manifoldx.modeling.views import extrude, silhouette, template, two_view_solid, view_atlas

solid = two_view_solid(front, side, height_range=(0.0, 0.9),
                       x_range=(-0.6, 0.6), z_range=(-0.4, 0.4))
atlas = view_atlas({"body": solid}, ["front", "back", "left", "right", "top"],
                   size=(1024, 1024))
paint, regions = template({"body": solid}, atlas, {"body": (190, 190, 195)}, "front")
print(kept_outline(paint, paint))
```
