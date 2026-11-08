# Assets in and out

A designer sends the plaza's stone bench as an OBJ file. The colleague who will
show the finished square to the town council uses a viewer that reads glTF, so
the plaza has to leave the same way. And the district around the square is far
too large to keep in memory at once: it has to arrive tile by tile as the
camera moves.

## The file from the designer

Wavefront OBJ is a text format: `v` lines are positions, `vn` lines are
normals, `vt` lines are texture coordinates, and `f` lines join them into
faces. Numbers on a face are 1-indexed references into those lists. That is
almost the geometry dictionary the engine already draws, so `load_obj` reads a
file and gives you the dictionary directly: `positions`, `indices`, and
`normals` or `uvs` when the file carries them.

The file below is the whole bench, written out so the chapter needs no binary
asset. It is two boxes: a seat and the plinth under it. The faces use the
`v//vn` form, and every box face has its own normal so the corners stay sharp.

```python {run export=examples/plaza_assets.py}
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.assets import ObjParseError, load_obj
from manifoldx.components import Material, Mesh, Transform
from manifoldx.gltf import ExportIncomplete, Node, build_glb
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane
from manifoldx.streaming import Streamer
from manifoldx.workers import ProcessStream

BENCH_OBJ = """\
# stone bench, metres, y up
o seat
v -0.9 0.40 -0.25
v 0.9 0.40 -0.25
v 0.9 0.52 -0.25
v -0.9 0.52 -0.25
v -0.9 0.40 0.25
v 0.9 0.40 0.25
v 0.9 0.52 0.25
v -0.9 0.52 0.25
vn 0 0 -1
vn 0 0 1
vn -1 0 0
vn 1 0 0
vn 0 -1 0
vn 0 1 0
f 5//2 6//2 7//2 8//2
f 2//1 1//1 4//1 3//1
f 2//4 6//4 7//4 3//4
f 1//3 5//3 8//3 4//3
f 4//6 8//6 7//6 3//6
f 1//5 2//5 6//5 5//5
o base
v -0.75 0.00 -0.20
v 0.75 0.00 -0.20
v 0.75 0.40 -0.20
v -0.75 0.40 -0.20
v -0.75 0.00 0.20
v 0.75 0.00 0.20
v 0.75 0.40 0.20
v -0.75 0.40 0.20
f 13//2 14//2 15//2 16//2
f 10//1 9//1 12//1 11//1
f 10//4 14//4 15//4 11//4
f 9//3 13//3 16//3 12//3
f 12//6 16//6 15//6 11//6
f 9//5 10//5 14//5 13//5
"""

work = Path(tempfile.mkdtemp(prefix="plaza-assets-"))
bench_path = work / "bench.obj"
bench_path.write_text(BENCH_OBJ)
bench = load_obj(bench_path)
print(f"bench: {len(bench['positions'])} vertices, {len(bench['indices']) // 3} triangles")
print(f"bounds: {bench['positions'].min(axis=0)} to {bench['positions'].max(axis=0)}")
```

Sixteen corners would be enough to describe two boxes, but the file reports 48
vertices: `load_obj` splits a corner for every distinct position, normal and UV
tuple, because a vertex buffer carries one normal per vertex. The twelve quads
become 24 triangles, and `indices` is one flat `uint32` array. Polygons with
more than three corners are fan-triangulated the same way.

A file that asks for something the parser cannot do stops with `ObjParseError`
and says where. It supports the four face forms (`v`, `v/vt`, `v/vt/vn` and
`v//vn`), but all faces in a file must use the same one, and indices must be
positive: relative (negative) indices are not supported in this version. A
typo in the file names the line instead of producing a mystery mesh.

```python {run continue export=examples/plaza_assets.py}
bad = work / "bench_bad.obj"
bad.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 -3\n")
try:
    load_obj(bad)
except ObjParseError as error:
    print(f"ObjParseError: {error}")
```

## The bench in the plaza

The dictionary goes straight into `Mesh`; `Material` gives it a surface and a
`Transform` puts it on the ground. The plaza's forty-metre floor, its fountain
and four benches are the same vocabulary as the earlier chapters. The camera
looks at the nearest bench so the asset the designer sent is visible before it
becomes one small object in a district.

```python {run continue export=examples/plaza_assets.py}
engine = mx.Engine("plaza assets", width=480, height=270)
engine.background_color = (0.72, 0.78, 0.86)
engine.set_sun(DirectionalLight(color="#fff4e0", intensity=3.0, direction=(-0.4, -1, -0.3)))

FLOOR = Transform.rotation(x=-np.pi / 2)
engine.spawn(
    Mesh(plane(40, 40)),
    Material(StandardMaterial("#c8b89a", roughness=0.9)),
    Transform(pos=(0, -0.01, 0), rot=FLOOR),
)
engine.spawn(
    Mesh(cube(4, 1, 4)),
    Material(StandardMaterial("#8a9bb0", roughness=0.55)),
    Transform(pos=(0, 0.5, 0)),
)

wood = StandardMaterial("#8a5a2b", roughness=0.7)
for x, z in ((-4, -3), (4, -3), (-4, 3), (4, 3)):
    engine.spawn(Mesh(bench), Material(wood), Transform(pos=(x, 0, z)))

engine.camera.set_pose((-1.4, 1.1, 5.2), (-4, 0.35, 3))
Image.fromarray(engine.render_frame(supersample=1))
```

The floor is a hair below zero so the streamed ground of the next section can
cover it without fighting for the same depth. The bench sits on the plaza at
(-4, 0, 3), and the fountain is the pale block on the right.

## The plaza leaves as glTF

glTF is the file format viewers, game engines and Blender agree on. The binary
form, `.glb`, is one file with a JSON document and its buffers side by side;
`GlbBuilder` packs those two chunks with their four-byte alignment, and
`build_glb` fills the document with meshes, materials and nodes. A `Node` is
one placed mesh: the geometry dictionary or a `modeling.Mesh`, a material, a
translation, a rotation, a scale and a name. `export_gltf` writes the bytes;
`engine.export_gltf` is the same for a whole engine scene.

The bench alone goes back to the designer this way. `build_glb` returns the
bytes and a report, so a single asset can leave without an engine.

```python {run continue export=examples/plaza_assets.py}
data, report = build_glb([Node(bench, wood, pos=(-4, 0, -3), name="bench")])
print(f"bench.glb: {len(data)} bytes, header {data[:4]}, {report.format()}")
(work / "bench.glb").write_bytes(data)
```

The header is the four bytes `glTF`; the rest is the JSON and the interleaved
vertex buffer. The report is empty because a mesh, a standard material and a
node all have an exact glTF equivalent.

A whole scene has more in it than meshes, and not all of it can leave. The
walkers, the event handlers and the widgets are Python code, not scene data;
they keep running in ManifoldX but a glTF viewer cannot run them. Instead of
dropping them silently, `engine.export_gltf` returns an `ExportReport`. Each
`Entry` says what was left out, whether it was dropped or approximated, why,
and where in your code it lives. The same entries are printed to standard
error and written to a `.export-report.json` next to the `.glb`.

```python {run continue export=examples/plaza_assets.py}
visitor = engine.spawn(
    Mesh(cube(0.5, 1.7, 0.5)),
    Material(StandardMaterial("#b04a3a", roughness=0.7)),
    Transform(pos=(-6, 0.85, 0)),
)


@engine.system
def walk(query, dt):
    visitor.transform.pos = (-6 + 3 * (engine.elapsed % 4), 0.85, 0)


plaza_path = work / "plaza.glb"
report = engine.export_gltf(plaza_path)
print(f"plaza.glb: {plaza_path.stat().st_size} bytes, {len(report)} entries")
for entry in report:
    print(f"  {entry.kind:<8} {entry.name}: {entry.effect} ({entry.detail})")
print(f"report file: {(work / 'plaza.export-report.json').exists()}")
```

The one entry is the `walk` system: custom Python that runs every frame. The
mesh, its material, the camera, the sun and the fountain are all in the file;
the visitor is exported at the position it had when the export ran, and the
system that would move it is not. A viewer will show a still plaza, which is
what a glTF file is.

When a pipeline cannot accept a partial scene, ask for a strict export. It
raises `ExportIncomplete` with the report as its message and writes nothing, so
a build fails instead of shipping a file that is missing its animation.

```python {run continue export=examples/plaza_assets.py}
try:
    engine.export_gltf(work / "strict.glb", strict=True)
except ExportIncomplete as error:
    print(f"strict export refused: {error}")
print(f"strict file written: {(work / 'strict.glb').exists()}")
```

## The district streams in

The plaza is forty metres. The district around it is a hundred and forty, and
keeping its buildings, pavements and roofs as entities would cost memory for
geometry the camera never sees. A `Streamer` cuts the ground into square cells
of side `cell`; cell `(i, j)` covers `x` from `i * cell` to `(i + 1) * cell`
and `z` from `j * cell` to `(j + 1) * cell`. It asks your `build` function for
one cell at a time and draws only the cells near a focus point.

The builder returns a `MeshData`: positions, normals, colours and indices, all
relative to the cell's corner, so the same tile can be moved anywhere without
rebuilding it. Colours are linear RGB, and the streamer's default material
reads them per vertex. The builder lives in a module of its own because a child
process has to import its functions by name.

```python {run continue export=examples/plaza_worker.py}
"""The district's survey and its tile builder, importable by name."""

import numpy as np

from manifoldx.streaming import MeshData

CELL = 20.0
SEGMENTS = 4
PLAN = {}  # (i, j) -> {"height": metres, "tint": linear rgb}, filled by the survey


def district_rows(j_values, i_values):
    """One row of the district's plan per yield; the work stays off the frame."""
    for j in j_values:
        row = {}
        for i in i_values:
            sample = np.random.default_rng(1_000 * (i + 10) + (j + 10)).random((64, 64))
            inside_plaza = i in (-1, 0) and j in (-1, 0)
            row[(i, j)] = {
                "height": 0.0 if inside_plaza else 6.0 + 10.0 * float(sample.mean()),
                "tint": tuple(float(c) for c in sample.mean(axis=0)[:3]),
            }
        yield row


def _box(cx, cz, width, depth, height, tint):
    x0, x1 = cx - width / 2, cx + width / 2
    z0, z1 = cz - depth / 2, cz + depth / 2
    faces = [
        ([(x0, 0, z1), (x1, 0, z1), (x1, height, z1), (x0, height, z1)], (0, 0, 1)),
        ([(x1, 0, z0), (x0, 0, z0), (x0, height, z0), (x1, height, z0)], (0, 0, -1)),
        ([(x1, 0, z1), (x1, 0, z0), (x1, height, z0), (x1, height, z1)], (1, 0, 0)),
        ([(x0, 0, z0), (x0, 0, z1), (x0, height, z1), (x0, height, z0)], (-1, 0, 0)),
        ([(x0, height, z1), (x1, height, z1), (x1, height, z0), (x0, height, z0)], (0, 1, 0)),
        ([(x0, 0, z0), (x1, 0, z0), (x1, 0, z1), (x0, 0, z1)], (0, -1, 0)),
    ]
    positions, normals, colors, indices = [], [], [], []
    for corners, normal in faces:
        base = len(positions)
        positions.extend(corners)
        normals.extend([normal] * 4)
        colors.extend([tint] * 4)
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return positions, normals, colors, indices


def build_cell(i, j):
    """Cell (i, j) as MeshData relative to its corner."""
    n = SEGMENTS + 1
    line = np.linspace(0.0, CELL, n)
    plot = PLAN.get((i, j), {"height": 0.0, "tint": (0.5, 0.5, 0.5)})
    inside_plaza = i in (-1, 0) and j in (-1, 0)
    ground = (0.36, 0.34, 0.30) if inside_plaza else (0.28, 0.26, 0.22)
    positions = [[x, 0.0, z] for z in line for x in line]
    normals = [[0.0, 1.0, 0.0] for _ in positions]
    colors = [ground for _ in positions]
    q = np.arange(SEGMENTS)
    a = (q[:, None] * n + q[None, :]).ravel()
    indices = np.stack([a, a + n, a + n + 1, a, a + n + 1, a + 1], -1).ravel().tolist()
    if plot["height"] > 0:
        tint = (
            0.26 + 0.10 * plot["tint"][0],
            0.22 + 0.08 * plot["tint"][1],
            0.18 + 0.06 * plot["tint"][2],
        )
        p, nr, c, idx = _box(CELL / 2, CELL / 2, CELL - 6, CELL - 6, plot["height"], tint)
        base = len(positions)
        positions += p
        normals += nr
        colors += c
        indices += [base + k for k in idx]
    return MeshData(
        np.asarray(positions, np.float32),
        np.asarray(normals, np.float32),
        np.asarray(colors, np.float32),
        np.asarray(indices, np.uint32),
    )
```

The four cells with `i` and `j` in `(-1, 0)` are the plaza itself: ground, no
building. Every other cell gets one block whose height and tint come from
`PLAN`. The plan is empty until the survey fills it, and `build_cell` falls
back to flat ground for a plot that was never surveyed.

## Where the plan comes from

A survey of the whole district is heavy pure Python: sampling each plot,
deciding its height, choosing its tint. Running that on the frame would stall
the camera, and a thread would not help because the work holds the GIL through
its Python-level parts. `ProcessStream` runs a generator function in a child
process and hands back one item per `take()`, so the frame loop unpickles one
row at a time and never waits for the whole survey.

The child is a fresh interpreter, not a spawn of your main script. It imports
the generator by module and name, which is why `district_rows` lives in
`plaza_worker.py` and not in the block that uses it. A function defined in
`__main__` cannot be imported by the child; `take()` raises a `RuntimeError`
with the child's traceback when that happens. A take that reaches the end
returns None and sets `done`; `close()` stops a worker that is still running.
For a single heavy result, `engine.submit_process` is the one-shot alternative;
for tiles that are built off the frame, `Streamer(workers="process")` uses the
same idea.

```python {run continue export=examples/plaza_assets.py}
try:  # the guide runs from docs/, where examples/ is a package
    from plaza_worker import CELL, PLAN, build_cell, district_rows
except ModuleNotFoundError:
    from examples.plaza_worker import CELL, PLAN, build_cell, district_rows

engine.camera.set_pose((38, 26, 45), (0, 1, 0))
survey = ProcessStream(district_rows, range(-3, 4), range(-3, 4))
rows = 0
while not survey.done:
    row = survey.take(0.1)
    if row is not None:
        PLAN.update(row)
        rows += 1
    engine.render_frame(supersample=1)
survey.close()
print(f"surveyed {len(PLAN)} plots in {rows} rows")
```

Each `take(0.1)` waits up to a tenth of a second, and the frame is drawn while
it waits. When a row arrives it goes into the plan; after the last row the
stream is done. The camera is now above the plaza, looking across the district.

## Drawing the tiles

`Streamer` keeps two rings around the focus. Cells within `ready_radius` are
built in the worker and uploaded to the GPU hidden; cells within `radius` are
drawn. It copies at most `upload_bytes_per_frame` of vertex data a frame, and
draws at most `uploads_per_frame` new cells, so a heavy tile takes several
frames instead of one long one. A cell that leaves `radius` stops drawing but
keeps its buffers; one that leaves `ready_radius` is dropped and its buffers
are freed. `focus(x, z)` moves the centre and is called once a frame;
`drawn()` reports the cells on screen and `stats()` counts what was built,
uploaded and dropped.

```python {run continue export=examples/plaza_assets.py}
streamer = Streamer(
    engine, CELL, build_cell, radius=2, ready_radius=2, workers="thread", uploads_per_frame=25
)
for x in (0.0, 0.0, 0.0, 20.0, 20.0, 0.0, 0.0):
    streamer.focus(x, 0.0)
    frame = engine.render_frame(supersample=1)
print(f"cells drawn: {len(streamer.drawn())}, stats: {streamer.stats()}")
streamer.close()
Image.fromarray(frame)
```

The focus steps one cell east and back, so the streamer builds a ring, draws it,
drops the cells that fell behind and brings them back. The frame is the
twenty-five cells around the plaza: the four open plaza cells in the middle and
twenty-one building plots with the heights the survey chose. The district is no
longer one mesh in memory; it is a window that follows the camera.

To open the finished plaza in a window, install the `desktop` extra and end the
program with `engine.run()`, which draws every frame until you close it.

```python
engine.run()
```

The next chapter stops moving and makes one still worth printing, by tracing
the paths of light through the same scene.
