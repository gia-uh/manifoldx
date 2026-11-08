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
bad = work / "bench_bad.obj"
bad.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 -3\n")
try:
    load_obj(bad)
except ObjParseError as error:
    print(f"ObjParseError: {error}")
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
data, report = build_glb([Node(bench, wood, pos=(-4, 0, -3), name="bench")])
print(f"bench.glb: {len(data)} bytes, header {data[:4]}, {report.format()}")
(work / "bench.glb").write_bytes(data)
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
try:
    engine.export_gltf(work / "strict.glb", strict=True)
except ExportIncomplete as error:
    print(f"strict export refused: {error}")
print(f"strict file written: {(work / 'strict.glb').exists()}")
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
streamer = Streamer(
    engine, CELL, build_cell, radius=2, ready_radius=2, workers="thread", uploads_per_frame=25
)
for x in (0.0, 0.0, 0.0, 20.0, 20.0, 0.0, 0.0):
    streamer.focus(x, 0.0)
    frame = engine.render_frame(supersample=1)
print(f"cells drawn: {len(streamer.drawn())}, stats: {streamer.stats()}")
streamer.close()
Image.fromarray(frame)
