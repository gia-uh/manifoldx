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
paving = GeoMesh.plane(width=40, depth=40, segments=128).displace(settle + wear, amount=1.0)
stones = fields.worley(seed=3, freq=1.6, feature="f2f1")
paving = paving.color_by(
    stones,
    Gradient([(0.00, "#6f685c"), (0.05, "#9a907e"), (0.30, "#c3b7a0"), (0.80, "#cfc4ac")]),
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
# The dome: a dented, smoothed sphere, cut in half, welded and crowned.
sphere = GeoMesh.icosphere(subdivisions=1, radius=4.2).subdivide(iterations=1, scheme="loop")
sphere = sphere.inflate(center=(3.0, 2.4, 1.6), radius=1.4, strength=-0.45)
sphere = sphere.smooth(iterations=2, strength=0.6, center=(3.0, 2.4, 1.6), radius=1.9)
dome = sphere.intersection(GeoMesh.box(20, 20, 20).translate((0, 10, 0)))
dome = dome.decimate(grid=128)
lattice = dome.ffd(resolution=(3, 3, 3))
for i in range(4):
    for k in range(4):
        lattice.move(i, 3, k, (0.0, 0.85, 0.0))
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
