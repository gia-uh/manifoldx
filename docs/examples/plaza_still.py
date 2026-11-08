import tempfile
from pathlib import Path

import manifoldx as mx
import numpy as np
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane, sphere
from manifoldx.trace import Camera, Environment, Scene, Tracer
from PIL import Image

FLOOR = Transform.rotation(x=-np.pi / 2)

engine = mx.Engine("plaza-cover", width=480, height=270)
engine.set_sun(DirectionalLight(color="#ffe6c4", intensity=3.0, direction=(-0.45, -0.30, 0.45)))

paving = StandardMaterial(color="#b9ad98", roughness=0.9)
stone = StandardMaterial(color="#a8a296", roughness=0.7)
dark = StandardMaterial(color="#5d5850", roughness=0.8)
wood = StandardMaterial(color="#7b5a3a", roughness=0.6)
water = StandardMaterial(color="#4f7385", roughness=0.02, metallic=1.0)
bronze = StandardMaterial(color="#7a5a2e", roughness=0.3, metallic=1.0)
glass = StandardMaterial(color="#2b3440", roughness=0.12, metallic=0.5)
plaster = StandardMaterial(color="#c2b49c", roughness=0.95)

engine.spawn(Mesh(plane(40, 40)), Material(paving), Transform(rot=FLOOR))

for half in (9.0, 13.0):
    for side in (-1, 1):
        engine.spawn(
            Mesh(cube(2 * half + 0.25, 0.02, 0.25)),
            Material(dark),
            Transform(pos=(0, 0.01, side * half)),
        )
        engine.spawn(
            Mesh(cube(0.25, 0.02, 2 * half + 0.25)),
            Material(dark),
            Transform(pos=(side * half, 0.01, 0)),
        )

engine.spawn(Mesh(cube(4.0, 1.0, 0.3)), Material(stone), Transform(pos=(0, 0.5, -1.85)))
engine.spawn(Mesh(cube(4.0, 1.0, 0.3)), Material(stone), Transform(pos=(0, 0.5, 1.85)))
engine.spawn(Mesh(cube(0.3, 1.0, 3.4)), Material(stone), Transform(pos=(-1.85, 0.5, 0)))
engine.spawn(Mesh(cube(0.3, 1.0, 3.4)), Material(stone), Transform(pos=(1.85, 0.5, 0)))
engine.spawn(Mesh(cube(3.4, 0.12, 3.4)), Material(dark), Transform(pos=(0, 0.06, 0)))
engine.spawn(
    Mesh(plane(3.4, 3.4)),
    Material(water),
    Transform(pos=(0, 0.995, 0), rot=FLOOR),
)
engine.spawn(Mesh(cube(0.8, 1.6, 0.8)), Material(stone), Transform(pos=(0, 0.9, 0)))
engine.spawn(Mesh(sphere(0.55)), Material(bronze), Transform(pos=(0, 2.25, 0)))


def bench(engine, at, yaw, stone, wood):
    """A seat, a back and two legs, turned to face the fountain."""
    x, z = at
    rot = Transform.rotation(y=yaw)
    along_x, along_z = np.cos(yaw), -np.sin(yaw)
    out_x, out_z = np.sin(yaw), np.cos(yaw)
    engine.spawn(Mesh(cube(1.9, 0.1, 0.55)), Material(wood), Transform(pos=(x, 0.46, z), rot=rot))
    engine.spawn(
        Mesh(cube(1.9, 0.55, 0.09)),
        Material(wood),
        Transform(pos=(x + 0.24 * out_x, 0.74, z + 0.24 * out_z), rot=rot),
    )
    for side in (-0.8, 0.8):
        engine.spawn(
            Mesh(cube(0.12, 0.46, 0.5)),
            Material(stone),
            Transform(pos=(x + side * along_x, 0.23, z + side * along_z), rot=rot),
        )


def building(engine, center, size, normal, plaster, stone, glass, window):
    """A block with a plinth, a cornice and a grid of windows on its plaza face."""
    x, _, z = center
    width, height, depth = size
    normal_x, normal_z = normal
    engine.spawn(
        Mesh(cube(width, height, depth)), Material(plaster), Transform(pos=(x, height / 2, z))
    )
    engine.spawn(
        Mesh(cube(width + 0.4, 0.3, depth + 0.4)),
        Material(stone),
        Transform(pos=(x, height + 0.15, z)),
    )
    engine.spawn(
        Mesh(cube(width + 0.3, 0.5, depth + 0.3)),
        Material(stone),
        Transform(pos=(x, 0.25, z)),
    )
    length = depth if normal_z else width
    columns = max(3, int(length // 2.6))
    for column in range(columns):
        offset = (column - (columns - 1) / 2) * (length / columns)
        for row in (3.4, 5.4, 7.4):
            if row > height - 1.5:
                continue
            if normal_z:
                engine.spawn(
                    Mesh(window),
                    Material(glass),
                    Transform(pos=(x + offset, row, z + normal_z * depth / 2)),
                )
            else:
                engine.spawn(
                    Mesh(window),
                    Material(glass),
                    Transform(
                        pos=(x + normal_x * width / 2, row, z + offset),
                        rot=Transform.rotation(y=np.pi / 2),
                    ),
                )
    for door in range(3):
        offset = (door - 1) * (length / 3.4)
        if normal_z:
            engine.spawn(
                Mesh(cube(1.5, 2.7, 0.2)),
                Material(glass),
                Transform(pos=(x + offset, 1.35, z + normal_z * depth / 2)),
            )
        else:
            engine.spawn(
                Mesh(cube(0.2, 2.7, 1.5)),
                Material(glass),
                Transform(pos=(x + normal_x * width / 2, 1.35, z + offset)),
            )


for corner in range(4):
    angle = corner * np.pi / 2
    bench(engine, (4.8 * np.sin(angle), 4.8 * np.cos(angle)), angle, stone, wood)

window = cube(1.2, 1.9, 0.16)
building(engine, (-13.5, 0, -22.5), (12, 9, 5), (0, 1), plaster, stone, glass, window)
building(engine, (0, 0, -22.5), (12, 10, 5), (0, 1), plaster, stone, glass, window)
building(engine, (13.5, 0, -22.5), (12, 8, 5), (0, 1), plaster, stone, glass, window)
building(engine, (-22.5, 0, -5), (5, 8, 18), (1, 0), plaster, stone, glass, window)
building(engine, (22.5, 0, -6), (5, 9, 16), (-1, 0), plaster, stone, glass, window)
engine.spawn(Mesh(cube(40, 1.0, 0.4)), Material(stone), Transform(pos=(0, 0.5, 20)))

path = Path(tempfile.gettempdir()) / "plaza-cover.glb"
report = engine.export_gltf(path)
scene = Scene.from_glb(path)

print(len(report), "things left out")
print(scene.triangle_count, len(scene.materials), len(scene.lights))
print(tuple(round(v, 1) for v in scene.bounds()[1]))
print([light.kind for light in scene.lights])
water_surface = next(m for m in scene.materials if m.roughness < 0.05)
print(
    water_surface.name, water_surface.metallic, tuple(round(c, 2) for c in water_surface.base_color)
)
sun = scene.lights[0]
sun.color = (1.0, 0.86, 0.68)
sun.intensity = 20.0

sun_direction = tuple(-v for v in sun.direction)
sky = Environment.sky(sun_direction, turbidity=2.4, sun_ratio=0.0, ground=(0.34, 0.30, 0.24))

camera = Camera(position=(6.2, 2.3, 7.6), target=(0, 1.2, -1.5), fov=48.0)
tracer = Tracer(scene, 480, 270, camera=camera, environment=sky)
print(type(tracer.bvh).__name__, tracer.bvh.node_count, tracer.bvh.depth())
print(f"built in {tracer.build_seconds:.3f} s, {tracer.width}x{tracer.height}")
seconds = tracer.step(8)
print(f"{tracer.samples} samples per pixel in {seconds:.1f} s")
Image.fromarray(tracer.image(denoise=False))
seconds = tracer.step(8)
print(f"{tracer.samples} samples per pixel in {seconds:.1f} s")
noisy = tracer.image(denoise=False)
smooth = tracer.image()
Image.fromarray(np.concatenate([noisy, smooth], axis=1))
seconds = tracer.step(16)
still = tracer.image()
print(f"{tracer.samples} samples per pixel in {seconds:.1f} s")
Image.fromarray(still)
