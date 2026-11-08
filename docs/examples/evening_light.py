import numpy as np
from PIL import Image

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import (
    DirectionalLight,
    FlatMaterial,
    PointLight,
    SpotLight,
    StandardMaterial,
    cube,
    plane,
    sphere,
)

engine = mx.Engine("plaza", width=480, height=270)
engine.background_color = (0.06, 0.08, 0.14)
FLOOR = Transform.rotation(x=-np.pi / 2)

engine.spawn(
    Mesh(plane(40, 40)),
    Material(StandardMaterial(color="#5c5850", roughness=0.95)),
    Transform(pos=(0, -0.02, 0), rot=FLOOR),
)
engine.spawn(
    Mesh(cube(4, 1, 4)),
    Material(StandardMaterial(color="#a8a196", roughness=0.72)),
    Transform(pos=(0, 0.5, 0)),
)
engine.spawn(
    Mesh(plane(3.4, 3.4)),
    Material(StandardMaterial(color="#3e6a72", roughness=0.5)),
    Transform(pos=(0, 1.005, 0), rot=FLOOR),
)

engine.camera.set_pose(position=(6.0, 4.2, 7.5), target=(0.5, 0.9, 0.0))
engine.set_sun(DirectionalLight(color="#ffd0a0", intensity=3.2, direction=(-0.7, -0.3, 0.35)))
engine.enable_shadows(resolution=1024)
Image.fromarray(engine.render_frame())
stone_a = StandardMaterial(color="#a49d90", roughness=0.92)
stone_b = StandardMaterial(color="#969084", roughness=0.88)
slab = plane(1.9, 1.9)
for ix in range(-9, 10):
    for iz in range(-9, 10):
        engine.spawn(
            Mesh(slab),
            Material(stone_a if (ix + iz) % 2 == 0 else stone_b),
            Transform(pos=(ix * 2.0, 0.0, iz * 2.0), rot=FLOOR),
        )

frame = engine.render_frame()
geometry_id = engine._geometry_registry.register(slab)
print("slab geometry:", geometry_id, engine._geometry_registry.get(geometry_id) is slab)
material_id = engine._material_registry.register(stone_a)
print("stone material:", material_id, engine._material_registry.get(material_id) is stone_a)
print("entities:", len(engine.store.get_component_view(["Transform"])))
print("instances drawn:", engine._render_pipeline.drawn)
cache = engine._render_pipeline._transform_cache
print("cached matrices:", cache.get_transforms(engine.store, np.arange(4)).shape)
Image.fromarray(frame)
wood = StandardMaterial(color="#6b4a2f", roughness=0.7)
brass = StandardMaterial(color="#e8b860", metallic=1.0, roughness=0.05)
engine.spawn(
    Mesh(cube(0.7, 0.12, 2.4)),
    Material(wood),
    Transform(pos=(4.2, 0.48, 2.6)),
)
for dz in (-1.0, 1.0):
    engine.spawn(
        Mesh(cube(0.6, 0.48, 0.12)),
        Material(brass),
        Transform(pos=(4.2, 0.24, 2.6 + dz)),
    )
engine.spawn(
    Mesh(cube(0.08, 0.08, 2.4)),
    Material(brass),
    Transform(pos=(4.55, 0.9, 2.6)),
)
for dz in (-1.0, 1.0):
    engine.spawn(
        Mesh(cube(0.08, 0.5, 0.08)),
        Material(brass),
        Transform(pos=(4.55, 0.65, 2.6 + dz)),
    )

engine.camera.set_pose(position=(7.2, 1.8, 5.2), target=(4.2, 0.6, 2.6))
frame = engine.render_frame()
engine.camera.set_pose(position=(6.0, 4.2, 7.5), target=(0.5, 0.9, 0.0))
Image.fromarray(frame)
dark = StandardMaterial(color="#2a2c30", roughness=0.4, metallic=0.6)
engine.spawn(
    Mesh(cube(0.14, 4.2, 0.14)),
    Material(dark),
    Transform(pos=(-4.5, 2.1, 3.4)),
)
engine.spawn(
    Mesh(cube(0.9, 0.1, 0.1)),
    Material(dark),
    Transform(pos=(-4.1, 4.2, 3.4)),
)
engine.spawn(
    Mesh(sphere(0.2, 20)),
    Material(FlatMaterial("#ffe2b0")),
    Transform(pos=(-3.7, 4.02, 3.4)),
)

lamp = PointLight(color="#ffb464", intensity=110.0, position=(-3.7, 4.02, 3.4))
engine.set_lights([lamp])
Image.fromarray(engine.render_frame())
spot = SpotLight(
    color="#ffe6c0",
    intensity=350.0,
    position=(-3.7, 4.02, 3.4),
    direction=(3.7, -3.1, -3.4),
    inner_angle=0.12,
    outer_angle=0.32,
    distance=12,
)
engine.set_spot(spot)
Image.fromarray(engine.render_frame())
engine.set_sun(DirectionalLight(color="#8fa6d8", intensity=0.08, direction=(-0.7, -0.3, 0.35)))
lamp.intensity = 60.0
spot.intensity = 700.0
engine.camera.set_pose(position=(5.5, 3.2, 7.5), target=(0.0, 0.8, 0.0))
Image.fromarray(engine.render_frame())
