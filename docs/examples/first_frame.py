import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane

engine = mx.Engine("plaza", width=480, height=270)
FLOOR = (-0.70710678, 0.0, 0.0, 0.70710678)  # a quarter turn about x: lay the plane flat

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
engine.camera.set_pose(position=(14, 9, 14), target=(0, 0, 0))
engine.set_sun(
    DirectionalLight(color="#ffffff", intensity=3.0, direction=(-0.4, -1, -0.3))
)
from PIL import Image

frame = engine.render_frame()
print(frame.shape, frame.dtype)
Image.fromarray(frame)
