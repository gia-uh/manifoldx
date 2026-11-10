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
