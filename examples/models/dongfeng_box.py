"""A recipe: the Dongfeng BOX as `CarParams`.

Its measurements come from the spec sheet and were fitted to photos; its lamps, grille, wheels, mirrors and cabin
are the shapes traced from drawings of the real car. Run it to render the car, doors and hatch open, to dongfeng_box.png:

    uv run python examples/models/dongfeng_box.py
"""

from PIL import Image

import manifoldx as mx
from manifoldx.modeling.assets.car import CarParams, build_car, place_car, spawn_car
from manifoldx.resources import DirectionalLight

BOX = CarParams(
    length=4.03,
    width=1.81,
    height=1.57,
    wheelbase=2.66,
    front_overhang=0.72,
    track=1.55,
    wheel_radius=0.318,
    nose_top=0.75,
    hood_front=0.925,
    cowl_height=1.20,
    hatch_height=1.165,
    tail_top=0.91,
    belt_front=1.03,
    belt_rear=1.11,
    headlamp="wedge",
    headlamp_center=(0.505, 0.826),
    headlamp_size=(0.36, 0.185),
    headlamp_yaw=30.0,
    drl_bar=True,
    drl_height=0.826,
    grille=(0.83, 0.37, 0.50),
    grille_style="smile",
    taillamp="ring",
    taillamp_center=(0.68, 1.03),
    taillamp_size=(0.33, 0.22),
    taillamp_yaw=35.0,
    spoiler=True,
    mirror="pod",
    wheel="swirl",
    dash="deck",
    steering="tri",
    door_card="quilt",
    console="tray",
    door_strip=True,
    fog_rings=True,
)

if __name__ == "__main__":
    engine = mx.Engine("dongfeng box", width=640, height=400)
    engine.background_color = (0.62, 0.72, 0.84)
    engine.set_sun(DirectionalLight(color="#fff3dd", intensity=3.0, direction=(-0.5, -0.8, -0.35)))
    car = build_car(BOX)
    place_car(spawn_car(engine, car), car, at=(0, 0, 0), doors=1.0, hatch=0.6, steer=0.35)
    engine.camera.set_pose(position=(3.6, 1.8, 3.4), target=(0.0, 0.7, 0.0))
    Image.fromarray(engine.render_frame(supersample=1)).save("dongfeng_box.png")
