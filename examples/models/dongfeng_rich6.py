"""A recipe: the Dongfeng Rich 6, a double-cab pickup, as `CarParams`.

Its measurements come from the spec sheet (5290 x 1850 x 1810 mm, 3150 mm wheelbase, 255/70 R16 tyres, a bed
1510 x 1562 x 475 mm) and its profile was fitted to photos. Run it to render the pickup, tailgate down, to
dongfeng_rich6.png:

    uv run python examples/models/dongfeng_rich6.py
"""

from PIL import Image

import manifoldx as mx
from manifoldx.modeling.assets.car import CarParams, build_car, place_car, spawn_car
from manifoldx.resources import DirectionalLight

RICH6 = CarParams(
    length=5.29, width=1.85, height=1.81, wheelbase=3.15, front_overhang=0.88, track=1.57,
    wheel_radius=0.382, tyre_width=0.255, rim_ratio=0.56, arch_gap=0.05,
    nose_top=1.08, hood_front=1.17, nose_blunt=0.85, cowl_u=0.26, cowl_height=1.29,
    windscreen_u=0.37, roof_peak_u=0.46, roof_end_u=0.655, roof_end_drop=0.03, hatch_u=0.672, hatch_height=1.37,
    tail_top=1.37, belt_front=1.22, belt_rear=1.30, sill=0.50, bumper_bottom_front=0.45, bumper_bottom_rear=0.55,
    hood_crown=0.04, nose_half=0.93, tail_half=0.99, round_front=0.07, round_rear=0.02, nose_bulge=0.03,
    tail_bulge=0.0, tumblehome=0.14, shoulder=0.97,
    a_pillar_foot=0.35, b_pillar_u=0.49, b_pillar_width=0.09, c_pillar_u=0.645,
    bed_u=0.69, bed_floor=0.86,
    headlamp_center=(0.66, 1.05), headlamp_size=(0.42, 0.15), headlamp_yaw=25.0,
    grille=(0.56, 0.80, 1.11), taillamp_center=(0.84, 1.10), taillamp_size=(0.14, 0.32), taillamp_yaw=5.0,
    mirror_size=(0.24, 0.22),
)

if __name__ == "__main__":
    engine = mx.Engine("dongfeng rich 6", width=640, height=400)
    engine.background_color = (0.62, 0.72, 0.84)
    engine.set_sun(DirectionalLight(color="#fff3dd", intensity=3.0, direction=(-0.5, -0.8, -0.35)))
    car = build_car(RICH6)
    place_car(spawn_car(engine, car), car, at=(0, 0, 0), hatch=1.0)
    engine.camera.set_pose(position=(-4.6, 2.6, 4.0), target=(-0.8, 0.9, 0.0))
    Image.fromarray(engine.render_frame(supersample=1)).save("dongfeng_rich6.png")
