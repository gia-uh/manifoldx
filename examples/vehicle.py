"""Drive a car on ray wheels: manifoldx.vehicles over a road with a dip, a ramp and two kerbs.

Arrows or WASD drive, Space is the handbrake, R puts the car back at the start, 1 to 9 set the assists from
fully helped (1) to none (9). The car is a build_car model on a car_spec built from the same measurements, so
the drawn body lands where the physics puts it. engine.state["car"] carries the telemetry for an agent.

With the native wheel, pass --native-physics to use the Rust fleet in the window,
or --native-physics-check to run the same track without opening a window.
"""

import sys

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling.assets.car import CarParams, build_car, place_car, spawn_car
from manifoldx.systems import Query
from manifoldx.vehicles import NativeVehicles, ShapeGround, Vehicles, car_spec, strengths_for
from manifoldx.vehicles.body import matrices

PROFILE = [(-60, 0.0), (20, 0.0), (35, -1.5), (50, -1.5), (60, 0.0), (90, 0.0), (100, 1.2), (104, 1.2),
           (106, 0.0), (400, 0.0)]
KERBS = (130.0, 150.0)

params = CarParams()
spec = car_spec(length=params.length, width=params.width, height=params.height, wheelbase=params.wheelbase,
                front_overhang=params.front_overhang, track=params.track, wheel_radius=params.wheel_radius,
                mass=1250, power_kw=85, drive="rear", ratios=(3.6, 2.1, 1.4, 1.0, 0.8))
ground = ShapeGround(profile=PROFILE, kerbs=KERBS, kerb_height=0.15)
native_check = __name__ == "__main__" and "--native-physics-check" in sys.argv
use_native = native_check or (__name__ == "__main__" and "--native-physics" in sys.argv)
fleet = NativeVehicles([spec], ground) if use_native else Vehicles([spec])


def reset_car():
    if use_native:
        fleet.place(0, (0.0, 0.0, 0.0), 0.0)
    else:
        fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)


def advance(controls, dt, realism):
    strengths = strengths_for(realism)
    if use_native:
        return fleet.advance(controls, dt, strengths)
    return fleet.advance(controls, ground, None, dt, strengths)


if native_check:
    reset_car()
    steps = sum(advance([[0.0, 1.0, 0.0, 0.0]], 1 / 60, 30)["substeps"] for _ in range(180))
    row = fleet.snapshot()[0]
    speed = fleet.telemetry(0)["speed"]
    if steps != 360 or row["position"][0] <= 10.0 or speed <= 10.0 or len(row["wheel_centres"]) != 4:
        raise AssertionError(f"native vehicle did not complete its drive: {steps=}, {speed=}, {row=}")
    print(f"{steps} native substeps, {row['position'][0]:.3f} m, {speed:.3f} m/s, {len(row['wheel_centres'])} wheels")
    raise SystemExit(0)

engine = mx.Engine("Vehicle")
model = build_car(params)
state = {"realism": 30, "handles": None}


def road():
    """The profile as tilted slabs, 8 m wide, and the kerbs as steps."""
    slab = mx.geometry.cube(1, 1, 1)
    grey = mx.material.phong(mx.colors.WHITE)
    for (x0, y0), (x1, y1) in zip(PROFILE, PROFILE[1:]):
        length, angle = np.hypot(x1 - x0, y1 - y0), np.arctan2(y1 - y0, x1 - x0)
        mid = ((x0 + x1) / 2, (y0 + y1) / 2 - 0.05, 0.0)
        rot = (0.0, 0.0, float(np.sin(angle / 2)), float(np.cos(angle / 2)))
        engine.spawn(Mesh(slab), Material(grey), Transform(pos=mid, rot=rot, scale=(length, 0.1, 8.0)))
    for i, x in enumerate(KERBS):  # each kerb steps the road up again
        engine.spawn(Mesh(slab), Material(grey), Transform(pos=(x + 100, 0.075 + 0.15 * i, 0.0), scale=(200, 0.15, 8.0)))


@engine.on("startup")
def setup(_payload):
    road()
    reset_car()
    state["handles"] = spawn_car(engine, model)


@engine.system
def drive(query: Query[Transform], dt: float):
    key = engine.input.is_pressed
    if engine.input.just_pressed("r"):
        reset_car()
    for n in range(1, 10):
        if engine.input.just_pressed(str(n)):
            state["realism"] = (n - 1) * 12.5
    controls = np.array([[float(key("ArrowRight") or key("d")) - float(key("ArrowLeft") or key("a")),
                          float(key("ArrowUp") or key("w")), float(key("ArrowDown") or key("s")),
                          float(key("Space"))]])
    advance(controls, dt, state["realism"])
    at, rotation = fleet.model_transform(0)
    if use_native:
        row = fleet.snapshot()[0]
        steer, spin = np.mean(row["steer_angle"][:2]), np.mean(row["wheel_turn"])
    else:
        steer, spin = fleet.steer_angle[0, :2].mean(), fleet.wheel_turn[0].mean()
    place_car(state["handles"], model, at=at, rotation=rotation, wheel_turn=float(spin),
              steer=-float(steer), travel=fleet.travel(0).tolist())
    heading = matrices(rotation[None])[0][:, 0] * np.array([1.0, 0.0, 1.0])
    heading /= max(np.linalg.norm(heading), 1e-6)
    engine.camera.set_pose(at - 9.0 * heading + np.array([0.0, 3.5, 0.0]), at + np.array([0.0, 1.0, 0.0]))
    engine.state["car"] = fleet.telemetry(0)


if __name__ == "__main__":
    engine.run()
