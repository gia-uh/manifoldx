# A car that drives

The plaza's cars have so far been placed by hand: `place_car` puts a built car at a position and a turn, and
nothing makes it move. `manifoldx.vehicles` makes it drive. A car is a rigid body on ray wheels, with springs,
tyres that grip and then slide, an engine, an automatic gearbox and brakes; driving assists sit on top and can
be turned from fully helping to not there at all. It is numpy only and knows nothing about the engine or the
ECS: a game calls it from a system and draws what it says.

## Three layers

The driver's controls go through the assists, then into the physics, which stands on a ground:

- **Controls** are four numbers per vehicle: steer in [-1, 1], positive to the right, then throttle, brake and
  handbrake in [0, 1].
- **Assists** read the physics and correct the controls before a step: `apply_assists` with a strength per
  assist, in the order of `ASSISTS` (steer, traction, abs, stability). `strengths_for(realism)` gives the four
  strengths for a realism from 0 (easy) to 100 (real), and can lock some at 0.
- **Physics** is the same with any realism. `Vehicles` steps N vehicles together at a fixed 120 Hz.
- **Ground** is anything with `height(x, z, near)` and `grip(x, z)`: the `Ground` protocol. `FlatGround` is a
  plane; `ShapeGround` is a road along x with a profile, square kerbs and a bridge deck, for tests and examples.

## Describing a vehicle

A `VehicleSpec` holds the mass, the inertia, the body's size, its `Wheel`s, an `Engine` and a `Gearbox`. Each
wheel has its mount on the body, its radius, its suspension travel, spring and damper, and a `Tyre`. Writing one
by hand is long; `car_spec` builds a four-wheeled one from a body's measurements and a few handling numbers.

```python {run}
from manifoldx.vehicles import car_spec

spec = car_spec(length=4.25, width=1.78, height=1.48, wheelbase=2.62, front_overhang=0.86, track=1.53,
                wheel_radius=0.315, mass=1250, power_kw=85, drive="rear", ratios=(3.6, 2.1, 1.4, 1.0, 0.8))
print(len(spec.wheels), "wheels,", spec.mass, "kg, model origin", [round(v, 2) for v in spec.model_origin])
```

`box_inertia` gives the inertia of a solid box, which `car_spec` uses. `pack` turns a list of specs into the
`Packed` arrays the physics runs on, one row per vehicle and one column per wheel.

## Ray wheels

From each mount a ray runs down the car's own -y, as long as the travel and the radius. Where it meets the ground
is the wheel; how far the ray was shortened is how far the spring is compressed, and the spring's force is the
wheel's load. Load moves between the wheels by itself: braking pitches the body, the front springs compress and
the front tyres carry more. A wheel whose ray meets nothing is in the air and carries nothing.

```python {run continue}
import numpy as np
from manifoldx.vehicles import FlatGround, Vehicles

ground = FlatGround()
fleet = Vehicles([spec])
fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
for _ in range(240):
    fleet.step(np.zeros((1, 4)), ground)
print("loads, N:", np.round(fleet.load[0]).tolist())
```

## Tyres

Each tyre turns its slip into force with the magic formula, `magic`: the force grows with the slip, peaks and
falls to about 80 % when the tyre slides. `forces` measures the slip ratio and the slip angle in their own
peaks and shares one curve between them, so a wheel that brakes hard grips less sideways and a locked wheel slides
along its slide. Each wheel spins with its own inertia: the throttle can spin it and the brake can lock it. The automatic shifts up early under a light pedal and late with it down, and the engine brakes only with the pedal released.

## Driving, with and without help

`advance` takes the frame's time, runs the 120 Hz steps it adds up to with the assists applied before each, and
returns how many ran. `pose` gives the position and rotation between the last two steps for drawing, and
`model_transform` where a model built from the same measurements goes. Here the car brakes from 100 km/h while
steering, with no help and then with all of it:

```python {run continue}
from manifoldx.vehicles import strengths_for


def stop_from_100(realism):
    fleet = Vehicles([spec])
    fleet.place(0, (0.0, 0.0, 0.0), 0.0, ground)
    for _ in range(60):
        fleet.step(np.zeros((1, 4)), ground)
    fleet.velocity[0] = (100 / 3.6, 0.0, 0.0)
    fleet.omega[0] = 100 / 3.6 / 0.315
    fleet.gear[0] = 4
    locked = 0
    while fleet.telemetry(0)["speed"] > 0.5:
        fleet.advance(np.array([[0.3, 0.0, 1.0, 0.0]]), ground, None, 1 / 60, strengths_for(realism))
        locked += int(np.any(fleet.omega[0] == 0))
    return round(float(np.hypot(*fleet.position[0, [0, 2]])), 1), locked


for realism in (100, 0):
    distance, locked = stop_from_100(realism)
    print(f"realism {realism}: stopped in {distance} m, {locked} frames with a wheel locked")
```

`telemetry` gives a vehicle's numbers for a HUD, a sound or an agent: speed, gear, rpm, each wheel's load, slip,
spin and travel, the last contact's impulse, and which assists acted.

## Walls and the ground

`step` and `advance` take walls as (x, z) segments. The body's footprint is tested against them by separating
axes, and a contact answers with an impulse at its point, with some bounce and friction: a glancing blow turns
the car along the wall, a corner clipping a wall's end spins it. No vehicle moves more than 0.25 m between two
tests. The body box's corners are tested against the ground too, so a car on its roof rests on it.

## Drawing it

`examples/vehicle.py` builds a car with `build_car`, a `car_spec` from the same `CarParams`, and drives it on a
`ShapeGround`: each frame it calls `advance` and moves the model with `place_car`, passing `rotation`, the
wheels' `wheel_turn` and `steer`, and each wheel's `travel` from `Vehicles.travel`, which `Car.pose` uses to
lift that wheel.

## The native fleet

With a native wheel, `NativeVehicles([spec], FlatGround())` owns the vehicle state in Rust.
Set the starting position with `place`, then call `advance(controls, dt)` once per frame.
Each call returns the number of fixed substeps, discarded time above 50 ms, and native
solver time in nanoseconds. A body can have no wheel load and still collide with a wall or
another body. The contact report gives the first body's index, the second body's index
(or `None` for a wall), the outward normal and the impulse applied to the first body.
Wheel load counts include all the substeps completed in that call. Rigid contact impulse
counts a pair once. The body snapshot reports the impulse on that body in the last substep.

The optional driving-assist strengths run the Python policy once per frame. The resulting
controls and per-wheel corrections are applied to every fixed substep of that frame.
Use `snapshot` for owned poses, wheel centres, compression, load, slip and motor state.
Use `checkpoint` and `restore` to replay the same clock residue, tile updates and inputs.
Draw using `pose` or `model_transform` for interpolation between fixed substeps.

To stream ground, register `TiledGround` with a cell size and a sequence of `GroundTile`
batches. A tile can contain a regular `Heightfield`, a list of `GroundTriangle` surfaces
with level IDs and grip, and vertical `Obstacle` segments with strengths. Clip input
geometry to its cell boundaries. Native lookup by XZ uses `near` to choose the closest
height and returns a `GroundSample` with tile version, level, normal and grip.
Use `update_ground` to stage edits and removals against positive, increasing tile
versions. Use `rebase` to stage a local origin shift. Both become visible at the next
actual substep. This API takes plain data. A custom Python `Ground` must prepare a tile batch
before registration.

Run `python examples/vehicle.py --native-physics-check` with the native wheel to drive
the example's car for 360 fixed steps and read back the wheels and final pose.
`--native-physics` selects the same fleet for the interactive example.
