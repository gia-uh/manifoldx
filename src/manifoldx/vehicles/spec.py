"""What a vehicle is made of: its body, wheels, tyres, engine and gearbox, as plain frozen dataclasses.

Axes are the car's own: x forward, y up, z to the right, in metres from the centre of gravity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tyre:
    """A simplified Pacejka tyre: B (stiffness), C (shape) and E (curvature) for each direction, and its grip."""

    mu: float = 1.0
    b_lat: float = 11.0  # peak near 0.16 rad, 80 % of the peak at a slip of 1
    c_lat: float = 1.5
    e_lat: float = 0.05
    b_long: float = 18.0  # peak near 0.09 slip ratio, 80 % of the peak when locked
    c_long: float = 1.45
    e_long: float = -0.3


@dataclass(frozen=True)
class Wheel:
    """A wheel hung from the body at mount: a ray of travel + radius straight down the car's -y."""

    mount: tuple[float, float, float]
    radius: float
    travel: float
    spring: float  # N/m
    damper: float  # N·s/m
    tyre: Tyre = Tyre()
    driven: bool = False
    steered: bool = False
    brake_share: float = 0.25
    handbrake: bool = False
    inertia: float = 1.2  # kg·m²


@dataclass(frozen=True)
class Engine:
    """A torque curve through three points, in N·m at each rpm, and the braking it does off the throttle."""

    idle_rpm: float
    peak_torque_rpm: float
    peak_power_rpm: float
    idle_torque: float
    peak_torque: float
    power_torque: float
    engine_brake: float = 0.02  # N·m per rpm with the throttle released


@dataclass(frozen=True)
class Gearbox:
    """Forward ratios from first up, the final drive, reverse, and how long a shift cuts the drive."""

    ratios: tuple[float, ...]
    final: float
    reverse: float = 3.5
    shift_time: float = 0.3


@dataclass(frozen=True)
class VehicleSpec:
    """One vehicle. size is (length, height, width) of its body box; inertia is diagonal in car axes."""

    mass: float
    inertia: tuple[float, float, float]
    size: tuple[float, float, float]
    wheels: tuple[Wheel, ...]
    engine: Engine
    gearbox: Gearbox
    arb: tuple[float, float] = (0.0, 0.0)  # N/m of the front and the rear anti-roll bar
    cd: float = 0.35
    area: float = 2.0
    rolling: float = 0.015  # rolling resistance per newton of load
    max_steer: float = 0.6  # radians of the outer front wheel at full lock
    brake_torque: float = 3000.0  # N·m over all wheels at full brake


def box_inertia(mass: float, size: tuple[float, float, float]) -> tuple[float, float, float]:
    """The diagonal inertia of a solid box of this mass and (length, height, width), about x, y and z."""
    lx, ly, lz = size
    return (mass / 12 * (ly * ly + lz * lz), mass / 12 * (lx * lx + lz * lz), mass / 12 * (lx * lx + ly * ly))
