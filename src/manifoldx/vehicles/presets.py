"""A VehicleSpec from a body's measurements and a few handling numbers, for cars, buses and trucks."""

from __future__ import annotations

import math

from .spec import Engine, Gearbox, Tyre, VehicleSpec, Wheel, box_inertia

G = 9.81


def car_spec(*, length, width, height, wheelbase, front_overhang, track, wheel_radius, mass, power_kw,
             drive="rear", ratios=(3.6, 2.1, 1.4, 1.0), final=3.9, spring_hz=(1.4, 1.4), damping=0.35,
             arb=(9000.0, 4000.0), tyre_mu=1.0, brake_torque=5000.0, brake_front=0.6, cg_height=0.38,
             cg_back=0.45, cd=0.35, max_steer=0.6, travel=0.2, peak_power_rpm=5500.0, shift_time=0.3,
             efficiency=0.85) -> VehicleSpec:
    """Four wheels on two axles. The CG sits cg_back of the wheelbase behind the front axle and cg_height of
    the height over the ground at rest; model_origin is the drawn model's origin (the ground under the middle of
    its length) in car axes, so a model built from the same measurements lands on the physics."""
    a = cg_back * wheelbase  # CG behind the front axle
    b = wheelbase - a
    front_x, rear_x = a, -b
    corner_front, corner_rear = mass * b / wheelbase / 2, mass * a / wheelbase / 2
    springs = [m * (2 * math.pi * f) ** 2 for m, f in ((corner_front, spring_hz[0]), (corner_rear, spring_hz[1]))]
    rest = [m * G / k for m, k in zip((corner_front, corner_rear), springs)]
    cg_y = cg_height * height
    tyre = Tyre(mu=tyre_mu)
    wheels = []
    for x, k, m, r0, front in ((front_x, springs[0], corner_front, rest[0], True), (rear_x, springs[1], corner_rear, rest[1], False)):
        mount_y = wheel_radius + travel - min(r0, travel) - cg_y  # so the wheel touches the ground at rest
        for z in (-track / 2, track / 2):
            wheels.append(Wheel(mount=(x, mount_y, z), radius=wheel_radius, travel=travel, spring=k,
                                damper=2 * damping * math.sqrt(k * m), tyre=tyre,
                                driven=drive == "all" or (drive == "front") == front, steered=front,
                                brake_share=(brake_front if front else 1 - brake_front) / 2, handbrake=not front))
    omega = peak_power_rpm * 2 * math.pi / 60
    t_power = power_kw * 1000 / omega
    engine = Engine(idle_rpm=850, peak_torque_rpm=0.6 * peak_power_rpm, peak_power_rpm=peak_power_rpm,
                    idle_torque=0.6 * t_power, peak_torque=1.3 * t_power, power_torque=t_power)
    middle = length / 2 - front_overhang - a  # the model's middle, ahead of the CG
    return VehicleSpec(mass=mass, inertia=box_inertia(mass, (length, height, width)), size=(length, height, width),
                       wheels=tuple(wheels), engine=engine, gearbox=Gearbox(ratios=tuple(ratios), final=final, shift_time=shift_time,
                                                                       efficiency=efficiency),
                       arb=arb, cd=cd, area=0.85 * width * height, max_steer=max_steer, brake_torque=brake_torque,
                       model_origin=(-middle, -cg_y, 0.0))
