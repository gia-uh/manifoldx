"""Vehicles: the cost of one 120 Hz step of a fleet whose cars all touch, wall to wall and car to car."""

import numpy as np

from manifoldx.vehicles import FlatGround, Vehicles, car_spec

from benchmarks.harness import case, measure


def _queue(n: int):
    """n sedans in a row, nose to tail and already touching, the last one pushing the rest."""
    spec = car_spec(length=4.5, width=1.8, height=1.45, wheelbase=2.6, front_overhang=0.9, track=1.53,
                    wheel_radius=0.31, mass=1300.0, power_kw=90.0)
    ground, fleet = FlatGround(), Vehicles([spec] * n)
    for i in range(n):
        fleet.place(i, (i * 4.45, 0.0, 0.0), 0.0, ground)
    controls = np.zeros((n, 4))
    controls[0, 1] = 1.0
    return fleet, ground, controls


@case("vehicles", "step_in_contact", sizes=[2, 8], smoke=2)
def step_in_contact(n):
    """One Vehicles step of n cars in a row, each touching the next."""
    fleet, ground, controls = _queue(n)
    fleet.step(controls, ground)
    return measure(lambda: fleet.step(controls, ground))
