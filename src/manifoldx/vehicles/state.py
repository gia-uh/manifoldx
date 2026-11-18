"""N vehicle specs packed into arrays: one row per vehicle, (N, W) per wheel with W the most wheels any has."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .spec import VehicleSpec
from .tyre import peak_slip

TYRE_FIELDS = ("mu", "b_lat", "c_lat", "e_lat", "b_long", "c_long", "e_long")


@dataclass
class Packed:
    mass: np.ndarray  # (N,)
    inertia: np.ndarray  # (N, 3)
    size: np.ndarray  # (N, 3)
    wheel_mask: np.ndarray  # (N, W) bool: a real wheel
    mount: np.ndarray  # (N, W, 3)
    radius: np.ndarray
    travel: np.ndarray
    spring: np.ndarray
    damper: np.ndarray
    tyre: np.ndarray  # (N, W, 7) in TYRE_FIELDS order
    kappa_peak: np.ndarray  # (N, W) slip ratio of the longitudinal peak
    alpha_peak: np.ndarray  # (N, W) slip-angle tangent of the lateral peak
    driven: np.ndarray
    steered: np.ndarray
    brake_share: np.ndarray
    handbrake: np.ndarray
    wheel_inertia: np.ndarray
    front: np.ndarray  # (N, W) bool: ahead of the CG
    pair: np.ndarray  # (N, W) int: the other wheel on the same axle, or itself
    arb: np.ndarray  # (N, W) N/m of the bar on this wheel's axle
    engine: np.ndarray  # (N, 7): idle_rpm, peak_torque_rpm, peak_power_rpm, idle_t, peak_t, power_t, brake
    ratios: np.ndarray  # (N, G) forward ratios, padded with 0
    n_gears: np.ndarray  # (N,)
    final: np.ndarray
    reverse: np.ndarray
    shift_time: np.ndarray
    cd: np.ndarray
    area: np.ndarray
    rolling: np.ndarray
    max_steer: np.ndarray
    brake_torque: np.ndarray


def pack(specs: Sequence[VehicleSpec]) -> Packed:
    """Pack specs into arrays. Raises ValueError when a vehicle's brake shares do not sum to 1."""
    n, w = len(specs), max(len(s.wheels) for s in specs)
    g = max(len(s.gearbox.ratios) for s in specs)
    f = lambda: np.zeros((n, w))
    mask, mount = np.zeros((n, w), bool), np.zeros((n, w, 3))
    radius, travel, spring, damper, share, inertia = f(), f(), f(), f(), f(), f()
    kpeak, apeak = np.ones((n, w)), np.ones((n, w))
    tyre = np.zeros((n, w, len(TYRE_FIELDS)))
    driven, steered, handbrake, front = (np.zeros((n, w), bool) for _ in range(4))
    pair, arb = np.tile(np.arange(w), (n, 1)), f()
    ratios = np.zeros((n, g))
    for i, s in enumerate(specs):
        if abs(sum(wh.brake_share for wh in s.wheels) - 1) > 1e-6:
            raise ValueError(f"vehicle {i}: brake_share sums to {sum(wh.brake_share for wh in s.wheels):.3f}, not 1")
        for j, wh in enumerate(s.wheels):
            mask[i, j], mount[i, j] = True, wh.mount
            radius[i, j], travel[i, j], spring[i, j], damper[i, j] = wh.radius, wh.travel, wh.spring, wh.damper
            tyre[i, j] = [getattr(wh.tyre, k) for k in TYRE_FIELDS]
            t = wh.tyre
            kpeak[i, j], apeak[i, j] = peak_slip(t.b_long, t.c_long, t.e_long), peak_slip(t.b_lat, t.c_lat, t.e_lat)
            driven[i, j], steered[i, j], handbrake[i, j] = wh.driven, wh.steered, wh.handbrake
            share[i, j], inertia[i, j] = wh.brake_share, wh.inertia
            front[i, j] = wh.mount[0] > 0
            arb[i, j] = s.arb[0] if front[i, j] else s.arb[1]
            for k, other in enumerate(s.wheels):
                if k != j and abs(other.mount[0] - wh.mount[0]) < 1e-6 and other.mount[2] * wh.mount[2] < 0:
                    pair[i, j] = k
        ratios[i, : len(s.gearbox.ratios)] = s.gearbox.ratios
    col = lambda get: np.array([get(s) for s in specs], float)
    e = lambda s: s.engine
    return Packed(
        mass=col(lambda s: s.mass), inertia=np.array([s.inertia for s in specs], float),
        size=np.array([s.size for s in specs], float), wheel_mask=mask, mount=mount, radius=radius, travel=travel,
        spring=spring, damper=damper, tyre=tyre, kappa_peak=kpeak, alpha_peak=apeak, driven=driven, steered=steered, brake_share=share,
        handbrake=handbrake, wheel_inertia=inertia, front=front, pair=pair, arb=arb,
        engine=np.array([[e(s).idle_rpm, e(s).peak_torque_rpm, e(s).peak_power_rpm, e(s).idle_torque,
                          e(s).peak_torque, e(s).power_torque, e(s).engine_brake] for s in specs], float),
        ratios=ratios, n_gears=np.array([len(s.gearbox.ratios) for s in specs]),
        final=col(lambda s: s.gearbox.final), reverse=col(lambda s: s.gearbox.reverse),
        shift_time=col(lambda s: s.gearbox.shift_time), cd=col(lambda s: s.cd), area=col(lambda s: s.area),
        rolling=col(lambda s: s.rolling), max_steer=col(lambda s: s.max_steer),
        brake_torque=col(lambda s: s.brake_torque))
