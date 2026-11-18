"""Driving assists: each reads the physics and corrects the driver's controls, by its own strength 0..1.

A strength mixes what the driver asked with what the assist would do: 0 leaves the controls alone, 1 applies
the whole correction. The physics never knows the assists exist.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .body import matrices

ASSISTS = ("steer", "traction", "abs", "stability")
NARROW_AT = 45.0  # m/s where the steering assist narrows the lock to NARROW of full
NARROW = 0.2
COUNTER = 0.8  # share of the front axle's slide the steering assist steers into with the wheel released
YAW_DEADBAND = 0.05  # rad/s of yaw error the stability assist lets pass
YAW_GAIN = 4000.0  # N·m of brake per rad/s of yaw error
YAW_MAX = 2500.0  # N·m at most on the one wheel it brakes
RECOVER = 8.0  # per second: how fast a released brake or a cut throttle comes back
G = 9.81


def strengths_for(realism: float, locked: Sequence[str] = ()) -> np.ndarray:
    """The four strengths, in ASSISTS order, for a realism of 0 (easy) to 100 (real); locked ones are 0."""
    k = np.full(len(ASSISTS), 1 - np.clip(realism, 0, 100) / 100)
    for name in locked:
        k[ASSISTS.index(name)] = 0.0
    return k


def apply_assists(fleet, controls: np.ndarray, strengths: np.ndarray) -> np.ndarray:
    """The controls (N, 4) after the assists of the given strengths (N, 4) or (4,). Writes fleet.extra_brake,
    fleet.brake_scale, fleet.drive_cut and fleet.lit (N, 4) for the step that follows."""
    p, dt = fleet.p, fleet.DT
    k = np.broadcast_to(np.asarray(strengths, float), (len(controls), len(ASSISTS)))
    k_steer, k_trac, k_abs, k_stab = k.T
    out = np.array(controls, float)
    lit = np.zeros((len(out), len(ASSISTS)), bool)
    R = matrices(fleet.rotation)
    v_car = np.einsum("nji,nj->ni", R, fleet.velocity)  # forward, up, right
    yaw = -np.einsum("nji,nj->ni", R, fleet.angular)[:, 1]  # rad/s, positive turning right
    vx, vz = v_car[:, 0], v_car[:, 2]
    speed = np.hypot(vx, vz)
    front_x = np.max(np.where(p.steered, p.mount[..., 0], -np.inf), 1)
    rear_x = np.min(np.where(p.wheel_mask, p.mount[..., 0], np.inf), 1)
    base = front_x - rear_x
    alpha_peak = np.max(np.where(p.steered, p.alpha_peak, 0.0), 1)

    # steering: narrow with speed, keep the front tyres at or under their peak, steer into a slide when released
    steer = out[:, 0]
    wheel = steer * p.max_steer
    beta = np.arctan2(vz + yaw * front_x, np.maximum(np.abs(vx), 1.0))
    narrow = 1 - (1 - NARROW) * np.clip(speed / NARROW_AT, 0, 1)
    want = wheel * narrow + COUNTER * beta * (1 - np.abs(steer))
    want = np.clip(want, beta - alpha_peak, beta + alpha_peak)
    want = np.where(speed > 3.0, want, wheel)
    new = wheel + (want - wheel) * k_steer
    lit[:, 0] = np.abs(new - wheel) > 0.02
    out[:, 0] = np.clip(new / p.max_steer, -1, 1)

    # traction: cut the drive while a driven wheel slips past its peak
    driven = p.driven & p.wheel_mask
    over = np.max(np.where(driven, np.abs(fleet.slip[..., 0]) / p.kappa_peak - 1, 0.0), 1)
    cut = k_trac * np.clip(over * 2, 0, 1)
    fleet.drive_cut = np.maximum(cut, fleet.drive_cut - RECOVER * dt)
    lit[:, 1] = cut > 0

    # anti-lock: let go of a braked wheel's pedal brake while it slips past its peak
    locking = (fleet.slip[..., 0] < -1.1 * p.kappa_peak) | ((fleet.omega == 0) & (np.abs(vx) > 1.0)[:, None])
    locking &= p.wheel_mask & (out[:, 2] > 0)[:, None]
    fleet.brake_scale = np.where(locking, 1 - k_abs[:, None], np.minimum(fleet.brake_scale + RECOVER * dt, 1.0))
    lit[:, 2] = np.any(locking & (k_abs[:, None] > 0), 1)

    # stability: brake one wheel when the car yaws more or less than the wheel asks, within the grip
    delta = out[:, 0] * p.max_steer
    target = vx * np.tan(delta) / np.maximum(base, 0.5)
    limit = np.max(p.tyre[..., 0], 1) * G / np.maximum(speed, 1.0)
    target = np.clip(target, -limit, limit)
    error = yaw - target
    acting = (speed > 4.0) & (np.abs(error) > YAW_DEADBAND) & (k_stab > 0)
    torque = k_stab * np.minimum(YAW_GAIN * (np.abs(error) - YAW_DEADBAND), YAW_MAX)
    turning_right = np.where(np.abs(target) > 0.02, target > 0, yaw > 0)
    oversteer = np.abs(yaw) > np.abs(target)
    left = p.mount[..., 2] < 0
    front = p.front
    # too much yaw: brake the front wheel outside the turn; too little: the rear wheel inside it
    outside = np.where(turning_right[:, None], left, ~left)
    pick = np.where(oversteer[:, None], front & outside, ~front & ~outside) & p.wheel_mask
    fleet.extra_brake = np.where(pick & acting[:, None], torque[:, None], 0.0)
    fleet.drive_cut = np.where(acting & ~oversteer, np.maximum(fleet.drive_cut, 0.5 * k_stab), fleet.drive_cut)
    lit[:, 3] = acting
    fleet.lit = lit
    return out
