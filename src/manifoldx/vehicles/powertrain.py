"""Engine, automatic gearbox, open differential, brakes and Ackermann steering, over N vehicles."""

from __future__ import annotations

import numpy as np

RPM = 60 / (2 * np.pi)  # rpm per rad/s
CLUTCH_RPM = 1200.0  # under this the clutch slips: the engine pulls as if at this speed
REDLINE = 1.15  # times the power peak's rpm where the torque falls to nothing
UP_AT = 0.9  # shift up at this share of the power peak's rpm with the pedal down
LIGHT_UP = 0.4  # with the pedal barely pressed, shift up at this share: the shift point rises with the pedal squared
OFF_PEDAL = 0.1  # the engine brakes only with the pedal under this, fully at zero
DOWN_AT = 0.75  # shift down when the lower gear would turn under this share of the shift-up rpm
REVERSE_BELOW = 0.5  # m/s: under this, the brake at rest selects reverse and the throttle selects first


def torque(engine: np.ndarray, rpm: np.ndarray, throttle: np.ndarray) -> np.ndarray:
    """Engine torque at rpm: the throttle's share of the three-point curve, less engine braking with the pedal
    (nearly) released."""
    idle, t_rpm, p_rpm, idle_t, peak_t, power_t, brake = engine.T
    xs = np.stack([idle, t_rpm, p_rpm, p_rpm * REDLINE], -1)
    ys = np.stack([idle_t, peak_t, power_t, np.zeros_like(idle_t)], -1)
    r = np.clip(rpm, idle, p_rpm * REDLINE)
    k = np.clip((r[:, None] >= xs[:, 1:]).sum(-1), 0, 2)
    rows = np.arange(len(r))
    x0, x1, y0, y1 = xs[rows, k], xs[rows, k + 1], ys[rows, k], ys[rows, k + 1]
    curve = y0 + (y1 - y0) * (r - x0) / np.maximum(x1 - x0, 1e-9)
    drag = brake * np.maximum(rpm - idle, 0.0) * np.clip(1 - throttle / OFF_PEDAL, 0.0, 1.0)
    return curve * throttle - drag


def up_rpm(peak_power_rpm, pedal):
    """The rpm the automatic shifts up at: early with a light pedal, at UP_AT of the power peak with it down."""
    return peak_power_rpm * (LIGHT_UP + (UP_AT - LIGHT_UP) * pedal * pedal)


def ackermann(steer: np.ndarray, mount: np.ndarray, steered: np.ndarray, max_steer: np.ndarray) -> np.ndarray:
    """Each wheel's angle for steer in [-1, 1], positive to the right: the outer front wheel turns
    steer·max_steer and the inner one more, toward the same turning centre on the rear axle's line."""
    x, z = mount[..., 0], mount[..., 2]
    big = np.where(steered, -np.inf, np.inf)
    base = np.max(np.where(steered, x, -np.inf), 1) - np.min(np.where(steered, np.inf, x), 1)
    track = np.max(np.where(steered, z, -np.inf), 1) - np.min(np.where(steered, z, np.inf), 1)
    del big
    outer = np.abs(steer) * max_steer
    turning = np.abs(steer) > 1e-6
    radius = base / np.tan(np.maximum(outer, 1e-6)) - track / 2  # to the car's middle line
    inner = np.arctan(base / np.maximum(radius - track / 2, 1e-6))
    side = np.sign(steer)[:, None]
    is_inner = np.sign(z) == side  # turning right (+), the right (+z) wheels are inside
    angle = np.where(is_inner, inner[:, None], outer[:, None]) * side
    return np.where(steered & turning[:, None], angle, 0.0)
