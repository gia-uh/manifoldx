"""Ray wheels: from each mount a ray runs down the car's -y, travel + radius long. Where it meets the ground is
the wheel; how much it was shortened is how far the spring is compressed, and the spring's force is the
wheel's load."""

from __future__ import annotations

import numpy as np

SAMPLES = (-0.6, 0.0, 0.6)  # where along the wheel, in radii, the ground is read; the wheel rests on the highest
NORMAL_STEP = 0.3  # metres between the reads that give the ground's normal
DAMPER_SPEED = 2.0  # m/s: the most compression speed the damper sees, as a tyre's give would allow
BUMP_STOP = 10.0  # times the spring's rate once the travel is used up


def contact(ground, mounts, down, forward, radius, travel, near):
    """For mounts (N, W, 3) with the car's down and forward (N, 3): the distance from each mount to its wheel's
    centre resting on the ground (the spring is compressed by travel minus that distance), the ground height under it, and the ground's normal (N, W, 3).

    The ray is taken as vertical for the ground reads: a car tilted a few degrees reads the ground a few
    centimetres off, which the wheel's width hides.
    """
    n, w, _ = mounts.shape
    fwd = forward[:, None, :] * radius[..., None]
    best, heights = np.full((n, w), -np.inf), np.zeros((n, w))
    for s in SAMPLES:
        p = mounts + fwd * s
        h = ground.height(p[..., 0].ravel(), p[..., 2].ravel(), near.ravel()).reshape(n, w)
        centre = h + radius * np.sqrt(1 - s * s)  # the wheel's centre height that touches the ground at p
        better = centre > best
        best, heights = np.where(better, centre, best), np.where(better, h, heights)
    # distance down the ray from the mount to that wheel centre, through the car's tilt
    dist = (mounts[..., 1] - best) / np.maximum(-down[:, None, 1], 0.2)
    x, z = mounts[..., 0], mounts[..., 2]
    hx = ground.height((x + NORMAL_STEP).ravel(), z.ravel(), heights.ravel()).reshape(n, w)
    hz = ground.height(x.ravel(), (z + NORMAL_STEP).ravel(), heights.ravel()).reshape(n, w)
    h0 = ground.height(x.ravel(), z.ravel(), heights.ravel()).reshape(n, w)
    normal = np.stack([-(hx - h0) / NORMAL_STEP, np.ones_like(h0), -(hz - h0) / NORMAL_STEP], -1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return dist, heights, normal


def spring_force(comp, comp_prev, dt, spring, damper, travel, arb, pair):
    """The force each wheel's spring, damper and anti-roll bar push with, never pulling."""
    rate = np.clip((comp - comp_prev) / dt, -DAMPER_SPEED, DAMPER_SPEED)
    over = np.maximum(comp - travel, 0.0)
    other = np.take_along_axis(comp, pair, axis=1)
    f = spring * comp + spring * BUMP_STOP * over + damper * rate + arb * (comp - other)
    return np.where(comp > 0, np.maximum(f, 0.0), 0.0)
