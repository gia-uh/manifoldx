"""Rigid-body helpers over N bodies: quaternions are xyzw, rotation matrices map car axes to world axes."""

from __future__ import annotations

import numpy as np


def matrices(q: np.ndarray) -> np.ndarray:
    """(N, 4) unit quaternions to (N, 3, 3) matrices whose columns are the car's x, y and z in the world."""
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], -1),
        np.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], -1),
        np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -1)


def yaw_quaternion(yaw: float) -> np.ndarray:
    """A turn of yaw radians about +y: a positive yaw turns the car's +x toward -z."""
    return np.array([0.0, np.sin(yaw / 2), 0.0, np.cos(yaw / 2)])


def integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """q turned by the world angular velocity omega over dt, normalised."""
    wx, wy, wz = omega.T
    x, y, z, w = q.T
    dq = 0.5 * dt * np.stack([wx * w + wy * z - wz * y, wy * w + wz * x - wx * z, wz * w + wx * y - wy * x,
                              -(wx * x + wy * y + wz * z)], -1)
    out = q + dq
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Normalised linear blend from a to b, taking the short way: enough between two steps 1/120 s apart."""
    b = np.where((np.sum(a * b, axis=1) < 0)[:, None], -b, b)
    out = a + (b - a) * t
    return out / np.linalg.norm(out, axis=1, keepdims=True)
