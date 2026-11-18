"""Vehicles: N rigid bodies on ray wheels, stepped together at a fixed 120 Hz."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import suspension
from .body import integrate, matrices, yaw_quaternion
from .spec import VehicleSpec
from .state import pack

G = 9.81


class Vehicles:
    DT = 1 / 120

    def __init__(self, specs: Sequence[VehicleSpec]):
        self.specs = list(specs)
        self.p = pack(self.specs)
        n, w = self.p.wheel_mask.shape
        self.position, self.velocity, self.angular = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
        self.rotation = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
        self.load, self.compression = np.zeros((n, w)), np.zeros((n, w))
        self.ground_y = np.zeros((n, w))

    def place(self, i: int, position, yaw: float, ground) -> None:
        """Put vehicle i at rest at (x, z) facing yaw, its wheels on the ground level nearest position's y."""
        p = self.p
        self.rotation[i] = yaw_quaternion(yaw)
        x, y, z = position
        R = matrices(self.rotation[i : i + 1])[0]
        mounts = (R @ p.mount[i].T).T
        near = np.full(len(mounts), float(y))
        h = ground.height(x + mounts[:, 0], z + mounts[:, 2], near)
        rest = p.mass[i] * G / p.wheel_mask[i].sum() / np.maximum(p.spring[i], 1.0)
        reach = p.radius[i] + p.travel[i] - np.minimum(rest, p.travel[i])
        body_y = np.max(np.where(p.wheel_mask[i], h + reach - mounts[:, 1], -np.inf))
        self.position[i] = (x, body_y, z)
        self.velocity[i] = self.angular[i] = 0.0
        self.ground_y[i] = h
        self.compression[i] = 0.0
        self.load[i] = 0.0

    def angular_body(self) -> np.ndarray:
        """Angular velocity in the car's own axes, (N, 3)."""
        return np.einsum("nji,nj->ni", matrices(self.rotation), self.angular)

    def step(self, controls: np.ndarray, ground, walls: np.ndarray | None = None) -> None:
        """Advance every vehicle by DT under controls (N, 4): steer, throttle, brake, handbrake."""
        p, dt = self.p, self.DT
        R = matrices(self.rotation)
        down, forward = -R[:, :, 1], R[:, :, 0]
        mounts = self.position[:, None, :] + np.einsum("nij,nwj->nwi", R, p.mount)
        dist, self.ground_y, normal = suspension.contact(ground, mounts, down, forward, p.radius, p.travel,
                                                         self.ground_y)
        comp = np.where(p.wheel_mask, p.travel - dist, 0.0)
        comp = np.maximum(comp, 0.0)
        load = suspension.spring_force(comp, self.compression, dt, p.spring, p.damper, p.travel, p.arb, p.pair)
        self.compression, self.load = comp, load

        up = R[:, :, 1]
        force = load[..., None] * normal  # along the ground normal at each wheel
        lever = mounts - self.position[:, None, :]
        total = force.sum(1) + p.mass[:, None] * np.array([0.0, -G, 0.0])
        torque = np.cross(lever, force).sum(1)
        self._integrate(R, total, torque)

    def _integrate(self, R, force, torque):
        p, dt = self.p, self.DT
        self.velocity = self.velocity + force / p.mass[:, None] * dt
        wb = np.einsum("nji,nj->ni", R, self.angular)
        tb = np.einsum("nji,nj->ni", R, torque)
        wb = wb + dt * (tb - np.cross(wb, p.inertia * wb)) / p.inertia
        self.angular = np.einsum("nij,nj->ni", R, wb)
        self.position = self.position + self.velocity * dt
        self.rotation = integrate(self.rotation, self.angular, dt)
