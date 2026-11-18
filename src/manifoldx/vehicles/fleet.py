"""Vehicles: N rigid bodies on ray wheels, stepped together at a fixed 120 Hz."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import suspension, tyre
from .body import cross, integrate, matrices, yaw_quaternion
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
        self.omega, self.wheel_turn = np.zeros((n, w)), np.zeros((n, w))
        self.slip = np.zeros((n, w, 2))  # relaxed slip ratio and slip-angle tangent
        self.steer_angle = np.zeros((n, w))
        self.extra_brake = np.zeros((n, w))  # N·m the assists add

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
        self.omega[i] = self.slip[i] = 0.0

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

        force = load[..., None] * normal  # along the ground normal at each wheel
        wheel_centre = mounts + down[:, None, :] * dist[..., None]
        contact_point = wheel_centre - normal * p.radius[..., None]
        force = force + self._tyres(controls, ground, R, normal, contact_point, load)
        lever = contact_point - self.position[:, None, :]
        total = force.sum(1) + p.mass[:, None] * np.array([0.0, -G, 0.0])
        torque = cross(lever, force).sum(1)
        self._integrate(R, total, torque)

    def _torques(self, controls):
        """Drive and brake torque on each wheel, N·m: the powertrain fills these in."""
        z = np.zeros_like(self.omega)
        return z, z + self.extra_brake

    def _steer(self, controls):
        """Each wheel's steering angle, positive to the right (toward the car's +z)."""
        return np.where(self.p.steered, controls[:, :1] * self.p.max_steer[:, None], 0.0)

    def _tyres(self, controls, ground, R, normal, contact_point, load):
        """The tyre forces (N, W, 3) at each contact; updates each wheel's spin and slips."""
        p, dt = self.p, self.DT
        self.steer_angle = self._steer(controls)
        cs, sn = np.cos(self.steer_angle)[..., None], np.sin(self.steer_angle)[..., None]
        heading = R[:, None, :, 0] * cs + R[:, None, :, 2] * sn
        heading = heading - np.sum(heading * normal, -1, keepdims=True) * normal
        heading /= np.maximum(np.linalg.norm(heading, axis=-1, keepdims=True), 1e-9)
        side = cross(heading, normal)  # the wheel's right
        v = self.velocity[:, None, :] + cross(self.angular[:, None, :], contact_point - self.position[:, None, :])
        v_long, v_lat = np.sum(v * heading, -1), np.sum(v * side, -1)
        on = load > 0
        mu = p.tyre[..., 0] * ground.grip(contact_point[..., 0].ravel(), contact_point[..., 2].ravel()).reshape(load.shape)

        # wheel spin, implicit in the tyre's longitudinal stiffness so that a locked or spinning wheel is stable
        drive, brake = self._torques(controls)
        r, inertia = p.radius, p.wheel_inertia
        den = 1 + dt * np.abs(v_long) / tyre.RELAX
        a = (self.slip[..., 0] - dt * v_long / tyre.RELAX) / den
        b = dt * r / tyre.RELAX / den
        s0 = a + b * self.omega
        fx0 = mu * load * tyre.magic(s0, p.tyre[..., 4], p.tyre[..., 5], p.tyre[..., 6])
        k = mu * load * np.maximum(tyre.slope(s0, p.tyre[..., 4], p.tyre[..., 5], p.tyre[..., 6]), 0.0)
        eff = inertia + dt * r * k * b * on
        free = self.omega + dt * (drive - r * fx0 * on) / eff
        stop = dt * brake / eff
        self.omega = np.where(np.abs(free) <= stop, 0.0, free - np.sign(free) * stop)
        self.wheel_turn = self.wheel_turn + self.omega * dt
        s_long = np.where(on, a + b * self.omega, 0.0)
        s_lat = np.where(on, (self.slip[..., 1] + dt * v_lat / tyre.RELAX) / den, 0.0)
        self.slip = np.stack([s_long, s_lat], -1)

        fx, fy = tyre.forces(s_long, s_lat, load, mu, p.tyre, p.kappa_peak, p.alpha_peak)
        # low-speed damping, fading out by V_MIN, so the tread's spring does not ring at rest
        fade = np.clip(1 - np.abs(v_long) / tyre.V_MIN, 0.0, 1.0)
        stiff = mu * load * p.tyre[..., 1] * p.tyre[..., 2] / tyre.RELAX
        c = 2 * tyre.LOW_SPEED_DAMPING * np.sqrt(stiff * load / G) * fade
        fx = fx + c * (self.omega * r - v_long)
        fy = fy - c * v_lat
        total = np.hypot(fx, fy)
        cap = np.where(total > mu * load, mu * load / np.maximum(total, 1e-9), 1.0)
        fx, fy = np.where(on, fx * cap, 0.0), np.where(on, fy * cap, 0.0)
        return fx[..., None] * heading + fy[..., None] * side

    def _integrate(self, R, force, torque):
        p, dt = self.p, self.DT
        self.velocity = self.velocity + force / p.mass[:, None] * dt
        wb = np.einsum("nji,nj->ni", R, self.angular)
        tb = np.einsum("nji,nj->ni", R, torque)
        wb = wb + dt * (tb - cross(wb, p.inertia * wb)) / p.inertia
        self.angular = np.einsum("nij,nj->ni", R, wb)
        self.position = self.position + self.velocity * dt
        self.rotation = integrate(self.rotation, self.angular, dt)
