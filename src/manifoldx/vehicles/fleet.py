"""Vehicles: N rigid bodies on ray wheels, stepped together at a fixed 120 Hz."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import powertrain, suspension, tyre
from .body import cross, integrate, matrices, yaw_quaternion
from .spec import VehicleSpec
from .state import pack

G = 9.81
AIR = 1.2  # kg/m³


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
        self.gear, self.shift_timer, self.rpm = np.ones(n, int), np.zeros(n), np.zeros(n)
        self.drive_cut = np.zeros(n)  # share of the drive the assists take away, 0..1

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
        self.gear[i], self.shift_timer[i], self.rpm[i] = 1, 0.0, self.p.engine[i, 0]

    def angular_body(self) -> np.ndarray:
        """Angular velocity in the car's own axes, (N, 3)."""
        return np.einsum("nji,nj->ni", matrices(self.rotation), self.angular)

    def step(self, controls: np.ndarray, ground, walls: np.ndarray | None = None) -> None:
        """Advance every vehicle by DT under controls (N, 4): steer, throttle, brake, handbrake."""
        p, dt = self.p, self.DT
        R = matrices(self.rotation)
        down, forward = -R[:, :, 1], R[:, :, 0]
        self._forward = forward
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
        speed = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        drag = -0.5 * AIR * (p.cd * p.area)[:, None] * speed * self.velocity
        total = force.sum(1) + drag + p.mass[:, None] * np.array([0.0, -G, 0.0])
        torque = cross(lever, force).sum(1)
        self._integrate(R, total, torque)

    def _torques(self, controls):
        """Drive and brake torque on each wheel, N·m, after the gearbox picks its gear."""
        p, dt = self.p, self.DT
        throttle, brake, hand = controls[:, 1], controls[:, 2], controls[:, 3]
        v = np.einsum("ni,ni->n", self.velocity, self._forward)
        slow = np.abs(v) < powertrain.REVERSE_BELOW
        back = self.gear < 0
        self.gear = np.where(~back & slow & (brake > 0) & (throttle == 0), -1, self.gear)
        self.gear = np.where((back & slow | (self.gear == 0)) & (throttle > 0) & (brake == 0), 1, self.gear)
        neutral = self.gear == 0
        back = self.gear < 0
        # in reverse the pedals swap: the brake drives backward and the throttle brakes
        pedal, stop = np.where(back, brake, throttle), np.where(back, throttle, brake)

        driven = p.driven & p.wheel_mask
        n_driven = np.maximum(driven.sum(1), 1)
        g = np.clip(self.gear, 1, None) - 1
        ratio = np.where(back, -p.reverse, p.ratios[np.arange(len(g)), g]) * p.final
        wheel_rpm = np.abs((self.omega * driven).sum(1) / n_driven * ratio) * powertrain.RPM
        self.rpm = np.maximum(wheel_rpm, np.where(pedal > 0, powertrain.CLUTCH_RPM, p.engine[:, 0]))
        # the box shifts on road speed, as an automatic reads its output shaft: a spinning wheel does not shift it
        road_rpm = np.abs(v / np.maximum((p.radius * driven).sum(1) / n_driven, 1e-3) * ratio) * powertrain.RPM

        # the automatic box, with a gap between up and down so it does not hunt
        self.shift_timer = np.maximum(self.shift_timer - dt, 0.0)
        up_rpm = powertrain.UP_AT * p.engine[:, 2]
        ready = (self.shift_timer == 0) & ~back
        up = ready & ~neutral & (road_rpm > up_rpm) & (self.gear < p.n_gears)
        lower = p.ratios[np.arange(len(g)), np.maximum(g - 1, 0)]
        down_rpm = road_rpm * lower / np.maximum(p.ratios[np.arange(len(g)), g], 1e-9)
        down = ready & ~up & (self.gear > 1) & (down_rpm < powertrain.DOWN_AT * up_rpm)
        self.gear = self.gear + up - down
        self.shift_timer = np.where(up | down, p.shift_time, self.shift_timer)

        engine = powertrain.torque(p.engine, self.rpm, pedal) * (1 - self.drive_cut)
        engine = np.where((self.shift_timer > 0) | neutral, np.where(neutral, 0.0, np.minimum(engine, 0.0)), engine)
        drive = np.where(driven, (engine * ratio / n_driven)[:, None], 0.0)
        n_hand = np.maximum((p.handbrake & p.wheel_mask).sum(1), 1)
        brakes = (stop * p.brake_torque)[:, None] * p.brake_share
        brakes = brakes + np.where(p.handbrake, (hand * p.brake_torque / n_hand)[:, None], 0.0)
        brakes = brakes + p.rolling[:, None] * self.load * p.radius + self.extra_brake
        return drive, brakes

    def _steer(self, controls):
        """Each wheel's steering angle, positive to the right (toward the car's +z), with Ackermann geometry."""
        return powertrain.ackermann(controls[:, 0], self.p.mount, self.p.steered, self.p.max_steer)

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
