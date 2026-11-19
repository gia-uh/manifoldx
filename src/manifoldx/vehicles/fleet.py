"""Vehicles: N rigid bodies on ray wheels, stepped together at a fixed 120 Hz."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import contacts, powertrain, suspension, tyre
from .assists import ASSISTS, apply_assists
from .body import cross, integrate, matrices, slerp, yaw_quaternion
from .spec import VehicleSpec
from .state import pack

G = 9.81
AIR = 1.2  # kg/m³


class Vehicles:
    DT = 1 / 120
    MAX_FRAME = 0.05  # seconds: a longer frame runs as this long, as the game did before

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
        self.impulse = np.zeros(n)  # N·s of every contact impulse in the last step
        self.brake_scale = np.ones((n, w))  # share of the pedal brake each wheel gets: the anti-lock lowers it
        self.lit = np.zeros((n, 4), bool)  # which assists corrected in the last step
        self._acc = 0.0
        self._before = (self.position.copy(), self.rotation.copy())

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
        self._before[0][i], self._before[1][i] = self.position[i], self.rotation[i]

    def pick_gear(self, i: int) -> int:
        """Put vehicle i in the gear an automatic would hold at its road speed, as after a launch; returns it."""
        p = self.p
        v = abs(float(self.velocity[i] @ matrices(self.rotation[i : i + 1])[0][:, 0]))
        driven = p.driven[i] & p.wheel_mask[i]
        radius = float(p.radius[i, driven].mean()) if driven.any() else 0.3
        gear = 1
        for g in range(1, int(p.n_gears[i]) + 1):
            gear = g
            rpm = v / radius * p.ratios[i, g - 1] * p.final[i] * powertrain.RPM
            if rpm < powertrain.UP_AT * p.engine[i, 2]:
                break
        self.gear[i] = gear
        return gear

    def advance(self, controls: np.ndarray, ground, walls, dt: float, strengths: np.ndarray | None = None) -> int:
        """Run as many DT steps as dt (cut at MAX_FRAME) and what was left over from the last call add up to,
        each after the assists of the given strengths; returns how many ran."""
        self._acc += min(dt, self.MAX_FRAME)
        n = 0
        while self._acc >= self.DT - 1e-12:
            c = apply_assists(self, controls, strengths) if strengths is not None else controls
            self._before = (self.position.copy(), self.rotation.copy())
            self.step(c, ground, walls)
            self._acc -= self.DT
            n += 1
        return n

    def pose(self, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Position (N, 3) and rotation (N, 4) between the last two steps; alpha defaults to the time left over."""
        a = min(max(self._acc / self.DT if alpha is None else alpha, 0.0), 1.0)
        p0, q0 = self._before
        return p0 + (self.position - p0) * a, slerp(q0, self.rotation, a)

    def telemetry(self, i: int) -> dict:
        """Vehicle i's numbers for a HUD, a sound or an agent: speeds in m/s, loads in N, slips, spins, travel."""
        R = matrices(self.rotation[i : i + 1])[0]
        v = R.T @ self.velocity[i]
        w = self.p.wheel_mask[i]
        return {
            "speed": float(v[0]), "lateral": float(v[2]), "gear": int(self.gear[i]), "rpm": float(self.rpm[i]),
            "load": self.load[i, w].tolist(), "slip": self.slip[i, w].tolist(), "omega": self.omega[i, w].tolist(),
            "travel": (self.compression[i, w]).tolist(), "steer": self.steer_angle[i, w].tolist(),
            "impulse": float(self.impulse[i]), "lit": {a: bool(self.lit[i, k]) for k, a in enumerate(ASSISTS)},
        }

    def model_transform(self, i: int, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Where vehicle i's drawn model goes: its origin's position and the rotation, between the last steps."""
        pos, rot = self.pose(alpha)
        R = matrices(rot[i : i + 1])[0]
        return pos[i] + R @ self.p.model_origin[i], rot[i]

    def travel(self, i: int) -> np.ndarray:
        """How far each of vehicle i's wheels has risen from its rest height, metres."""
        w = self.p.wheel_mask[i]
        return self.compression[i, w] - self.p.rest[i, w]

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
        upright = (-down[:, 1] > 0.3)[:, None]  # past about 72° of tilt the rays point away from the ground
        comp = np.where(upright, np.maximum(comp, 0.0), 0.0)
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
        self._integrate(R, total, torque, ground, walls)

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
        engine = np.where(engine > 0, engine * p.efficiency, engine)  # losses on drive, not on engine braking
        drive = np.where(driven, (engine * ratio / n_driven)[:, None], 0.0)
        n_hand = np.maximum((p.handbrake & p.wheel_mask).sum(1), 1)
        brakes = (stop * p.brake_torque)[:, None] * p.brake_share * self.brake_scale
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

    def _integrate(self, R, force, torque, ground, walls):
        p, dt = self.p, self.DT
        self.velocity = self.velocity + force / p.mass[:, None] * dt
        wb = np.einsum("nji,nj->ni", R, self.angular)
        tb = np.einsum("nji,nj->ni", R, torque)
        wb = wb + dt * (tb - cross(wb, p.inertia * wb)) / p.inertia
        self.angular = np.einsum("nij,nj->ni", R, wb)
        self.rotation = integrate(self.rotation, self.angular, dt)
        self.impulse = np.zeros(len(p.mass))
        R = matrices(self.rotation)
        inv_inertia = np.einsum("nij,nj,nkj->nik", R, 1 / p.inertia, R)
        speed = np.linalg.norm(self.velocity, axis=1).max(initial=0.0)
        n_sub = max(1, int(np.ceil(speed * dt / contacts.STEP)))
        if walls is not None and len(walls):
            walls = contacts.within(np.asarray(walls, float), self.position[:, [0, 2]],
                                    np.hypot(p.size[:, 0], p.size[:, 2]) / 2 + speed * dt + 0.5)
        # the body's corners only matter tilted past 30°, or with a wheel in the air or on its bump stop
        loose = (R[:, 1, 1] < 0.87) | np.any(p.wheel_mask & ((self.load <= 0) | (self.compression >= p.travel)), 1)
        for _ in range(n_sub):
            self.position = self.position + self.velocity * dt / n_sub
            if walls is not None and len(walls):
                self._walls(R, inv_inertia, walls)
            if np.any(loose):
                self._body_on_ground(R, inv_inertia, ground)

    def _apply(self, inv_inertia, lever, normal, depth):
        """One contact per vehicle: impulse at lever along normal where depth > 0, then push the overlap out."""
        p = self.p
        active = depth > 0
        j = contacts.impulse(self.velocity, self.angular, 1 / p.mass, inv_inertia, lever, normal, active)
        self.velocity = self.velocity + j / p.mass[:, None]
        self.angular = self.angular + np.einsum("nij,nj->ni", inv_inertia, cross(lever, j))
        push = np.maximum(depth - contacts.SLOP, 0.0) * contacts.PUSH
        self.position = self.position + normal * np.where(active, push, 0.0)[:, None]
        self.impulse = self.impulse + np.linalg.norm(j, axis=1)

    def _walls(self, R, inv_inertia, walls):
        p = self.p
        heading = R[:, [0, 2], 0]
        heading = heading / np.maximum(np.linalg.norm(heading, axis=1, keepdims=True), 1e-9)
        half = p.size[:, [0, 2]] / 2
        for _ in range(contacts.ROUNDS):
            depth, normal, point = contacts.footprint_hit(walls, self.position[:, [0, 2]], heading, half)
            if not np.any(depth > 0):
                return
            n3 = np.stack([normal[:, 0], np.zeros(len(normal)), normal[:, 1]], -1)
            lever = np.stack([point[:, 0] - self.position[:, 0], np.zeros(len(point)),
                              point[:, 1] - self.position[:, 2]], -1)
            self._apply(inv_inertia, lever, n3, depth)

    def _body_on_ground(self, R, inv_inertia, ground):
        """The body box's eight corners against the ground: a car on its side or roof rests on them."""
        p = self.p
        lo = np.min(np.where(p.wheel_mask, p.mount[..., 1], np.inf), 1)
        hx, hz = p.size[:, 0] / 2, p.size[:, 2] / 2
        corners = np.stack([np.stack([sx * hx, y, sz * hz], -1) for sx in (-1, 1) for sz in (-1, 1)
                            for y in (lo, lo + p.size[:, 1])], 1)  # (N, 8, 3) car axes
        world = self.position[:, None, :] + np.einsum("nij,nkj->nki", R, corners)
        h = ground.height(world[..., 0].ravel(), world[..., 2].ravel(), world[..., 1].ravel()).reshape(world.shape[:2])
        if not np.any(h > world[..., 1]):
            return
        up = np.tile([0.0, 1.0, 0.0], (len(h), 1))
        for _ in range(2):  # sequential impulses: every corner, twice, so resting corners share the weight
            for k in range(corners.shape[1]):
                point = self.position + np.einsum("nij,nj->ni", R, corners[:, k])
                self._apply(inv_inertia, point - self.position, up, h[:, k] - point[:, 1])
