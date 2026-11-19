"""One vehicle stepped in plain Python floats: the same physics as the vectorised Vehicles.step and
apply_assists, without numpy's fixed cost per call, which dominates when N is 1.

Every function here mirrors one in fleet.py, assists.py, suspension.py, tyre.py, powertrain.py or contacts.py;
a change to the physics goes in both, and tests/vehicles/test_single.py drives both side by side."""

from __future__ import annotations

from math import atan, atan2, ceil, cos, hypot, inf, sin, sqrt, tan

import numpy as np

from . import assists as A
from . import contacts as C
from . import powertrain as P
from . import suspension as S
from . import tyre as T

G = 9.81
AIR = 1.2


class _Constants:
    """Vehicle 0's spec as Python floats and lists, one entry per real wheel."""

    def __init__(self, p):
        idx = [j for j in range(p.wheel_mask.shape[1]) if p.wheel_mask[0, j]]
        self.idx = idx
        self.full = len(idx) == p.wheel_mask.shape[1]
        self.W = p.wheel_mask.shape[1]

        def row(a):
            return [float(a[0, j]) for j in idx]

        def flags(a):
            return [bool(a[0, j]) for j in idx]

        self.mount = [tuple(float(v) for v in p.mount[0, j]) for j in idx]
        self.radius, self.travel, self.spring, self.damper = row(p.radius), row(p.travel), row(p.spring), row(p.damper)
        self.arb, self.share, self.w_inertia = row(p.arb), row(p.brake_share), row(p.wheel_inertia)
        self.kpeak, self.apeak = row(p.kappa_peak), row(p.alpha_peak)
        self.tyre = [tuple(float(v) for v in p.tyre[0, j]) for j in idx]
        self.driven, self.steered = flags(p.driven), flags(p.steered)
        self.handbrake, self.front = flags(p.handbrake), flags(p.front)
        self.pair = [idx.index(int(p.pair[0, j])) for j in idx]
        self.mass, self.inertia = float(p.mass[0]), tuple(float(v) for v in p.inertia[0])
        self.size = tuple(float(v) for v in p.size[0])
        self.engine = tuple(float(v) for v in p.engine[0])
        self.ratios, self.n_gears = [float(v) for v in p.ratios[0]], int(p.n_gears[0])
        self.final, self.reverse = float(p.final[0]), float(p.reverse[0])
        self.shift_time, self.efficiency = float(p.shift_time[0]), float(p.efficiency[0])
        self.drag = 0.5 * AIR * float(p.cd[0]) * float(p.area[0])
        self.rolling, self.max_steer = float(p.rolling[0]), float(p.max_steer[0])
        self.brake_torque = float(p.brake_torque[0])
        n = len(idx)
        self.n_driven = max(sum(self.driven), 1)
        self.n_hand = max(sum(self.handbrake), 1)
        self.driven_radius = sum(r for r, d in zip(self.radius, self.driven) if d) / self.n_driven
        # Ackermann geometry, as powertrain.ackermann reads it
        xs, zs = [m[0] for m in self.mount], [m[2] for m in self.mount]
        st = self.steered
        self.ack_base = max((x for x, s in zip(xs, st) if s), default=-inf) - min(
            (x for x, s in zip(xs, st) if not s), default=inf)
        self.ack_track = max((z for z, s in zip(zs, st) if s), default=-inf) - min(
            (z for z, s in zip(zs, st) if s), default=inf)
        self.zsign = [_sign(z) for z in zs]
        # the assists' geometry
        self.front_x = max((x for x, s in zip(xs, st) if s), default=-inf)
        self.base = self.front_x - min(xs)
        self.alpha_peak = max(a if s else 0.0 for a, s in zip(self.apeak, st))
        self.left = [z < 0 for z in zs]
        self.mu_max = max(t[0] for t in self.tyre)
        self.lo = min(m[1] for m in self.mount)
        self.reach = hypot(self.size[0], self.size[2]) / 2 + 0.5
        self.n = n


def _sign(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def _magic(s, b, c, e):
    bs = b * s
    return sin(c * atan(bs - e * (bs - atan(bs))))


def _rotation(q):
    """The columns of q's matrix: the car's x (forward), y (up) and z (right) in the world."""
    x, y, z, w = q
    return ((1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)),
            (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)),
            (2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _constants(fleet) -> _Constants:
    k = fleet.__dict__.get("_single")
    if k is None or k[0] is not fleet.p:
        k = (fleet.p, _Constants(fleet.p))
        fleet._single = k
    return k[1]


def _wheels(a, k):
    """Row 0 of an (N, W) array as a list over the real wheels."""
    r = a[0].tolist()
    return r if k.full else [r[j] for j in k.idx]


def _store(a, k, values):
    if k.full:
        a[0] = values
    else:
        a[0, k.idx] = values


def assists(fleet, controls, strengths) -> np.ndarray:
    """apply_assists for a fleet of one vehicle."""
    k, dt = _constants(fleet), fleet.DT
    s = np.asarray(strengths, float).ravel().tolist()
    k_steer, k_trac, k_abs, k_stab = s[-4:]
    out = np.asarray(controls, float).ravel().tolist()
    ax, ay, az = _rotation(fleet.rotation[0].tolist())
    vel, ang = fleet.velocity[0].tolist(), fleet.angular[0].tolist()
    vx, vz = _dot(ax, vel), _dot(az, vel)
    yaw = -_dot(ay, ang)
    speed = hypot(vx, vz)
    lit = [False, False, False, False]

    steer = out[0]
    wheel = steer * k.max_steer
    beta = atan2(vz + yaw * k.front_x, max(abs(vx), 1.0))
    narrow = 1 - (1 - A.NARROW) * _clip(speed / A.NARROW_AT, 0, 1)
    want = wheel * narrow + A.COUNTER * beta * (1 - abs(steer))
    want = min(max(want, beta - k.alpha_peak), beta + k.alpha_peak)
    if not vx > 3.0:
        want = wheel
    new = wheel + (want - wheel) * k_steer
    lit[0] = abs(new - wheel) > 0.02
    out[0] = _clip(new / k.max_steer, -1, 1)

    slip = fleet.slip[0].tolist()
    if not k.full:
        slip = [slip[j] for j in k.idx]
    over = max((abs(sl[0]) / kp - 1) if d else 0.0 for sl, kp, d in zip(slip, k.kpeak, k.driven))
    cut = k_trac * _clip(over * 2, 0, 1)
    drive_cut = max(cut, float(fleet.drive_cut[0]) - A.RECOVER * dt)
    lit[1] = cut > 0

    omega, scale = _wheels(fleet.omega, k), _wheels(fleet.brake_scale, k)
    fast = abs(vx) > 1.0
    braking = out[2] > 0
    any_lock = False
    for j in range(k.n):
        locking = braking and (slip[j][0] < -1.1 * k.kpeak[j] or (omega[j] == 0 and fast))
        if locking:
            scale[j] = scale[j] * (1 - (1 - A.ABS_STEP) * k_abs)
            any_lock = True
        else:
            scale[j] = min(scale[j] + A.ABS_RECOVER * dt, 1.0)
    _store(fleet.brake_scale, k, scale)
    lit[2] = any_lock and k_abs > 0

    delta = out[0] * k.max_steer
    target = vx * tan(delta) / max(k.base, 0.5)
    limit = k.mu_max * G / max(speed, 1.0)
    target = min(max(target, -limit), limit)
    error = yaw - target
    acting = speed > 4.0 and abs(error) > A.YAW_DEADBAND and k_stab > 0
    extra = [0.0] * k.n
    if acting:
        torque = k_stab * min(A.YAW_GAIN * (abs(error) - A.YAW_DEADBAND), A.YAW_MAX)
        turning_right = target > 0 if abs(target) > 0.02 else yaw > 0
        oversteer = abs(yaw) > abs(target)
        for j in range(k.n):
            outside = k.left[j] if turning_right else not k.left[j]
            pick = (k.front[j] and outside) if oversteer else (not k.front[j] and not outside)
            if pick:
                extra[j] = torque
        if not oversteer:
            drive_cut = max(drive_cut, 0.5 * k_stab)
    _store(fleet.extra_brake, k, extra)
    fleet.drive_cut[0] = drive_cut
    lit[3] = acting
    fleet.lit[0] = lit
    return np.array([out])


def step(fleet, controls, ground, walls) -> None:
    """Vehicles.step for a fleet of one vehicle."""
    k, dt = _constants(fleet), fleet.DT
    n = k.n
    c = np.asarray(controls, float).ravel().tolist()
    pos, vel, ang = fleet.position[0].tolist(), fleet.velocity[0].tolist(), fleet.angular[0].tolist()
    ax, ay, az = _rotation(fleet.rotation[0].tolist())
    fleet._forward = np.array([ax])
    near = _wheels(fleet.ground_y, k)

    # suspension.contact: every ground read in one call
    mounts = []
    for m in k.mount:
        mounts.append((pos[0] + ax[0] * m[0] + ay[0] * m[1] + az[0] * m[2],
                       pos[1] + ax[1] * m[0] + ay[1] * m[1] + az[1] * m[2],
                       pos[2] + ax[2] * m[0] + ay[2] * m[1] + az[2] * m[2]))
    xs, zs = [], []
    for s in S.SAMPLES:
        for m, r in zip(mounts, k.radius):
            xs.append(m[0] + ax[0] * r * s)
            zs.append(m[2] + ax[2] * r * s)
    for m in mounts:
        xs.append(m[0] + S.NORMAL_STEP)
        zs.append(m[2])
    for m in mounts:
        xs.append(m[0])
        zs.append(m[2] + S.NORMAL_STEP)
    h = np.asarray(ground.height(np.array(xs), np.array(zs), np.array(near * (len(S.SAMPLES) + 2))), float)
    h = h.ravel().tolist()
    lift = [sqrt(1 - s * s) for s in S.SAMPLES]
    mid = S.SAMPLES.index(0.0)
    tilt = max(ay[1], 0.2)
    upright = ay[1] > 0.3
    dist, heights, normal, comp = [], [], [], []
    for j in range(n):
        best_k, best = 0, h[j] + k.radius[j] * lift[0]
        for kk in range(1, len(S.SAMPLES)):
            centre = h[kk * n + j] + k.radius[j] * lift[kk]
            if centre > best:
                best_k, best = kk, centre
        heights.append(h[best_k * n + j])
        d = (mounts[j][1] - best) / tilt
        dist.append(d)
        h0 = h[mid * n + j]
        nx, nz = -(h[len(S.SAMPLES) * n + j] - h0) / S.NORMAL_STEP, -(h[(len(S.SAMPLES) + 1) * n + j] - h0) / S.NORMAL_STEP
        ln = sqrt(nx * nx + 1.0 + nz * nz)
        normal.append((nx / ln, 1.0 / ln, nz / ln))
        comp.append(max(k.travel[j] - d, 0.0) if upright else 0.0)

    # suspension.spring_force
    prev = _wheels(fleet.compression, k)
    load = []
    for j in range(n):
        cj = comp[j]
        if cj > 0:
            rate = _clip((cj - prev[j]) / dt, -S.DAMPER_SPEED, S.DAMPER_SPEED)
            f = (k.spring[j] * cj + k.spring[j] * S.BUMP_STOP * max(cj - k.travel[j], 0.0) + k.damper[j] * rate
                 + k.arb[j] * (cj - comp[k.pair[j]]))
            load.append(max(f, 0.0))
        else:
            load.append(0.0)
    _store(fleet.ground_y, k, heights)
    _store(fleet.compression, k, comp)
    _store(fleet.load, k, load)

    down = (-ay[0], -ay[1], -ay[2])
    contact = []
    for j in range(n):
        m, d, nn, r = mounts[j], dist[j], normal[j], k.radius[j]
        contact.append((m[0] + down[0] * d - nn[0] * r, m[1] + down[1] * d - nn[1] * r, m[2] + down[2] * d - nn[2] * r))
    tyre = _tyres(fleet, k, c, ground, ax, az, normal, contact, load, pos, vel, ang, dt)

    fx = fy = fz = tx = ty = tz = 0.0
    for j in range(n):
        nn, ld, t = normal[j], load[j], tyre[j]
        f = (ld * nn[0] + t[0], ld * nn[1] + t[1], ld * nn[2] + t[2])
        lever = (contact[j][0] - pos[0], contact[j][1] - pos[1], contact[j][2] - pos[2])
        fx, fy, fz = fx + f[0], fy + f[1], fz + f[2]
        q = _cross(lever, f)
        tx, ty, tz = tx + q[0], ty + q[1], tz + q[2]
    speed = sqrt(_dot(vel, vel))
    drag = -k.drag * speed
    total = (fx + drag * vel[0], fy + drag * vel[1] - k.mass * G, fz + drag * vel[2])
    _integrate(fleet, k, (ax, ay, az), total, (tx, ty, tz), ground, walls, pos, vel, ang, load, comp, dt)


def _steer(k, steer):
    """powertrain.ackermann for one vehicle."""
    outer = abs(steer) * k.max_steer
    if not abs(steer) > 1e-6:
        return [0.0] * k.n
    radius = k.ack_base / tan(max(outer, 1e-6)) - k.ack_track / 2
    inner = atan(k.ack_base / max(radius - k.ack_track / 2, 1e-6))
    side = _sign(steer)
    return [((inner if zs == side else outer) * side if st else 0.0) for zs, st in zip(k.zsign, k.steered)]


def _tyres(fleet, k, c, ground, ax, az, normal, contact, load, pos, vel, ang, dt):
    n = k.n
    steer = _steer(k, c[0])
    _store(fleet.steer_angle, k, steer)
    mu_ground = np.asarray(ground.grip(np.array([p[0] for p in contact]), np.array([p[2] for p in contact])),
                           float).ravel().tolist()
    heading, side, v_long, v_lat, mu = [], [], [], [], []
    for j in range(n):
        cs, sn = cos(steer[j]), sin(steer[j])
        nn = normal[j]
        hd = (ax[0] * cs + az[0] * sn, ax[1] * cs + az[1] * sn, ax[2] * cs + az[2] * sn)
        dn = _dot(hd, nn)
        hd = (hd[0] - dn * nn[0], hd[1] - dn * nn[1], hd[2] - dn * nn[2])
        ln = max(sqrt(_dot(hd, hd)), 1e-9)
        hd = (hd[0] / ln, hd[1] / ln, hd[2] / ln)
        sd = _cross(hd, nn)
        w = _cross(ang, (contact[j][0] - pos[0], contact[j][1] - pos[1], contact[j][2] - pos[2]))
        v = (vel[0] + w[0], vel[1] + w[1], vel[2] + w[2])
        heading.append(hd)
        side.append(sd)
        v_long.append(_dot(v, hd))
        v_lat.append(_dot(v, sd))
        mu.append(k.tyre[j][0] * mu_ground[j])

    drive, brake = _torques(fleet, k, c, vel, ax, load, dt)
    slip = fleet.slip[0].tolist()
    if not k.full:
        slip = [slip[j] for j in k.idx]
    omega, turn = _wheels(fleet.omega, k), _wheels(fleet.wheel_turn, k)
    relax, h = T.RELAX, 1e-4
    out, new_slip = [], []
    for j in range(n):
        r, t, ld, m, vl = k.radius[j], k.tyre[j], load[j], mu[j], v_long[j]
        on = ld > 0
        den = 1 + dt * abs(vl) / relax
        a = (slip[j][0] - dt * vl / relax) / den
        b = dt * r / relax / den
        s0 = a + b * omega[j]
        grip = m * ld
        fx0 = grip * _magic(s0, t[4], t[5], t[6])
        kk = grip * max((_magic(s0 + h, t[4], t[5], t[6]) - _magic(s0 - h, t[4], t[5], t[6])) / (2 * h), 0.0)
        eff = k.w_inertia[j] + (dt * r * kk * b if on else 0.0)
        free = omega[j] + dt * (drive[j] - (r * fx0 if on else 0.0)) / eff
        stop = dt * brake[j] / eff
        om = 0.0 if abs(free) <= stop else free - _sign(free) * stop
        omega[j] = om
        turn[j] = turn[j] + om * dt
        if not on:
            new_slip.append((0.0, 0.0))
            out.append((0.0, 0.0, 0.0))
            continue
        s_long = a + b * om
        s_lat = (slip[j][1] + dt * v_lat[j] / relax) / den
        new_slip.append((s_long, s_lat))
        # tyre.forces
        kn, an = s_long / k.kpeak[j], s_lat / k.apeak[j]
        rho = hypot(kn, an)
        if rho > 1e-12:
            fx = _magic(rho * k.kpeak[j], t[4], t[5], t[6]) * kn / rho * grip
            fy = -_magic(rho * k.apeak[j], t[1], t[2], t[3]) * an / rho * grip
        else:
            fx = fy = 0.0
        fade = _clip(1 - abs(vl) / T.V_MIN, 0.0, 1.0)
        stiff = grip * t[1] * t[2] / relax
        cd = 2 * T.LOW_SPEED_DAMPING * sqrt(stiff * ld / G) * fade
        fx = fx + cd * (om * r - vl)
        fy = fy - cd * v_lat[j]
        total = hypot(fx, fy)
        if total > grip:
            cap = grip / max(total, 1e-9)
            fx, fy = fx * cap, fy * cap
        hd, sd = heading[j], side[j]
        out.append((fx * hd[0] + fy * sd[0], fx * hd[1] + fy * sd[1], fx * hd[2] + fy * sd[2]))
    _store(fleet.omega, k, omega)
    _store(fleet.wheel_turn, k, turn)
    if k.full:
        fleet.slip[0] = new_slip
    else:
        fleet.slip[0, k.idx] = new_slip
    return out


def _torques(fleet, k, c, vel, forward, load, dt):
    """Vehicles._torques for one vehicle: (drive, brake) per wheel."""
    throttle, brake, hand = c[1], c[2], c[3]
    v = _dot(vel, forward)
    slow = abs(v) < P.REVERSE_BELOW
    gear = int(fleet.gear[0])
    back = gear < 0
    if not back and slow and brake > 0 and throttle == 0:
        gear = -1
    if ((back and slow) or gear == 0) and throttle > 0 and brake == 0:
        gear = 1
    neutral = gear == 0
    back = gear < 0
    pedal, stop = (brake, throttle) if back else (throttle, brake)

    omega = _wheels(fleet.omega, k)
    g = max(gear, 1) - 1
    ratio = (-k.reverse if back else k.ratios[g]) * k.final
    spin = sum(o for o, d in zip(omega, k.driven) if d)
    wheel_rpm = abs(spin / k.n_driven * ratio) * P.RPM
    idle, t_rpm, p_rpm, idle_t, peak_t, power_t, e_brake = k.engine
    rpm = max(wheel_rpm, P.CLUTCH_RPM if pedal > 0 else idle)
    road_rpm = abs(v / max(k.driven_radius, 1e-3) * ratio) * P.RPM

    timer = max(float(fleet.shift_timer[0]) - dt, 0.0)
    up_rpm = P.UP_AT * p_rpm
    ready = timer == 0 and not back
    up = ready and not neutral and road_rpm > up_rpm and gear < k.n_gears
    lower = k.ratios[max(g - 1, 0)]
    down_rpm = road_rpm * lower / max(k.ratios[g], 1e-9)
    down = ready and not up and gear > 1 and down_rpm < P.DOWN_AT * up_rpm
    gear = gear + up - down
    if up or down:
        timer = k.shift_time
    fleet.gear[0], fleet.shift_timer[0], fleet.rpm[0] = gear, timer, rpm

    # powertrain.torque
    top = p_rpm * P.REDLINE
    r = _clip(rpm, idle, top)
    if r >= p_rpm:
        x0, x1, y0, y1 = p_rpm, top, power_t, 0.0
    elif r >= t_rpm:
        x0, x1, y0, y1 = t_rpm, p_rpm, peak_t, power_t
    else:
        x0, x1, y0, y1 = idle, t_rpm, idle_t, peak_t
    curve = y0 + (y1 - y0) * (r - x0) / max(x1 - x0, 1e-9)
    engine = (curve * pedal - e_brake * max(rpm - idle, 0.0) * (1 - pedal)) * (1 - float(fleet.drive_cut[0]))
    if timer > 0 or neutral:
        engine = 0.0 if neutral else min(engine, 0.0)
    if engine > 0:
        engine = engine * k.efficiency
    per = engine * ratio / k.n_driven
    drive = [per if d else 0.0 for d in k.driven]
    scale, extra = _wheels(fleet.brake_scale, k), _wheels(fleet.extra_brake, k)
    hb = hand * k.brake_torque / k.n_hand
    pedal_brake = stop * k.brake_torque
    brakes = [pedal_brake * k.share[j] * scale[j] + (hb if k.handbrake[j] else 0.0)
              + k.rolling * load[j] * k.radius[j] + extra[j] for j in range(k.n)]
    return drive, brakes


def _integrate(fleet, k, R, force, torque, ground, walls, pos, vel, ang, load, comp, dt):
    ax, ay, az = R
    m = k.mass
    vel = [vel[0] + force[0] / m * dt, vel[1] + force[1] / m * dt, vel[2] + force[2] / m * dt]
    inertia = k.inertia
    wb = (_dot(ax, ang), _dot(ay, ang), _dot(az, ang))
    tb = (_dot(ax, torque), _dot(ay, torque), _dot(az, torque))
    gyro = _cross(wb, (inertia[0] * wb[0], inertia[1] * wb[1], inertia[2] * wb[2]))
    wb = [wb[i] + dt * (tb[i] - gyro[i]) / inertia[i] for i in range(3)]
    ang = [ax[i] * wb[0] + ay[i] * wb[1] + az[i] * wb[2] for i in range(3)]
    # body.integrate
    x, y, z, w = fleet.rotation[0].tolist()
    wx, wy, wz = ang
    hd = 0.5 * dt
    q = (x + hd * (wx * w + wy * z - wz * y), y + hd * (wy * w + wz * x - wx * z),
         z + hd * (wz * w + wx * y - wy * x), w - hd * (wx * x + wy * y + wz * z))
    ln = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    q = (q[0] / ln, q[1] / ln, q[2] / ln, q[3] / ln)
    R = _rotation(q)
    ax, ay, az = R
    # R · diag(1/I) · Rᵀ, row by row
    inv = tuple(tuple(ax[i] * ax[j] / inertia[0] + ay[i] * ay[j] / inertia[1] + az[i] * az[j] / inertia[2] for j in range(3))
                for i in range(3))
    body = _Body(pos, vel, ang, m, inv)
    speed = sqrt(_dot(vel, vel))
    n_sub = max(1, ceil(speed * dt / C.STEP))
    near = None
    if walls is not None and len(walls):
        near = _within(np.asarray(walls, float).reshape(-1, 2, 2), pos[0], pos[2], k.reach + speed * dt)
    loose = ay[1] < 0.87 or any(ld <= 0 or cp >= tr for ld, cp, tr in zip(load, comp, k.travel))
    for _ in range(n_sub):
        f = dt / n_sub
        body.pos = [body.pos[0] + body.vel[0] * f, body.pos[1] + body.vel[1] * f, body.pos[2] + body.vel[2] * f]
        if near:
            _walls(body, k, ax, near)
        if loose:
            _body_on_ground(body, k, R, ground)
    fleet.position[0], fleet.velocity[0], fleet.angular[0] = body.pos, body.vel, body.ang
    fleet.rotation[0] = q
    fleet.impulse[0] = body.impulse


class _Body:
    __slots__ = ("pos", "vel", "ang", "inv_mass", "inv", "impulse")

    def __init__(self, pos, vel, ang, mass, inv):
        self.pos, self.vel, self.ang, self.inv_mass, self.inv, self.impulse = pos, vel, ang, 1 / mass, inv, 0.0


def _mul(m, v):
    return (_dot(m[0], v), _dot(m[1], v), _dot(m[2], v))


def _apply(b, lever, n, depth):
    """Vehicles._apply for one contact of one body, depth > 0; contacts.impulse inlined."""
    vp = [b.vel[i] + c for i, c in enumerate(_cross(b.ang, lever))]
    vn = _dot(vp, n)
    if vn < 0:
        rn = _cross(lever, n)
        k_n = b.inv_mass + _dot(n, _cross(_mul(b.inv, rn), lever))
        jn = -(1 + (C.RESTITUTION if vn < -C.RESTING else 0.0)) * vn / k_n
        tg = (vp[0] - vn * n[0], vp[1] - vn * n[1], vp[2] - vn * n[2])
        speed_t = sqrt(_dot(tg, tg))
        s = max(speed_t, 1e-9)
        t = (tg[0] / s, tg[1] / s, tg[2] / s)
        rt = _cross(lever, t)
        k_t = b.inv_mass + _dot(t, _cross(_mul(b.inv, rt), lever))
        jt = min(speed_t / k_t, C.FRICTION * jn)
        j = (jn * n[0] - jt * t[0], jn * n[1] - jt * t[1], jn * n[2] - jt * t[2])
        b.vel = [b.vel[i] + j[i] * b.inv_mass for i in range(3)]
        dw = _mul(b.inv, _cross(lever, j))
        b.ang = [b.ang[i] + dw[i] for i in range(3)]
        b.impulse += sqrt(_dot(j, j))
    push = max(depth - C.SLOP, 0.0) * C.PUSH
    b.pos = [b.pos[i] + n[i] * push for i in range(3)]


def _within(walls, cx, cz, reach):
    """contacts.within for one centre: the walls within reach, as ((ax, az), (bx, bz)) tuples."""
    a, ab = walls[:, 0], walls[:, 1] - walls[:, 0]
    rx, rz = cx - a[:, 0], cz - a[:, 1]
    t = np.clip((rx * ab[:, 0] + rz * ab[:, 1]) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
    gap = np.hypot(rx - ab[:, 0] * t, rz - ab[:, 1] * t)
    return walls[gap <= reach].tolist()


def _walls(b, k, ax, walls):
    hx, hz = ax[0], ax[2]
    ln = max(hypot(hx, hz), 1e-9)
    heading = (hx / ln, hz / ln)
    half = (k.size[0] / 2, k.size[2] / 2)
    for _ in range(C.ROUNDS):
        hit = _footprint_hit(walls, (b.pos[0], b.pos[2]), heading, half)
        if hit is None:
            return
        depth, normal, point = hit
        _apply(b, (point[0] - b.pos[0], 0.0, point[1] - b.pos[2]), (normal[0], 0.0, normal[1]), depth)


def _footprint_hit(walls, centre, heading, half):
    """contacts.footprint_hit for one footprint: (depth, normal, point) of the deepest overlap, or None."""
    h0, h1 = heading
    s0, s1 = -h1, h0
    hx, hz = half
    best = None
    for (e0, e1) in walls:
        r0x, r0z, r1x, r1z = e0[0] - centre[0], e0[1] - centre[1], e1[0] - centre[0], e1[1] - centre[1]
        u0, v0 = r0x * h0 + r0z * h1, r0x * s0 + r0z * s1
        u1, v1 = r1x * h0 + r1z * h1, r1x * s0 + r1z * s1
        du, dv = u1 - u0, v1 - v0
        ln = max(hypot(du, dv), 1e-9)
        axes = ((1.0, 0.0), (0.0, 1.0), (-dv / ln, du / ln))
        found, kk, sign, ok = inf, 0, 1.0, True
        for a, (p, q) in enumerate(axes):
            r = hx * abs(p) + hz * abs(q)
            pa, pb = u0 * p + v0 * q, u1 * p + v1 * q
            neg, pos = r - min(pa, pb), max(pa, pb) + r
            d = min(neg, pos)
            if not d > 0:
                ok = False
                break
            if d < found:
                found, kk, sign = d, a, (-1.0 if neg <= pos else 1.0)
        if ok and (best is None or found > best[0]):
            best = (found, kk, sign, axes[kk], ((u0, v0), (u1, v1)))
    if best is None:
        return None
    depth, kk, sign, (p, q), ends = best
    axis = (p * sign, q * sign)
    if kk == 2:
        pc = tuple(0.0 if abs(a) < 1e-9 else -_sign(a) * hh for a, hh in zip(axis, half))
    else:
        e = ends[0] if ends[0][0] * axis[0] + ends[0][1] * axis[1] <= ends[1][0] * axis[0] + ends[1][1] * axis[1] \
            else ends[1]
        pc = (_clip(e[0], -half[0], half[0]), _clip(e[1], -half[1], half[1]))
    normal = (axis[0] * h0 + axis[1] * s0, axis[0] * h1 + axis[1] * s1)
    point = (centre[0] + pc[0] * h0 + pc[1] * s0, centre[1] + pc[0] * h1 + pc[1] * s1)
    return depth, normal, point


def _body_on_ground(b, k, R, ground):
    """Vehicles._body_on_ground for one body."""
    ax, ay, az = R
    hx, hz = k.size[0] / 2, k.size[2] / 2
    corners = [(sx * hx, y, sz * hz) for sx in (-1, 1) for sz in (-1, 1) for y in (k.lo, k.lo + k.size[1])]
    offs = [tuple(ax[i] * c[0] + ay[i] * c[1] + az[i] * c[2] for i in range(3)) for c in corners]
    pos = b.pos
    world = [(pos[0] + o[0], pos[1] + o[1], pos[2] + o[2]) for o in offs]
    h = np.asarray(ground.height(np.array([w[0] for w in world]), np.array([w[2] for w in world]),
                                 np.array([w[1] for w in world])), float).ravel().tolist()
    if not any(hk > w[1] for hk, w in zip(h, world)):
        return
    up = (0.0, 1.0, 0.0)
    for _ in range(2):
        for o, hk in zip(offs, h):
            depth = hk - (b.pos[1] + o[1])
            if depth > 0:
                _apply(b, o, up, depth)
