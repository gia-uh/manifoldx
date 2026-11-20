"""Contacts by impulse: the body's footprint against vertical walls given as (x, z) segments, and the body
box's corners against the ground. Each contact gets an impulse at its point, with restitution and friction,
and the overlap is pushed out."""

from __future__ import annotations

from math import hypot

import numpy as np

from .body import cross

RESTITUTION = 0.2
FRICTION = 0.4
STEP = 0.25  # metres: the most any vehicle moves between two wall tests
ROUNDS = 3  # contacts resolved per test, deepest first; a corner may need a second push
RESTING = 1.0  # m/s: slower closing contacts do not bounce
SLOP = 0.01  # metres of overlap left alone so resting contacts do not jitter
PUSH = 0.8  # share of the rest of the overlap pushed out each time


def within(walls: np.ndarray, centres: np.ndarray, reach: np.ndarray) -> np.ndarray:
    """The walls (M, 2, 2), or (N, M, 2, 2) per vehicle, that come within reach of any vehicle's centre, as one
    shared (K, 2, 2) array: a car only ever touches the few segments around it."""
    w = walls.reshape(-1, 2, 2)
    a, ab = w[:, 0], w[:, 1] - w[:, 0]
    rel = centres[:, None, :] - a[None]  # (N, M, 2)
    t = np.clip(np.sum(rel * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-12), 0.0, 1.0)
    gap = np.linalg.norm(rel - ab * t[..., None], axis=-1)
    return w[np.any(gap <= reach[:, None], 0)]


def footprint_hit(walls, centre, heading, half):
    """The deepest overlap of each footprint with any wall, by separating axes.

    walls (N, M, 2, 2) or (M, 2, 2) in world (x, z); centre (N, 2); heading (N, 2) unit; half (N, 2) half length
    and half width. Returns depth (N,), the normal (N, 2) the car must move along, and the contact point (N, 2)."""
    n = len(centre)
    w = np.broadcast_to(walls, (n,) + walls.shape[-3:]) if walls.ndim == 3 else walls
    side = np.stack([-heading[:, 1], heading[:, 0]], -1)
    rel = w - centre[:, None, None, :]
    # the walls in car axes: (N, M, 2 ends, 2) with u along the car and v across it
    u = np.einsum("nmei,ni->nme", rel, heading)
    v = np.einsum("nmei,ni->nme", rel, side)
    d = np.stack([u[..., 1] - u[..., 0], v[..., 1] - v[..., 0]], -1)
    length = np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    nrm = np.stack([-d[..., 1], d[..., 0]], -1) / length
    axes = np.stack([np.broadcast_to([1.0, 0.0], nrm.shape), np.broadcast_to([0.0, 1.0], nrm.shape), nrm], -2)
    hx, hz = half[:, 0][:, None, None], half[:, 1][:, None, None]
    r = hx * np.abs(axes[..., 0]) + hz * np.abs(axes[..., 1])  # (N, M, 3)
    proj = np.einsum("nmei,nmai->nmae", _ends(u, v), axes)  # (N, M, 3 axes, 2 ends)
    smin, smax = proj.min(-1), proj.max(-1)
    push_neg = r - smin  # move the car along -axis by this much
    push_pos = smax + r  # or along +axis by this much
    depth_axes = np.minimum(push_neg, push_pos)
    sign = np.where(push_neg <= push_pos, -1.0, 1.0)
    k = np.argmin(depth_axes, -1)
    depth = np.take_along_axis(depth_axes, k[..., None], -1)[..., 0]
    overlap = np.all(depth_axes > 0, -1)
    depth = np.where(overlap, depth, -np.inf)
    j = np.argmax(depth, -1)
    rows = np.arange(n)
    best, kk = depth[rows, j], k[rows, j]
    axis = axes[rows, j, kk] * sign[rows, j, kk][:, None]  # in car axes, the way the car must move
    # the contact point, in car axes: a car corner for the wall's face, a wall end for the car's sides
    corner = -np.sign(axis) * half
    corner = np.where(np.abs(axis) < 1e-9, 0.0, corner)
    ends = np.stack([u[rows, j], v[rows, j]], -1)  # (N, 2 ends, 2)
    deepest_end = ends[rows, np.argmin(np.einsum("nei,ni->ne", ends, axis), -1)]
    point_car = np.where((kk == 2)[:, None], corner, np.clip(deepest_end, -half, half))
    normal = axis[:, :1] * heading + axis[:, 1:] * side
    point = centre + point_car[:, :1] * heading + point_car[:, 1:] * side
    return best, normal, point


def box_hit(ca, ua, ha, cb, ub, hb):
    """Two footprints by separating axes, in plain Python: centres (x, z), headings (x, z) and half length and
    width of a and b. None when apart; otherwise the normal a must move along, the contact point (a corner of the
    box that goes into the other's face) and the depth."""
    la, lb = hypot(*ua), hypot(*ub)
    ua, ub = (ua[0] / la, ua[1] / la), (ub[0] / lb, ub[1] / lb)
    axes_a, axes_b = (ua, (-ua[1], ua[0])), (ub, (-ub[1], ub[0]))
    d = (ca[0] - cb[0], ca[1] - cb[1])
    best = None
    for owner, axis in [(0, ax) for ax in axes_a] + [(1, ax) for ax in axes_b]:
        ra = ha[0] * abs(axes_a[0][0] * axis[0] + axes_a[0][1] * axis[1]) + ha[1] * abs(
            axes_a[1][0] * axis[0] + axes_a[1][1] * axis[1])
        rb = hb[0] * abs(axes_b[0][0] * axis[0] + axes_b[0][1] * axis[1]) + hb[1] * abs(
            axes_b[1][0] * axis[0] + axes_b[1][1] * axis[1])
        along = d[0] * axis[0] + d[1] * axis[1]
        depth = ra + rb - abs(along)
        if depth <= 0:
            return None
        if best is None or depth < best[0]:
            best = (depth, owner, axis if along >= 0 else (-axis[0], -axis[1]))
    depth, owner, n = best

    def corners(c, axes, h):
        return [(c[0] + sx * h[0] * axes[0][0] + sz * h[1] * axes[1][0],
                 c[1] + sx * h[0] * axes[0][1] + sz * h[1] * axes[1][1]) for sx in (-1, 1) for sz in (-1, 1)]

    if owner == 1:  # a face of b: a's corner deepest into b
        point = min(corners(ca, axes_a, ha), key=lambda q: q[0] * n[0] + q[1] * n[1])
    else:  # a face of a: b's corner deepest into a
        point = max(corners(cb, axes_b, hb), key=lambda q: q[0] * n[0] + q[1] * n[1])
    return n, point, depth


def _ends(u, v):
    return np.stack([u, v], -1)  # (N, M, 2 ends, 2)


def impulse(v, w, inv_mass, inv_inertia, lever, normal, active):
    """Impulse (N, 3) at lever from the centre of mass along normal, with restitution and Coulomb friction,
    for the bodies where active; zero elsewhere or where the point already separates."""
    vp = v + cross(w, lever)
    vn = np.sum(vp * normal, -1)
    rn = cross(lever, normal)
    k_n = inv_mass + np.sum(normal * cross(np.einsum("nij,nj->ni", inv_inertia, rn), lever), -1)
    bounce = np.where(vn < -RESTING, RESTITUTION, 0.0)
    jn = np.where(active & (vn < 0), -(1 + bounce) * vn / k_n, 0.0)
    tangent = vp - vn[:, None] * normal
    speed_t = np.linalg.norm(tangent, axis=-1)
    t = tangent / np.maximum(speed_t, 1e-9)[:, None]
    rt = cross(lever, t)
    k_t = inv_mass + np.sum(t * cross(np.einsum("nij,nj->ni", inv_inertia, rt), lever), -1)
    jt = np.minimum(speed_t / k_t, FRICTION * jn)
    return jn[:, None] * normal - jt[:, None] * t
