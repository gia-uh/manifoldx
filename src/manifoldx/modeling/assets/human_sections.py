"""A base body read once into a section table: rings of radii along each chain, a head grid and the rest joints.

A humanoid is built from a table, never from a mesh. The table is read offline from one base body (a mesh an
image-to-mesh model made from drawings of a figure in an A-pose) and committed as package data; a figure at run
time reshapes the table's rings and lofts them (see ``human_skin``).

A chain is an axis from ``start`` to ``end`` with stations along it. Station ``i`` sits at fraction ``t[i]`` of
the axis, its ring centred at offset ``c[i]`` in the chain's (u, v) frame, with K radii at angles
``-pi + 2*pi*k/K`` measured from ``u`` toward ``v``. The head is a grid of radii from its centre by latitude (top
to the neck) and longitude, since a ring centred on the neck cannot follow a chin that overhangs it.

Axes of a figure: metres, Y up, the figure faces +Z, its left side at +X, feet on y = 0, centred on x = z = 0.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from manifoldx.humanoid.proportions import BODIES

CHAINS = {
    "torso": 40,
    "leg_l": 24,
    "leg_r": 24,
    "foot_l": 10,
    "foot_r": 10,
    "arm_l": 28,
    "arm_r": 28,
}
K = 16
HEAD = (12, 16)  # latitudes, longitudes
JOINTS = (
    "pelvis",
    "spine",
    "chest",
    "neck",
    "head",
    "shoulder_l",
    "elbow_l",
    "wrist_l",
    "shoulder_r",
    "elbow_r",
    "wrist_r",
    "hip_l",
    "knee_l",
    "ankle_l",
    "toe_l",
    "hip_r",
    "knee_r",
    "ankle_r",
    "toe_r",
)


def shoulder_rows(k: int) -> int:
    """Torso rings from axilla to acromion at K = `k` (see human_skin's shoulder stitch): 5 at 16, 3 at 8 and 6."""
    return max(3, round(5 * k / 16))

DATA = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Chain:
    start: np.ndarray
    end: np.ndarray
    u: np.ndarray
    v: np.ndarray
    t: np.ndarray
    c: np.ndarray
    r: np.ndarray

    def centres(self) -> np.ndarray:
        """(n, 3) ring centres in the figure's axes."""
        return (
            self.start
            + np.outer(self.t, self.end - self.start)
            + np.outer(self.c[:, 0], self.u)
            + np.outer(self.c[:, 1], self.v)
        )


@dataclass(frozen=True)
class SectionTable:
    stature: float
    chains: dict
    head_center: np.ndarray
    head_r: np.ndarray
    joints: dict
    crotch_y: float
    axilla_y: float
    acromion_y: float


# ---------------------------------------------------------------------------------------------- mesh helpers


def _weld(positions, faces):
    key = np.round(positions / 1e-5).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return positions[first], inverse.reshape(-1)[faces]


def _components(faces) -> np.ndarray:
    """A component label per face, joining faces across edges that exactly two faces share.

    An image-to-mesh model can weld its inner shell to the outer one along a few edges four faces share; those
    edges do not join, so the shells come apart.
    """
    m = len(faces)
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    owner = np.repeat(np.arange(m), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    e, o = edges[order], owner[order]
    same = (e[1:] == e[:-1]).all(1)
    starts = np.flatnonzero(np.r_[True, ~same])
    counts = np.diff(np.r_[starts, len(e)])
    two = starts[counts == 2]
    a, b = o[two], o[two + 1]
    label = np.arange(m)
    while True:
        new = label.copy()
        lo = np.minimum(new[a], new[b])
        np.minimum.at(new, a, lo)
        np.minimum.at(new, b, lo)
        while True:
            jumped = new[new]
            if np.array_equal(jumped, new):
                break
            new = jumped
        if np.array_equal(new, label):
            return label
        label = new


def _outer_shell(positions, faces):
    """The component seen first from the six axis directions (an inner shell and loose bits never win)."""
    comp = _components(faces)
    centroids = positions[faces].mean(1)
    lo, hi = positions.min(0), positions.max(0)
    cell = (hi - lo).max() / 160
    hits = np.zeros(comp.max() + 1, np.int64)
    for axis in range(3):
        a, b = [i for i in range(3) if i != axis]
        ij = np.floor((centroids[:, [a, b]] - lo[[a, b]]) / cell).astype(np.int64)
        key = ij[:, 0] * 1000 + ij[:, 1]
        for sign in (1, -1):
            order = np.lexsort((sign * centroids[:, axis], key))
            k = key[order]
            first = order[np.r_[True, k[1:] != k[:-1]]]
            np.add.at(hits, comp[first], 1)
    keep = comp == hits.argmax()
    used = np.unique(faces[keep])
    remap = np.full(len(positions), -1)
    remap[used] = np.arange(len(used))
    return positions[used], remap[faces[keep]]


def _slice(positions, faces, origin, normal) -> list:
    """Closed loops (each an (m, 3) array) where the plane cuts the mesh."""
    d = (positions - origin) @ normal
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    s = d[faces] > 0
    cut = s.any(1) & ~s.all(1)
    f = faces[cut]
    df = d[f]
    pts, keys = [], []
    for e0, e1 in ((0, 1), (1, 2), (2, 0)):
        crosses = (df[:, e0] > 0) != (df[:, e1] > 0)
        a, b = f[:, e0], f[:, e1]
        with np.errstate(divide="ignore", invalid="ignore"):
            w = df[:, e0] / (df[:, e0] - df[:, e1])
            p = positions[a] + (positions[b] - positions[a]) * w[:, None]
        key = np.minimum(a, b) * (len(positions) + 1) + np.maximum(a, b)
        pts.append(np.where(crosses[:, None], p, np.nan))
        keys.append(np.where(crosses, key, -1))
    pts, keys = np.stack(pts, 1), np.stack(keys, 1)
    point_of, nxt = {}, {}
    for i in range(len(f)):
        ks = [keys[i, j] for j in range(3) if keys[i, j] >= 0]
        if len(ks) != 2:
            continue
        for j in range(3):
            if keys[i, j] >= 0:
                point_of[keys[i, j]] = pts[i, j]
        nxt.setdefault(ks[0], []).append(ks[1])
        nxt.setdefault(ks[1], []).append(ks[0])
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            options = [k for k in nxt[cur] if k != prev]
            if not options or options[0] == start or options[0] in seen:
                break
            prev, cur = cur, options[0]
            seen.add(cur)
            loop.append(cur)
        if len(loop) >= 3:
            loops.append(np.array([point_of[k] for k in loop]))
    return _outermost(loops, origin, normal)


def _inside(point, poly) -> bool:
    x, y = point
    xs, ys = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    crosses = (ys > y) != (yn > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = xs + (y - ys) * (xn - xs) / (yn - ys)
    return bool((crosses & (x < xc)).sum() % 2)


def _outermost(loops, origin, normal) -> list:
    """The loops not inside another (an inner shell cuts into loops nested in the outer one's)."""
    if len(loops) < 2:
        return loops
    ref = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    u, v = _frame(normal, ref)
    flat = [np.c_[(lp - origin) @ u, (lp - origin) @ v] for lp in loops]
    keep = [
        i
        for i, a in enumerate(flat)
        if not any(j != i and _inside(a[0], b) for j, b in enumerate(flat))
    ]
    return [loops[i] for i in keep]


def _frame(axis, ref):
    axis = axis / np.linalg.norm(axis)
    u = ref - axis * (ref @ axis)
    u /= np.linalg.norm(u)
    return u, np.cross(axis, u)


def _ring(loop2, centre=None):
    """K radii of a 2-D loop around its centroid (or `centre`), interpolated by angle with wrap-around."""
    c = loop2.mean(0) if centre is None else centre
    rel = loop2 - c
    ang, rad = np.arctan2(rel[:, 1], rel[:, 0]), np.hypot(rel[:, 0], rel[:, 1])
    o = np.argsort(ang)
    ang, rad = ang[o], rad[o]
    ang = np.r_[ang - 2 * np.pi, ang, ang + 2 * np.pi]
    rad = np.r_[rad, rad, rad]
    theta = -np.pi + 2 * np.pi * np.arange(K) / K
    return c, np.interp(theta, ang, rad)


def _fill(values, ok):
    """Linear interpolation over stations whose read failed."""
    idx = np.arange(len(values))
    if ok.all():
        return values
    out = values.copy()
    for col in range(values.shape[1]):
        out[~ok, col] = np.interp(idx[~ok], idx[ok], values[ok, col])
    return out


def _read_chain(positions, faces, start, end, ts, ref, side=0) -> Chain:
    axis = end - start
    u, v = _frame(axis, ref)
    n = len(ts)
    c, r, ok = np.zeros((n, 2)), np.zeros((n, K)), np.zeros(n, bool)
    near = np.zeros(2)
    unit = axis / np.linalg.norm(axis)
    for i, t in enumerate(ts):
        o = start + axis * t
        # A loop the axis runs through wins over the nearest one: from a ring where two limbs join, the
        # nearest loop below can be the other limb.
        best = None
        for loop in _slice(positions, faces, o, unit):
            p2 = np.c_[(loop - o) @ u, (loop - o) @ v]
            key = (not _inside((0.0, 0.0), p2), np.linalg.norm(p2.mean(0) - near))
            if best is None or key < best[0]:
                best = (key, p2)
        if best is not None:
            best = (best[0][1], best[1])
        if best is None or best[0] > 0.15 or np.ptp(best[1], axis=0).max() < 2e-3:
            continue  # nothing near, or a cut through a lone tip vertex
        c[i], r[i] = _ring(best[1])
        near, ok[i] = c[i], True
    if not ok.any():
        raise ValueError("no section found along a chain; is the mesh a figure in an A-pose?")
    return Chain(start, end, u, v, np.asarray(ts, float), _fill(c, ok), _fill(r, ok))


def _rays(positions, faces, origin, dirs) -> np.ndarray:
    """Distance from `origin` along each direction to the farthest hit (Moller-Trumbore, in chunks)."""
    tri = positions[faces]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    s = origin - tri[:, 0]
    out = np.zeros(len(dirs))
    for i, d in enumerate(dirs):
        p = np.cross(d, e2)
        det = (e1 * p).sum(1)
        good = np.abs(det) > 1e-12
        inv = np.where(good, 1.0 / np.where(good, det, 1), 0)
        uu = (s * p).sum(1) * inv
        q = np.cross(s, e1)
        vv = (q @ d) * inv
        tt = (e2 * q).sum(1) * inv
        hit = good & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (tt > 0)
        out[i] = tt[hit].max() if hit.any() else 0.0
    return out


def _torso_ts(n, t_axilla, t_acromion, rows) -> np.ndarray:
    """n station fractions: `rows` of them from axilla to acromion, three quarters of the rest below, a quarter above.

    The counts depend only on n and rows, so every body puts its shoulder rows at the same ring indices.
    """
    rest = n - rows
    below = round(rest * 0.75)
    above = rest - below
    lo = np.linspace(0.0, t_axilla, below + 1)[:-1]
    mid = np.linspace(t_axilla, t_acromion, rows)
    hi = np.linspace(t_acromion, 1.0, above + 1)[1:]
    return np.r_[lo, mid, hi]


# ----------------------------------------------------------------------------------------------- the reading


def _central_loops(positions, faces, y, half_width):
    loops = _slice(positions, faces, np.array([0.0, y, 0.0]), np.array([0.0, 1.0, 0.0]))
    return [lp for lp in loops if abs(lp[:, 0].mean()) < half_width]


def read_sections(positions, faces, stature: float, body: str = "female") -> SectionTable:
    """Read a base body (an A-pose figure facing -Z, as image-to-mesh models emit it) into a section table."""
    positions, faces = _weld(np.asarray(positions, float), np.asarray(faces, np.int64))
    positions, faces = _outer_shell(positions, faces)
    positions = positions * [-1, 1, -1]  # turn round to face +Z
    faces = faces.copy()
    lo, hi = positions.min(0), positions.max(0)
    positions = (positions - [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]) * (
        stature / (hi[1] - lo[1])
    )
    fr = BODIES[body]
    h = stature

    # Crotch: scanning up, the first height where the two legs' loops have merged into one.
    crotch = None
    for y in np.arange(0.30 * h, 0.62 * h, 0.002 * h):
        if len(_central_loops(positions, faces, y, 0.2 * h)) == 1:
            crotch = y
            break
    if crotch is None:
        raise ValueError("no crotch found: the legs never join")
    # Axilla: scanning up from the waist, the first height where torso and both arms are one loop.
    axilla = None
    for y in np.arange(fr["waist_height"] * h, 0.9 * h, 0.002 * h):
        if len(_slice(positions, faces, np.array([0.0, y, 0.0]), np.array([0.0, 1.0, 0.0]))) == 1:
            axilla = y
            break
    if axilla is None:
        raise ValueError("no axilla found: the arms never join the torso")
    acromion = max(axilla + 0.04 * h, fr["acromial_height"] * h)

    chains = {}
    top = positions[:, 1].max()
    neck_y = fr["cervicale_height"] * h
    gap = 0.006 * h
    t_ax = (axilla - (crotch + gap)) / (neck_y - (crotch + gap))
    t_ac = (acromion - (crotch + gap)) / (neck_y - (crotch + gap))
    chains["torso"] = _read_chain(
        positions,
        faces,
        np.array([0.0, crotch + gap, 0.0]),
        np.array([0.0, neck_y, 0.0]),
        _torso_ts(CHAINS["torso"], t_ax, t_ac, shoulder_rows(K)),
        np.array([1.0, 0, 0]),
    )
    knee_y, ankle_y = fr["knee_height"] * h, 0.045 * h
    joints = {}
    for s, side in (("l", 1), ("r", -1)):
        near_knee = positions[
            (np.abs(positions[:, 1] - 0.29 * h) < 0.01 * h)
            & (positions[:, 0] * side > 0)
            & (np.abs(positions[:, 0]) < 0.2 * h)
        ]
        near_ankle = positions[
            (np.abs(positions[:, 1] - 0.08 * h) < 0.01 * h)
            & (positions[:, 0] * side > 0)
            & (np.abs(positions[:, 0]) < 0.2 * h)
        ]
        a, b = near_knee.mean(0), near_ankle.mean(0)
        a[1], b[1] = 0.29 * h, 0.08 * h
        line = (b - a) / (b[1] - a[1])
        b = a + line * (ankle_y - a[1])
        start = a + line * (crotch - gap - a[1])
        chains[f"leg_{s}"] = _read_chain(
            positions, faces, start, b, np.linspace(0, 1, CHAINS["leg_l"]), np.array([1.0, 0, 0])
        )
        # Foot: rings around an axis along +Z at a quarter of the ankle height, binned from the foot's vertices.
        foot = positions[
            (positions[:, 1] < ankle_y)
            & (positions[:, 0] * side > 0)
            & (np.abs(positions[:, 0]) < 0.2 * h)
        ]
        z0, z1 = foot[:, 2].min(), foot[:, 2].max()
        fy = 0.5 * ankle_y
        fx = foot[:, 0].mean()
        fs, fe = np.array([fx, fy, z0]), np.array([fx, fy, z1])
        chains[f"foot_{s}"] = _binned_chain(foot, fs, fe, CHAINS["foot_l"], np.array([1.0, 0, 0]))
        # Arm: the principal axis of the vertices beyond the shoulder near the line from the axilla to the
        # farthest point out (the hand), refined twice; hips or a belly wider than the shoulders stay out.
        shoulder_x = fr["biacromial_breadth"] / 2 * h
        beyond = positions[
            (positions[:, 0] * side > shoulder_x + 0.02 * h) & (positions[:, 1] > crotch)
        ]
        centre = np.array([side * shoulder_x, axilla, 0.0])
        d = beyond[np.argmax(beyond[:, 0] * side)] - centre
        d /= np.linalg.norm(d)
        for _ in range(3):
            rel = beyond - centre
            along = rel @ d
            arm = beyond[(np.linalg.norm(rel - np.outer(along, d), axis=1) < 0.08 * h)]
            centre = arm.mean(0)
            _, vecs = np.linalg.eigh(np.cov((arm - centre).T))
            d = vecs[:, -1] * (1 if vecs[0, -1] * side > 0 else -1)
        tip = arm[np.argmax((arm - centre) @ d)]
        shoulder = centre + d * ((side * shoulder_x - centre[0]) / d[0])
        arm_start = shoulder + d * 0.04 * h
        chains[f"arm_{s}"] = _read_chain(
            positions,
            faces,
            arm_start,
            tip,
            np.linspace(0, 1, CHAINS["arm_l"]),
            np.array([0.0, 0, 1.0]),
        )
        upper, fore = fr["upper_arm"] * h, fr["forearm"] * h
        joints |= {
            f"shoulder_{s}": shoulder,
            f"elbow_{s}": shoulder + d * upper,
            f"wrist_{s}": shoulder + d * (upper + fore),
            f"hip_{s}": np.array([start[0], fr["trochanter_height"] * h, 0.0]),
            f"knee_{s}": a + line * (knee_y - a[1]),
            f"ankle_{s}": b.copy(),
            f"toe_{s}": np.array([fx, fy, z0 + 0.7 * (z1 - z0)]),
        }

    # Head: radii from its centre by latitude and longitude, its bottom row at the neck's top station.
    chin = 0.87 * h
    head_pts = positions[positions[:, 1] > chin]
    hc = np.array([0.0, head_pts[:, 1].mean(), head_pts[:, 2].mean()])
    lat = np.linspace(0.0, 1.0, HEAD[0])
    # Polar angle 0 at the crown; the last row aims at the neck's side at the neck's top station.
    polar = lat * (np.pi - np.arctan2(0.035 * h, hc[1] - neck_y))
    lon = -np.pi + 2 * np.pi * np.arange(HEAD[1]) / HEAD[1]
    dirs = np.stack(
        [
            np.sin(polar)[:, None] * np.cos(lon)[None],
            np.cos(polar)[:, None] * np.ones_like(lon)[None],
            np.sin(polar)[:, None] * np.sin(lon)[None],
        ],
        -1,
    ).reshape(-1, 3)
    head_faces = faces[(positions[faces][:, :, 1] > neck_y - 0.02 * h).all(1)]
    head_r = _fill_grid(_rays(positions, head_faces, hc, dirs).reshape(HEAD), 0.01 * h)

    joints |= {
        "pelvis": np.array([0.0, fr["trochanter_height"] * h, 0.0]),
        "spine": np.array([0.0, fr["waist_height"] * h, 0.0]),
        "chest": np.array([0.0, axilla, 0.0]),
        "neck": np.array([0.0, neck_y, 0.0]),
        "head": hc,
    }
    joints = {n: np.asarray(joints[n], float) for n in JOINTS}
    return SectionTable(
        float(stature), chains, hc, head_r, joints, float(crotch), float(axilla), float(acromion)
    )


def _fill_grid(r, least):
    """Radii a ray missed (below `least`) filled down each longitude, then around each row, from those it hit."""
    r = r.copy()
    rows, cols = np.arange(r.shape[0]), np.arange(r.shape[1])
    for j in cols:
        ok = r[:, j] >= least
        if ok.any() and not ok.all():
            r[~ok, j] = np.interp(rows[~ok], rows[ok], r[ok, j])
    for i in rows:
        ok = r[i] >= least
        if not ok.all():
            n = len(cols)
            xs = np.r_[cols[ok] - n, cols[ok], cols[ok] + n]
            r[i, ~ok] = np.interp(cols[~ok], xs, np.tile(r[i, ok], 3))
    return r


def _binned_chain(points, start, end, n, ref) -> Chain:
    """Rings around an axis read as the farthest point in each angle bin of nearby points (a foot is open at the top)."""
    axis = end - start
    u, v = _frame(axis, ref)
    length = np.linalg.norm(axis)
    unit = axis / length
    s = (points - start) @ unit
    ts = np.linspace(0.04, 0.96, n)
    c, r, ok = np.zeros((n, 2)), np.zeros((n, K)), np.zeros(n, bool)
    theta = -np.pi + 2 * np.pi * np.arange(K) / K
    for i, t in enumerate(ts):
        near = points[np.abs(s - t * length) < 0.6 * length / n]
        if len(near) < 6:
            continue
        rel = near - (start + axis * t)
        p2 = np.c_[rel @ u, rel @ v]
        ang = np.arctan2(p2[:, 1], p2[:, 0])
        rad = np.hypot(p2[:, 0], p2[:, 1])
        b = np.floor((ang + np.pi) / (2 * np.pi) * K).astype(int) % K
        best = np.zeros(K)
        np.maximum.at(best, b, rad)
        have = best > 0
        if have.sum() < 3:
            continue
        idx = np.arange(K)
        r[i] = np.interp(
            idx,
            np.r_[idx[have] - K, idx[have], idx[have] + K],
            np.r_[best[have], best[have], best[have]],
        )
        ok[i] = True
    return Chain(start, end, u, v, ts, _fill(c, ok), _fill(r, ok))


# ------------------------------------------------------------------------------------------- storage, levels


def save_table(table: SectionTable, path) -> None:
    arrays = {
        "stature": table.stature,
        "head_center": table.head_center,
        "head_r": table.head_r,
        "crotch_y": table.crotch_y,
        "axilla_y": table.axilla_y,
        "acromion_y": table.acromion_y,
    }
    for name, ch in table.chains.items():
        for f in ("start", "end", "u", "v", "t", "c", "r"):
            arrays[f"chain/{name}/{f}"] = getattr(ch, f)
    for name, p in table.joints.items():
        arrays[f"joint/{name}"] = p
    np.savez_compressed(path, **arrays)


def load_table(body: str | os.PathLike) -> SectionTable:
    """A packaged table ("female", "male", or a variant such as "male_heavy") or one saved at a path."""
    packaged = DATA / f"human_{body}.npz"
    path = packaged if packaged.exists() else Path(body)
    z = np.load(path)
    chains = {}
    for name in CHAINS:
        chains[name] = Chain(
            *(z[f"chain/{name}/{f}"] for f in ("start", "end", "u", "v", "t", "c", "r"))
        )
    joints = {n: z[f"joint/{n}"] for n in JOINTS}
    return SectionTable(
        float(z["stature"]),
        chains,
        z["head_center"],
        z["head_r"],
        joints,
        float(z["crotch_y"]),
        float(z["axilla_y"]),
        float(z["acromion_y"]),
    )


def _resample_angles(r, k):
    if r.shape[1] == k:
        return r.copy()
    src = np.arange(r.shape[1]) / r.shape[1]
    dst = np.arange(k) / k
    xs = np.r_[src - 1, src, src + 1]
    return np.stack([np.interp(dst, xs, np.r_[row, row, row]) for row in r])


def _resample_stations(ch: Chain, ts) -> Chain:
    c = np.stack([np.interp(ts, ch.t, ch.c[:, j]) for j in range(2)], 1)
    r = np.stack([np.interp(ts, ch.t, ch.r[:, j]) for j in range(ch.r.shape[1])], 1)
    return replace(ch, t=np.asarray(ts, float), c=c, r=r)


def resample(table: SectionTable, k: int, step: int) -> SectionTable:
    """The table at K = `k` radii per ring and about one station in `step`, keeping the ends and the shoulder rows."""
    chains = {}
    for name, ch in table.chains.items():
        ch = replace(ch, r=_resample_angles(ch.r, k))
        n = -(-CHAINS[name] // step)
        if name == "torso":
            y = lambda t: ch.start[1] + (ch.end[1] - ch.start[1]) * t  # noqa: E731
            t_ax = (table.axilla_y - ch.start[1]) / (ch.end[1] - ch.start[1])
            t_ac = (table.acromion_y - ch.start[1]) / (ch.end[1] - ch.start[1])
            assert y(t_ax) <= y(t_ac)
            ts = _torso_ts(n, t_ax, t_ac, shoulder_rows(k))
        else:
            ts = np.linspace(ch.t[0], ch.t[-1], n)
        chains[name] = _resample_stations(ch, ts)
    lat = max(4, round(HEAD[0] * k / K)) if k != K else HEAD[0]
    lat = {16: 12, 8: 6, 6: 4}.get(k, lat)
    head = _resample_angles(table.head_r, k)
    rows = np.linspace(0, HEAD[0] - 1, lat)
    head = np.stack([np.interp(rows, np.arange(HEAD[0]), head[:, j]) for j in range(k)], 1)
    return replace(table, chains=chains, head_r=head)
