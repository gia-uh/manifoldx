"""A section table lofted into one closed skin, with fixed stitches at the crotch, the shoulders and the neck.

Every index layout here depends only on the level (K radii per ring, the station counts), never on the shape,
so two tables give the same faces and any body between them is a weighted sum of their vertices.

- Rings: vertex k of a ring sits at angle -pi + 2*pi*k/K in its chain's (u, v); consecutive rings join by two
  triangles per k. For the torso u = +X (the figure's left) and v = -Z, so k = K/4 is the front centre and
  3K/4 the back centre.
- Crotch: the torso's lowest ring splits at its front and back centres into a left and a right half of K/2 + 1
  vertices each; K/2 - 1 midline vertices on the plane x = 0 run from the front centre down through the crotch
  and up to the back centre. Each half plus the midline is a loop of K vertices that joins its leg's top ring.
- Shoulder: on each side the torso rings from axilla to acromion (R rows) leave out a run of C angles centred on
  that side; the rectangle's boundary has 2C + 2R - 4 = K vertices and joins the arm's first ring.
- Neck and head: the head grid's rows below the crown join the torso's top ring; the crown is one vertex.
- Feet are their own closed tubes overlapping the ankle, the legs and arms close with a fan at their ends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifoldx.modeling.assets.human_sections import SectionTable, resample, shoulder_rows
from manifoldx.modeling.assets.vehicle import orient_outward

LEVELS = {0: (16, 1), 1: (8, 2), 2: (6, 4)}  # K, station step


def shoulder(k: int) -> tuple[int, int]:
    """(C columns, R rows) of a shoulder opening at K = `k`, so its boundary 2C + 2R - 4 has K vertices."""
    r = shoulder_rows(k)
    return k // 2 + 2 - r, r

TORSO, LEG_L, LEG_R, FOOT_L, FOOT_R, ARM_L, ARM_R, HEAD, BRIDGE = range(9)
REGION_OF = {
    "torso": TORSO,
    "leg_l": LEG_L,
    "leg_r": LEG_R,
    "foot_l": FOOT_L,
    "foot_r": FOOT_R,
    "arm_l": ARM_L,
    "arm_r": ARM_R,
}


@dataclass(frozen=True)
class Skin:
    positions: np.ndarray
    faces: np.ndarray
    region: np.ndarray
    station: np.ndarray
    ring: dict


class _Builder:
    def __init__(self):
        self.pos, self.region, self.station, self.faces = [], [], [], []

    def add(self, points, region, station):
        start = sum(len(p) for p in self.pos)
        points = np.atleast_2d(points)
        self.pos.append(points)
        self.region.append(np.full(len(points), region, np.uint8))
        self.station.append(np.broadcast_to(np.asarray(station, np.float32), (len(points),)).copy())
        return np.arange(start, start + len(points))

    def quads(self, a, b):
        """Two triangles per step between two loops of the same length (closed)."""
        a2, b2 = np.roll(a, -1), np.roll(b, -1)
        self.faces.append(np.stack([a, a2, b2], 1))
        self.faces.append(np.stack([a, b2, b], 1))

    def fan(self, centre, loop):
        self.faces.append(np.stack([np.full(len(loop), centre), loop, np.roll(loop, -1)], 1))


def head_polar(table: SectionTable, rows: int) -> np.ndarray:
    """Polar angles of the head grid's rows: 0 at the crown, the last aimed at the neck's side."""
    neck_y = table.joints["neck"][1]
    last = np.pi - np.arctan2(0.035 * table.stature, table.head_center[1] - neck_y)
    return np.linspace(0.0, 1.0, rows) * last


def _rings(chain, k):
    theta = -np.pi + 2 * np.pi * np.arange(k) / k
    cs = chain.centres()
    return cs[:, None, :] + (
        chain.r[..., None]
        * (np.cos(theta)[None, :, None] * chain.u + np.sin(theta)[None, :, None] * chain.v)
    )


def _aligned(loop_pos, ring_pos):
    """The ring's order (offset and direction) that best follows the loop, vertex by vertex."""
    k = len(ring_pos)
    best = None
    for direction in (1, -1):
        for s in range(k):
            idx = (s + direction * np.arange(k)) % k
            cost = np.linalg.norm(loop_pos - ring_pos[idx], axis=1).sum()
            if best is None or cost < best[0]:
                best = (cost, idx)
    return best[1]


_ALIGN = {}


def _alignment(k, step):
    """How each stitch's ring is turned, fixed per resolution from the female table so every body shares the faces."""
    if (k, step) not in _ALIGN:
        from manifoldx.modeling.assets.human_sections import load_table

        found = {}
        skin(load_table("female"), k=k, step=step, _found=found)
        _ALIGN[k, step] = found
    return _ALIGN[k, step]


def skin(
    table: SectionTable,
    level: int = 0,
    *,
    k: int | None = None,
    step: int | None = None,
    _found: dict | None = None,
) -> Skin:
    """Loft `table` at `level` (or at K = `k` radii per ring, one station in `step`) into a closed skin.

    Its faces depend only on the resolution, never on the table.
    """
    if k is None:
        k, step = LEVELS[level]
    if k < 6 or k % 2:
        raise ValueError(f"k: {k} must be even and at least 6")
    align = _alignment(k, step) if _found is None else None

    def aligned(key, loop_pos, ring_pos):
        if _found is not None:
            _found[key] = _aligned(loop_pos, ring_pos)
            return _found[key]
        return align[key]

    t = resample(table, k, step)
    b = _Builder()
    ring = {}
    pos_of = {}

    # Torso, with the shoulder openings.
    torso = t.chains["torso"]
    tr = _rings(torso, k)
    cs = torso.centres()
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(cs, axis=0), axis=1))]
    y = cs[:, 1]
    rows = np.flatnonzero((y >= t.axilla_y - 1e-6) & (y <= t.acromion_y + 1e-6))
    c_cols, r_rows = shoulder(k)
    assert len(rows) == r_rows, (len(rows), r_rows)
    r0 = rows[0]
    first = k // 2 - (c_cols - 1) // 2
    left_cols = np.arange(first, first + c_cols) % k
    right_cols = (left_cols + k // 2) % k
    drop = np.zeros((len(tr), k), bool)
    for cols in (left_cols, right_cols):
        for i in range(r0 + 1, r0 + r_rows - 1):
            drop[i, cols[1:-1]] = True
    idx = np.full((len(tr), k), -1)
    for i in range(len(tr)):
        keep = ~drop[i]
        idx[i, keep] = b.add(tr[i, keep], TORSO, arc[i])
    ring["torso"] = idx
    open_quad = np.zeros((len(tr) - 1, k), bool)  # quad (i, j) spans rings i, i+1 and angles j, j+1
    for cols in (left_cols, right_cols):
        open_quad[r0 : r0 + r_rows - 1, cols[:-1]] = True
    for i in range(len(tr) - 1):
        for j in range(k):
            if open_quad[i, j]:
                continue
            a, a2, c, c2 = idx[i, j], idx[i, (j + 1) % k], idx[i + 1, j], idx[i + 1, (j + 1) % k]
            b.faces.append(np.array([[a, a2, c2], [a, c2, c]]))

    def boundary(cols):
        bottom = [idx[r0, c] for c in cols]
        right = [idx[r, cols[-1]] for r in range(r0 + 1, r0 + r_rows - 1)]
        top = [idx[r0 + r_rows - 1, c] for c in cols[::-1]]
        left = [idx[r, cols[0]] for r in range(r0 + r_rows - 2, r0, -1)]
        return np.array(bottom + right + top + left)

    allpos = lambda: np.concatenate(b.pos)  # noqa: E731

    # Limbs: legs, arms; each closes with a fan at its far end.
    for name in ("leg_l", "leg_r", "arm_l", "arm_r"):
        ch = t.chains[name]
        rr = _rings(ch, k)
        ca = ch.centres()
        arc_l = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ca, axis=0), axis=1))]
        li = np.stack([b.add(rr[i], REGION_OF[name], arc_l[i]) for i in range(len(rr))])
        for i in range(len(rr) - 1):
            b.quads(li[i], li[i + 1])
        tip = b.add(ca[-1] + (ca[-1] - ca[-2]) * 0.5, REGION_OF[name], arc_l[-1])[0]
        b.fan(tip, li[-1])
        ring[name] = li

    # Shoulders: each opening's boundary joins its arm's first ring.
    for name, cols in (("arm_l", left_cols), ("arm_r", right_cols)):
        loop = boundary(cols)
        p = allpos()
        arm0 = ring[name][0]
        b.quads(loop, arm0[aligned(name, p[loop], p[arm0])])

    # Crotch: two loops of K vertices around a shared midline join the legs' top rings.
    bottom = idx[0]
    front, back = k // 4, 3 * k // 4
    p = allpos()
    fp, bp = p[bottom[front]], p[bottom[back]]
    zc = (fp[2] + bp[2]) / 2
    n_mid = k // 2 - 1
    s = np.arange(1, n_mid + 1) / (k // 2)
    mid = np.stack(
        [
            np.zeros(n_mid),
            (1 - s) * fp[1] + s * bp[1] - ((fp[1] + bp[1]) / 2 - t.crotch_y) * np.sin(np.pi * s),
            (1 - s) * fp[2] + s * bp[2] + (zc - (fp[2] + bp[2]) / 2),
        ],
        1,
    )
    mid_idx = b.add(mid, BRIDGE, 0.0)  # front to back
    left_half = bottom[
        np.arange(front, back + 1)
    ]  # front centre through the left side to the back centre
    right_half = bottom[
        np.r_[np.arange(back, k), np.arange(0, front + 1)]
    ]  # back through the right to the front
    loops = {"leg_l": np.r_[left_half, mid_idx[::-1]], "leg_r": np.r_[right_half, mid_idx]}
    p = allpos()
    for name, loop in loops.items():
        top = ring[name][0]
        b.quads(loop, top[aligned(name, p[loop], p[top])])

    # Head: rows below the crown, joined to the torso's top ring; the crown is one vertex.
    rows_h, _ = t.head_r.shape
    polar = head_polar(t, rows_h)
    lon = -np.pi + 2 * np.pi * np.arange(k) / k
    dirs = np.stack(
        [
            np.sin(polar)[:, None] * np.cos(lon)[None],
            np.cos(polar)[:, None] * np.ones((1, k)),
            np.sin(polar)[:, None] * np.sin(lon)[None],
        ],
        -1,
    )
    grid = t.head_center + dirs * t.head_r[..., None]
    top_arc = arc[-1]
    crown = b.add(t.head_center + [0, t.head_r[0].mean(), 0], HEAD, top_arc + t.head_r[0].mean())[0]
    head_rows = [
        b.add(grid[i], HEAD, top_arc + (grid[i, :, 1].mean() - y[-1])) for i in range(1, rows_h - 1)
    ]
    b.fan(crown, head_rows[0])
    for i in range(len(head_rows) - 1):
        b.quads(head_rows[i], head_rows[i + 1])
    p = allpos()
    neck = idx[-1]
    b.quads(head_rows[-1], neck[aligned("neck", p[head_rows[-1]], p[neck])])

    # Feet: closed tubes of their own, sole on the ground.
    for name in ("foot_l", "foot_r"):
        ch = t.chains[name]
        rr = _rings(ch, k)
        rr[..., 1] = np.maximum(rr[..., 1], 0.0)
        ca = ch.centres()
        arc_f = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ca, axis=0), axis=1))]
        fi = np.stack([b.add(rr[i], REGION_OF[name], arc_f[i]) for i in range(len(rr))])
        for i in range(len(rr) - 1):
            b.quads(fi[i], fi[i + 1])
        heel = b.add(
            np.maximum(ca[0] - (ca[1] - ca[0]) * 0.5, [-np.inf, 0, -np.inf]), REGION_OF[name], 0.0
        )[0]
        toe = b.add(ca[-1] + (ca[-1] - ca[-2]) * 0.5, REGION_OF[name], arc_f[-1])[0]
        b.fan(heel, fi[0][::-1])
        b.fan(toe, fi[-1])
        ring[name] = fi

    positions = np.concatenate(b.pos).astype(np.float32)
    faces = orient_outward(positions, np.concatenate(b.faces).astype(np.int64)).astype(np.uint32)
    return Skin(positions, faces, np.concatenate(b.region), np.concatenate(b.station), ring)
