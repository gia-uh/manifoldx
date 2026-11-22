"""Clothes, hair and a painted face on a figure.

Near (level 0) a garment is the body's own vertices over its span, pushed out along their normals by its
looseness and painted: the skin it covers is gone and the mesh stays closed with the same faces. A skirt, a bun
and a hat's brim are closed shells of their own, weighted to the joints they follow. Farther away (levels 1 and 2)
garments are only painted on the skin; hair and hats keep their shape at level 1 and are painted at level 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifoldx.modeling.assets.human_sections import JOINTS
from manifoldx.modeling.assets.human_skin import (
    ARM_L,
    ARM_R,
    BRIDGE,
    FOOT_L,
    FOOT_R,
    HEAD,
    LEG_L,
    LEG_R,
    TORSO,
    Skin,
)
from manifoldx.modeling.assets.vehicle import ParamsRanges, choice, orient_outward, rng
from manifoldx.modeling.mesh import Mesh

SHELL = 255  # the region of a garment shell's vertices
HEM = 0.002  # metres a garment stands off the skin beyond its looseness
SKIRT_RINGS = {0: 8, 1: 5, 2: 3}
SHELL_LAT = {0: 6, 1: 4, 2: 3}


@dataclass
class Outfit(ParamsRanges):
    """What a figure wears. Colours are sRGB hex; `stripe` is (from, to, colour), a band by torso height."""

    top: str | None = choice("tshirt", None, "tshirt", "shirt", "blouse", "tank")
    sleeves: float = rng(0.25, 0.0, 1.0, step=0.05, unit="")  # fraction of the arm, shoulder to wrist
    bottom: str = choice("trousers", "trousers", "shorts", "skirt")
    bottom_length: float = rng(1.0, 0.2, 1.0, step=0.05, unit="")  # fraction of the leg, hip to ankle
    looseness: float = rng(0.012, 0.004, 0.05, step=0.002, unit="m")
    shoes: str | None = choice("shoes", None, "sandals", "shoes", "boots")
    hair: str | None = choice("short", None, "short", "long", "bun", "curly")
    hat: str | None = choice(None, None, "cap", "brim")
    top_color: str = "#e8e2d0"
    bottom_color: str = "#3b4a5c"
    shoe_color: str = "#2b2522"
    hair_color: str = "#2a1d14"
    hat_color: str = "#d9c48a"
    stripe: tuple | None = None


def _rgb(hex_colour: str) -> np.ndarray:
    return np.array([int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)], np.float32) / 255


class _Dressing:
    """The body mesh being dressed, plus the shells added to it."""

    def __init__(self, mesh: Mesh, sk: Skin):
        self.pos = mesh.positions.astype(float).copy()
        self.normals = mesh.recompute_normals().normals.astype(float)
        self.faces = [mesh.faces.astype(np.int64)]
        self.colors = mesh.colors.astype(np.float32).copy()
        self.pairs = mesh.joints.astype(np.int64).copy()
        self.weights = mesh.weights.astype(np.float32).copy()
        self.region = sk.region
        self.station = sk.station
        self.ring = sk.ring
        self.parts: dict[str, list] = {}

    def push(self, verts, by):
        self.pos[verts] += self.normals[verts] * np.asarray(by, float).reshape(-1, 1)

    def paint(self, verts, rgb):
        self.colors[verts] = rgb

    def mark(self, name, verts):
        self.parts.setdefault(name, []).append(np.asarray(verts, np.int64))

    def shell(self, name, positions, faces, pairs, weights, rgb):
        """A closed shell of its own, appended and turned outward."""
        n = len(self.pos)
        faces = orient_outward(positions, np.asarray(faces, np.int64))
        self.pos = np.concatenate([self.pos, positions])
        self.faces.append(faces + n)
        self.colors = np.concatenate([self.colors, np.broadcast_to(rgb, (len(positions), 3))]).astype(np.float32)
        self.pairs = np.concatenate([self.pairs, pairs])
        self.weights = np.concatenate([self.weights, weights]).astype(np.float32)
        self.region = np.concatenate([self.region, np.full(len(positions), SHELL, np.uint8)])
        self.station = np.concatenate([self.station, np.zeros(len(positions), np.float32)])
        self.mark(name, np.arange(n, n + len(positions)))

    def mesh(self) -> tuple[Mesh, dict]:
        mesh = Mesh(
            positions=self.pos.astype(np.float32),
            faces=np.concatenate(self.faces).astype(np.uint32),
            colors=np.clip(self.colors, 0, 1).astype(np.float32),
            joints=self.pairs.astype(np.uint32),
            weights=np.clip(self.weights, 0, 1).astype(np.float32),
        )
        return mesh, {k: np.unique(np.concatenate(v)) for k, v in self.parts.items()}


def _ring_verts(rings, i0, i1):
    """Vertex indices of rings i0..i1 (inclusive), leaving out the shoulders' gaps."""
    if i1 < i0:
        return np.zeros(0, np.int64)
    v = np.asarray(rings[i0 : i1 + 1]).reshape(-1)
    return v[v >= 0]


def _nearest_ring(d: _Dressing, chain, point):
    rings = d.ring[chain]
    centres = np.stack([d.pos[np.asarray(r)[np.asarray(r) >= 0]].mean(0) for r in rings])
    return int(np.argmin(np.linalg.norm(centres - point, axis=1)))


def _limb(d: _Dressing, region, chain, upto_ring):
    """A limb's vertices from its root to ring `upto_ring` (by station, so its tip goes with its last ring)."""
    if upto_ring < 0:
        return np.zeros(0, np.int64)
    last = d.station[np.asarray(d.ring[chain][upto_ring])].max()
    return np.flatnonzero((d.region == region) & (d.station <= last + 1e-6))


def _annulus(centre, r_in, r_out, thickness, k):
    """A flat closed ring (four loops of k vertices) around +Y at `centre`, radii by angle."""
    t = -np.pi + 2 * np.pi * np.arange(k) / k
    d = np.stack([np.cos(t), np.zeros(k), np.sin(t)], 1)
    up = np.array([0, thickness / 2, 0])
    loops = [centre + d * r_out[:, None] + up, centre + d * r_out[:, None] - up,
             centre + d * r_in[:, None] - up, centre + d * r_in[:, None] + up]
    pos = np.concatenate(loops)
    faces = []
    for a in range(4):
        b = (a + 1) % 4
        for j in range(k):
            j2 = (j + 1) % k
            p, q, r, s = a * k + j, a * k + j2, b * k + j2, b * k + j
            faces += [[p, q, r], [p, r, s]]
    return pos, np.array(faces)


def _sphere(centre, radius, k, lat):
    polar = np.linspace(0, np.pi, lat + 2)[1:-1]
    t = -np.pi + 2 * np.pi * np.arange(k) / k
    rows = [centre + radius * np.stack([np.sin(p) * np.cos(t), np.full(k, np.cos(p)), np.sin(p) * np.sin(t)], 1)
            for p in polar]
    pos = np.concatenate(rows + [centre[None] + [0, radius, 0], centre[None] - [0, radius, 0]])
    top, bottom = lat * k, lat * k + 1
    faces = []
    for i in range(lat - 1):
        for j in range(k):
            j2 = (j + 1) % k
            a, b, c, e = i * k + j, i * k + j2, (i + 1) * k + j2, (i + 1) * k + j
            faces += [[a, b, c], [a, c, e]]
    for j in range(k):
        j2 = (j + 1) % k
        faces += [[top, j2, j], [bottom, (lat - 1) * k + j, (lat - 1) * k + j2]]
    return pos, np.array(faces)


def dress(mesh: Mesh, sk: Skin, joints: dict, outfit: Outfit, level: int, tone) -> tuple[Mesh, dict]:
    """`mesh` (a figure's skin with colours, joints and weights) wearing `outfit` at `level`; the garments' vertices."""
    outfit.validate()
    d = _Dressing(mesh, sk)
    idx = {n: i for i, n in enumerate(JOINTS)}
    near = level == 0
    k = np.asarray(sk.ring["torso"]).shape[1]
    tone = np.asarray(tone, np.float32)
    loose = outfit.looseness * (1.5 if outfit.top == "blouse" else 1.0)

    torso = sk.ring["torso"]
    ring_y = np.array([d.pos[np.asarray(r)[np.asarray(r) >= 0], 1].mean() for r in torso])
    waist = int(np.argmin(np.abs(ring_y - joints["spine"][1])))
    shoulder_rows = np.flatnonzero((np.asarray(torso) < 0).any(1))
    acromion = int(shoulder_rows.max()) + 1 if len(shoulder_rows) else len(torso) - 2

    # Top: the torso from the waist to the base of the neck, and the sleeves.
    if outfit.top is not None:
        verts = [_ring_verts(torso, waist, min(acromion + 1, len(torso) - 2))]
        sleeves = 0.0 if outfit.top == "tank" else outfit.sleeves
        for region, chain, side in ((ARM_L, "arm_l", "l"), (ARM_R, "arm_r", "r")):
            wrist = _nearest_ring(d, chain, joints[f"wrist_{side}"])
            verts.append(_limb(d, region, chain, round(sleeves * wrist) if sleeves > 0 else -1))
        top = np.unique(np.concatenate(verts))
        d.paint(top, _rgb(outfit.top_color))
        if outfit.stripe is not None:
            lo, hi, colour = outfit.stripe
            y0, y1 = ring_y[0], ring_y[-1]
            band = top[(d.pos[top, 1] >= y0 + lo * (y1 - y0)) & (d.pos[top, 1] <= y0 + hi * (y1 - y0))]
            d.paint(band, _rgb(colour))
        if near:
            d.push(top, loose + HEM)
            d.mark("top", top)

    # Bottom: below the waist and down the legs; a skirt is a tube of its own around both legs.
    length = min(outfit.bottom_length, 0.35) if outfit.bottom == "shorts" else outfit.bottom_length
    hip_y = joints["hip_l"][1]
    ankle_y = joints["ankle_l"][1]
    hem_y = hip_y - length * (hip_y - ankle_y)
    pelvis = np.concatenate([_ring_verts(torso, 0, waist - 1), np.flatnonzero(d.region == BRIDGE)])
    legs = []
    for region, chain, side in ((LEG_L, "leg_l", "l"), (LEG_R, "leg_r", "r")):
        ring_ys = np.array([d.pos[np.asarray(r), 1].mean() for r in sk.ring[chain]])
        upto = int(np.flatnonzero(ring_ys >= hem_y).max()) if (ring_ys >= hem_y).any() else -1
        legs.append(_limb(d, region, chain, upto))
    bottom_rgb = _rgb(outfit.bottom_color)
    if outfit.bottom == "skirt":
        d.paint(pelvis, bottom_rgb)
        if near:
            d.push(pelvis, outfit.looseness + HEM)
            d.mark("waist", pelvis)  # the band above a skirt; "bottom" is the tube itself
            _skirt(d, sk, joints, idx, outfit, hem_y, k, level, bottom_rgb)
        else:
            covered = np.concatenate([pelvis, *legs])
            d.paint(covered, bottom_rgb)
    else:
        covered = np.concatenate([pelvis, *legs])
        d.paint(covered, bottom_rgb)
        if near:
            d.push(covered, outfit.looseness + HEM)
            d.mark("bottom", covered)

    # Shoes: the feet, a darker sole; boots climb the shin; sandals only paint the sole.
    if outfit.shoes is not None:
        feet = np.flatnonzero((d.region == FOOT_L) | (d.region == FOOT_R))
        sole = feet[d.pos[feet, 1] < 0.015]
        shoe_rgb = _rgb(outfit.shoe_color)
        if outfit.shoes == "sandals":
            d.paint(sole, shoe_rgb)
        else:
            shoe = feet
            if outfit.shoes == "boots":
                for region, side in ((LEG_L, "l"), (LEG_R, "r")):
                    top_y = joints[f"ankle_{side}"][1] + 0.12
                    shoe = np.concatenate([shoe, np.flatnonzero((d.region == region) & (d.pos[:, 1] < top_y))])
            d.paint(shoe, shoe_rgb)
            d.paint(sole, shoe_rgb * 0.6)
            if near:
                d.push(shoe, 0.004)
                d.pos[shoe, 1] = np.maximum(d.pos[shoe, 1], 0.0)
                d.mark("shoes", shoe)

    _head(d, joints, idx, outfit, level, k, tone)
    return d.mesh()


def _skirt(d, sk, joints, idx, outfit, hem_y, k, level, rgb):
    """One closed tube from the crotch to the hem around both legs, widening downward."""
    top_y = np.array([d.pos[np.asarray(r)[np.asarray(r) >= 0], 1].mean() for r in sk.ring["torso"][:1]])[0]
    n = SKIRT_RINGS[level]
    ys = np.linspace(top_y, hem_y, n)
    leg_pts = d.pos[(d.region == LEG_L) | (d.region == LEG_R) | (d.region == BRIDGE)]
    torso0 = d.pos[np.asarray(sk.ring["torso"][0])]
    zc = torso0[:, 2].mean()
    t = -np.pi + 2 * np.pi * np.arange(k) / k
    dirs = np.stack([np.cos(t), np.zeros(k), np.sin(t)], 1)
    outer = []
    for i, y in enumerate(ys):
        band = torso0 if i == 0 else leg_pts[np.abs(leg_pts[:, 1] - y) < 0.03]
        if len(band) < 3:
            band = torso0
        c = np.array([0.0, y, zc])
        support = ((band - c) * [1, 0, 1]) @ dirs.T  # how far the body reaches toward each direction
        r = support.max(0) + outfit.looseness + HEM + 0.25 * outfit.looseness * (top_y - y) / 0.1
        outer.append(c + dirs * np.maximum(r, 0.02)[:, None])
    outer = np.stack(outer)
    inner = outer - dirs[None] * 0.004
    loops = [outer[i] for i in range(n)] + [inner[i] for i in range(n - 1, -1, -1)]
    pos = np.concatenate(loops)
    faces = []
    m = len(loops)
    for a in range(m):
        b = (a + 1) % m
        for j in range(k):
            j2 = (j + 1) % k
            p, q, r, s = a * k + j, a * k + j2, b * k + j2, b * k + j
            faces += [[p, q, r], [p, r, s]]
    side = np.where(pos[:, 0] >= 0, "l", "r")
    hip = np.array([idx[f"hip_{s}"] for s in side])
    knee = np.array([idx[f"knee_{s}"] for s in side])
    hy, ky = joints["hip_l"][1], joints["knee_l"][1]
    w = np.clip((pos[:, 1] - ky) / max(hy - ky, 1e-6), 0, 1)
    d.shell("bottom", pos, faces, np.stack([hip, knee], 1), w, rgb)


def _head(d, joints, idx, outfit, level, k, tone):
    head = np.flatnonzero(d.region == HEAD)
    centre = joints["head"]
    rel = d.pos[head] - centre
    polar = np.degrees(np.arccos(np.clip(rel[:, 1] / np.linalg.norm(rel, axis=1), -1, 1)))
    lon = np.degrees(np.arctan2(rel[:, 0], rel[:, 2]))  # 0 straight ahead
    shaped = level <= 1

    # Face at level 0: eyes, brows and a mouth in a darker tone, painted by angle from the head's centre.
    if level == 0:
        for lo, hi, lons, shade in ((80, 98, (-22.5, 22.5), 0.45), (68, 80, (-22.5, 22.5), 0.55),
                                    (108, 124, (0.0,), 0.62)):
            for c in lons:
                spot = head[(polar >= lo) & (polar < hi) & (np.abs(lon - c) < 14)]
                d.paint(spot, tone * shade)

    if outfit.hair is not None:
        style = outfit.hair
        reach = {"short": 70, "bun": 70, "curly": 80, "long": 150}[style]
        cover = (polar < reach) & ((polar < 70) | (np.abs(lon) > 60) if style == "long" else True)
        hair = head[cover]
        thick = {"short": 0.01, "bun": 0.01, "curly": 0.015, "long": 0.02}[style]
        by = np.full(len(hair), thick)
        if style == "curly":
            p = d.pos[hair]
            by += 0.01 * np.sin(p[:, 0] * 97 + p[:, 1] * 131 + p[:, 2] * 173)
        rgb = _rgb(outfit.hair_color)
        d.paint(hair, rgb)
        if shaped:
            d.push(hair, by)
            d.mark("hair", hair)
            if style == "bun":
                back = d.pos[head][:, 2].min()
                pos, faces = _sphere(centre + [0, 0.04, back - centre[2] - 0.02], 0.045, k, SHELL_LAT[level])
                n = len(pos)
                d.shell("hair", pos, faces, np.full((n, 2), idx["head"]), np.ones(n), rgb)

    if outfit.hat is not None:
        crown = head[polar < 62]
        rgb = _rgb(outfit.hat_color)
        d.paint(crown, rgb)
        if shaped:
            d.push(crown, 0.016)
            d.mark("hat", crown)
            ring = head[(polar >= 55) & (polar < 75)]
            y = d.pos[ring, 1].mean()
            t = np.degrees(-np.pi + 2 * np.pi * np.arange(k) / k)
            rel_r = d.pos[ring] - [centre[0], 0, centre[2]]
            ang = np.degrees(np.arctan2(rel_r[:, 2], rel_r[:, 0]))
            rad = np.hypot(rel_r[:, 0], rel_r[:, 2])
            r_in = np.array([rad[np.argmin(np.abs((ang - a + 180) % 360 - 180))] for a in t]) + 0.012
            width = 0.07 if outfit.hat == "brim" else 0.015
            pos, faces = _annulus(np.array([centre[0], y, centre[2]]), r_in, r_in + width, 0.006, k)
            n = len(pos)
            d.shell("hat", pos, faces, np.full((n, 2), idx["head"]), np.ones(n), rgb)
