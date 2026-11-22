"""A parametric person: two base bodies and their variants blended into one section table, lofted into a skin.

The female and male tables are the ends of ``build``. Each has three variant tables read from the same body edited
to look heavy, muscular and old; ``weight``, ``muscle`` and ``age`` are blend weights of those variants, added to
the base as differences field by field (blend shapes), so a belly grows forward and an old back bends where the
drawings show it rather than where a girth curve guesses. Every table shares one ring layout, so any blend lofts
into the same faces. Stature, leg length, shoulder and hip breadth then reshape the result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from manifoldx.modeling.assets.human_sections import (
    CHAINS,
    JOINTS,
    K,
    SectionTable,
    load_table,
    shoulder_rows,
)
from manifoldx.modeling.assets.human_skin import (
    ARM_L,
    ARM_R,
    BRIDGE,
    FOOT_L,
    FOOT_R,
    HEAD,
    LEG_L,
    LEG_R,
    LEVELS,
    TORSO,
    skin,
)
from manifoldx.modeling.assets.vehicle import ParamsRanges, choice, rng
from manifoldx.modeling.mesh import Mesh

SKIN_TONES = {  # sRGB
    "light": (0.93, 0.78, 0.67),
    "medium": (0.80, 0.60, 0.46),
    "tan": (0.69, 0.50, 0.36),
    "brown": (0.52, 0.35, 0.23),
    "dark": (0.33, 0.21, 0.14),
}
PARENTS = {
    "pelvis": None,
    "spine": "pelvis",
    "chest": "spine",
    "neck": "chest",
    "head": "neck",
    **{
        n: p
        for s in "lr"
        for n, p in (
            (f"shoulder_{s}", "chest"),
            (f"elbow_{s}", f"shoulder_{s}"),
            (f"wrist_{s}", f"elbow_{s}"),
            (f"hip_{s}", "pelvis"),
            (f"knee_{s}", f"hip_{s}"),
            (f"ankle_{s}", f"knee_{s}"),
            (f"toe_{s}", f"ankle_{s}"),
        )
    },
}
VARIANTS = ("heavy", "muscular", "elder")
SMOOTH = {0: 6, 1: 3, 2: 0}
STITCH_SMOOTH = 0.6  # how far the crotch stitch follows the smoothing  # Taubin iterations of the lofted skin by level: rings and stitches lose their facets
BLEND = {  # metres over which a joint takes over from its parent, on each side of it
    "shoulder": 0.16,
    "hip": 0.08,
    "elbow": 0.05,
    "knee": 0.09,
    "wrist": 0.03,
    "ankle": 0.03,
    "neck": 0.04,
    "spine": 0.06,
    "chest": 0.06,
    "toe": 0.02,
    "head": 0.04,
}
ROOT = {"shoulder": 0.12, "hip": 0.0}  # a limb's root joint blends this far past the limb's first ring


@dataclass
class HumanParams(ParamsRanges):
    """A person's body. ``build`` runs from the female table (0) to the male one (1)."""

    stature: float = rng(1.70, 1.45, 1.95, step=0.01, unit="m")
    build: float = rng(0.5, 0.0, 1.0, step=0.05, unit="")
    weight: float = rng(1.0, 0.8, 1.5, step=0.05, unit="x")  # 1.5 is the whole heavy variant
    muscle: float = rng(1.0, 0.8, 1.3, step=0.05, unit="x")  # 1.3 is the whole muscular variant
    shoulders: float = rng(1.0, 0.85, 1.2, step=0.01, unit="x")
    hips: float = rng(1.0, 0.85, 1.25, step=0.01, unit="x")
    leg_ratio: float = rng(1.0, 0.94, 1.06, step=0.01, unit="x")
    age: str = choice("adult", "adult", "elder")
    skin: str = choice("medium", *SKIN_TONES)
    detail: float = rng(1.0, 0.25, 2.0, step=0.05, unit="x")

    @classmethod
    def sample(cls, seed: int) -> HumanParams:
        """A plausible person: stature around 1.63 m or 1.75 m by build, one in four old, the rest mid-range."""
        g = np.random.default_rng(seed)
        spec = cls.ranges()

        def middle(name):
            lo, hi = spec[name]["min"], spec[name]["max"]
            return round(float(g.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))), 3)

        build = middle("build")
        stature = float(np.clip(g.normal(1.63 + 0.12 * build, 0.07), 1.45, 1.95))
        return cls(
            stature=round(stature, 3),
            build=build,
            **{n: middle(n) for n in ("weight", "muscle", "shoulders", "hips", "leg_ratio")},
            age="elder" if g.random() < 0.25 else "adult",
            skin=str(g.choice(spec["skin"]["choices"])),
        )


@dataclass(frozen=True)
class Figure:
    params: HumanParams
    level: int
    mesh: Mesh  # positions, faces, colours
    joints: dict  # rest positions, the 19 names in JOINTS order
    parents: dict
    rings: dict  # each chain's ring vertex indices into mesh.positions (-1 where the shoulders open)
    parts: dict  # each garment's vertex indices (top, bottom, waist, shoes, hair, hat), where it has shape


# ---------------------------------------------------------------------------------------------- the table


@lru_cache(maxsize=None)
def _table(name: str) -> SectionTable:
    return load_table(name)


def _combine(tables, weights) -> SectionTable:
    """The table sum(w * table), field by field; chain frames are made orthonormal again."""

    def lin(get):
        return sum(w * np.asarray(get(t), float) for t, w in zip(tables, weights))

    chains = {}
    for name in CHAINS:
        fs = {f: lin(lambda t, f=f: getattr(t.chains[name], f)) for f in ("start", "end", "t", "c", "r")}
        axis = fs["end"] - fs["start"]
        axis /= np.linalg.norm(axis)
        u = lin(lambda t: t.chains[name].u)
        u -= axis * (u @ axis)
        u /= np.linalg.norm(u)
        chains[name] = replace(tables[0].chains[name], u=u, v=np.cross(axis, u), **fs)
    return SectionTable(
        stature=float(lin(lambda t: t.stature)),
        chains=chains,
        head_center=lin(lambda t: t.head_center),
        head_r=lin(lambda t: t.head_r),
        joints={n: lin(lambda t, n=n: t.joints[n]) for n in JOINTS},
        crotch_y=float(lin(lambda t: t.crotch_y)),
        axilla_y=float(lin(lambda t: t.axilla_y)),
        acromion_y=float(lin(lambda t: t.acromion_y)),
    )


def _shoulder_band(table: SectionTable) -> SectionTable:
    """Axilla and acromion heights put back on the torso's shoulder rows, which a blend moves apart slightly."""
    torso = table.chains["torso"]
    rows = shoulder_rows(K)
    below = round((CHAINS["torso"] - rows) * 0.75)
    y = torso.start[1] + (torso.end[1] - torso.start[1]) * torso.t
    return replace(table, axilla_y=float(y[below]), acromion_y=float(y[below + rows - 1]))


def _blended(p: HumanParams) -> SectionTable:
    weights = {
        "heavy": max(0.0, (p.weight - 1) / 0.5),
        "muscular": max(0.0, (p.muscle - 1) / 0.3),
        "elder": 1.0 if p.age == "elder" else 0.0,
    }
    ends = [(1 - p.build, "female"), (p.build, "male")]
    tables, coeffs = [], []
    for w_body, body in ends:
        if w_body == 0:
            continue
        tables.append(_table(body))
        coeffs.append(w_body * (1 - sum(weights.values())))
        for v in VARIANTS:
            if weights[v]:
                tables.append(_table(f"{body}_{v}"))
                coeffs.append(w_body * weights[v])
    return _shoulder_band(_combine(tables, coeffs))


def _thinner(t: SectionTable, p: HumanParams) -> SectionTable:
    """Weight and muscle below 1 have no variant: they narrow the torso and limbs, and the limbs alone."""
    body = 1 + min(0.0, p.weight - 1) * 0.8
    limbs = body * (1 + min(0.0, p.muscle - 1) * 0.6)
    scale = {"torso": body, "leg_l": limbs, "leg_r": limbs, "arm_l": limbs, "arm_r": limbs}
    chains = {n: replace(ch, r=ch.r * scale.get(n, 1.0)) for n, ch in t.chains.items()}
    return replace(t, chains=chains)


def _scaled(t: SectionTable, s: float) -> SectionTable:
    chains = {
        n: replace(ch, start=ch.start * s, end=ch.end * s, c=ch.c * s, r=ch.r * s)
        for n, ch in t.chains.items()
    }
    return SectionTable(
        t.stature * s,
        chains,
        t.head_center * s,
        t.head_r * s,
        {n: j * s for n, j in t.joints.items()},
        t.crotch_y * s,
        t.axilla_y * s,
        t.acromion_y * s,
    )


def _legs(t: SectionTable, ratio: float) -> SectionTable:
    """The crotch moved to `ratio` times its height: legs stretch below it, the rest moves up with it."""
    lift = t.crotch_y * (ratio - 1)

    def y_map(q):
        q = np.array(q, float)
        q[1] = q[1] * ratio if q[1] < t.crotch_y else q[1] + lift
        return q

    def shift(q):
        return np.asarray(q, float) + [0, lift, 0]

    chains = {}
    for n, ch in t.chains.items():
        if n.startswith("foot"):
            chains[n] = ch
        elif n.startswith("arm"):
            chains[n] = replace(ch, start=shift(ch.start), end=shift(ch.end))
        else:
            chains[n] = replace(ch, start=y_map(ch.start), end=y_map(ch.end))
    joints = {
        n: (shift(j) if n[:-2] in ("shoulder", "elbow", "wrist") else j if n.startswith("toe") else y_map(j))
        for n, j in t.joints.items()
    }
    return replace(
        t,
        chains=chains,
        head_center=shift(t.head_center),
        joints=joints,
        crotch_y=t.crotch_y * ratio,
        axilla_y=t.axilla_y + lift,
        acromion_y=t.acromion_y + lift,
    )


def _spine_on_torso(t: SectionTable) -> SectionTable:
    """Spine, chest and neck joints moved onto the torso's ring centres, so a bent back bends its skeleton."""
    torso = t.chains["torso"]
    cs = torso.centres()
    joints = dict(t.joints)
    for n in ("spine", "chest", "neck"):
        j = joints[n]
        joints[n] = np.array([0.0, j[1], np.interp(j[1], cs[:, 1], cs[:, 2])])
    return replace(t, joints=joints)


# ----------------------------------------------------------------------------------------------- the weights


def _blend_along(coord, before, stops):
    """Joint pairs and weights along one coordinate: `before` drives up to the first stop, each stop after it.

    `stops` is [(joint index, position, blend length)] in increasing position. Within a stop's blend length the
    vertex takes the stop's joint with weight 0.5 + (coord - position) / (2 * length), its parent the rest.
    """
    n = len(coord)
    drivers = [before] + [j for j, _, _ in stops]
    seg = np.searchsorted([p for _, p, _ in stops], coord, side="right")
    a = np.array(drivers)[seg]
    pairs = np.stack([a, a], 1)
    w = np.ones(n)
    best = np.full(n, np.inf)
    for k, (j, p, length) in enumerate(stops):
        d = coord - p
        near = (np.abs(d) < length) & (np.abs(d) < best)
        best[near] = np.abs(d[near])
        pairs[near] = [j, drivers[k]]
        w[near] = 0.5 + d[near] / (2 * length)
    return pairs, w


def _arc_of(point, centres, stations):
    """The station of the point on the polyline of ring centres nearest `point` (before its start: negative)."""
    best = (np.inf, 0.0)
    for i in range(len(centres) - 1):
        a, b = centres[i], centres[i + 1]
        seg = b - a
        t = (point - a) @ seg / max(seg @ seg, 1e-12)
        if i == 0:
            t = min(t, 1.0)
        elif i == len(centres) - 2:
            t = max(t, 0.0)
        else:
            t = np.clip(t, 0.0, 1.0)
        dist = np.linalg.norm(a + t * seg - point)
        if dist < best[0]:
            best = (dist, stations[i] + t * (stations[i + 1] - stations[i]))
    return best[1]


def _weights(sk, joints) -> tuple[np.ndarray, np.ndarray]:
    """Two joints per vertex of `sk` and the first one's weight."""
    idx = {n: i for i, n in enumerate(JOINTS)}
    n = len(sk.positions)
    pairs = np.full((n, 2), idx["pelvis"])
    w = np.ones(n)
    y = sk.positions[:, 1].astype(float)

    def stop(name):
        return idx[name], BLEND[name.rsplit("_", 1)[0] if name[-2:] in ("_l", "_r") else name]

    # Torso and head by height: pelvis, then spine, chest, neck, and the head from the neck's top ring up.
    top = sk.positions[np.asarray(sk.ring["torso"][-1]), 1].mean()
    body = (sk.region == TORSO) | (sk.region == HEAD)
    stops = [(*stop(j)[:1], joints[j][1], stop(j)[1]) for j in ("spine", "chest", "neck")]
    stops.append((idx["head"], top, BLEND["head"]))
    pairs[body], w[body] = _blend_along(y[body], idx["pelvis"], stops)

    # Limbs along their stations; each starts from its root joint's parent.
    limbs = {
        ARM_L: ("arm_l", "chest", ("shoulder_l", "elbow_l", "wrist_l")),
        ARM_R: ("arm_r", "chest", ("shoulder_r", "elbow_r", "wrist_r")),
        LEG_L: ("leg_l", "pelvis", ("hip_l", "knee_l", "ankle_l")),
        LEG_R: ("leg_r", "pelvis", ("hip_r", "knee_r", "ankle_r")),
        FOOT_L: ("foot_l", "ankle_l", ("toe_l",)),
        FOOT_R: ("foot_r", "ankle_r", ("toe_r",)),
    }
    for region, (chain, before, names) in limbs.items():
        mask = sk.region == region
        rings = [np.asarray(r) for r in sk.ring[chain]]
        centres = np.stack([sk.positions[r].mean(0) for r in rings]).astype(float)
        stations = np.array([sk.station[r[0]] for r in rings], float)
        at = [_arc_of(joints[j], centres, stations) for j in names]
        root = names[0].rsplit("_", 1)[0]
        if root in ROOT:  # the root sits inside the torso: blend across the stitch instead
            at[0] = stations[0] + ROOT[root]
        stops = [(idx[j], a, stop(j)[1]) for j, a in zip(names, at)]
        pairs[mask], w[mask] = _blend_along(sk.station[mask].astype(float), idx[before], stops)

    # Below each hip the torso shares it, up to half at the crotch ring, fading to nothing at the midline, so
    # the crotch stitch bends with the leg.
    low = (sk.region == TORSO) | (sk.region == BRIDGE)
    bottom = sk.positions[np.asarray(sk.ring["torso"][0]), 1].mean()
    x = sk.positions[:, 0].astype(float)
    for s, side in (("l", 1), ("r", -1)):
        hy = joints[f"hip_{s}"][1]
        share = 0.5 * np.clip((hy - y) / max(hy - bottom, 1e-6), 0, 1) * np.clip(x * side / 0.04, 0, 1)
        m = low & (share > 0)
        pairs[m] = [idx[f"hip_{s}"], idx["pelvis"]]
        w[m] = share[m]
        leg = (sk.region == (LEG_L if s == "l" else LEG_R)) & (pairs[:, 1] == idx["pelvis"])
        # Its top ring's midline vertices stay with the pelvis, the fade gone 6 cm down the leg.
        fade = np.clip(x[leg] * side / 0.04, 0, 1)
        down = np.clip((sk.station[leg] - sk.station[np.asarray(sk.ring[f"leg_{s}"][0])].min()) / 0.06, 0, 1)
        w[leg] *= fade + (1 - fade) * down
    return pairs.astype(np.uint32), w.astype(np.float32)


# ------------------------------------------------------------------------------------------------- the build


def _smooth(x, a, b):
    s = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    return s * s * (3 - 2 * s)


def _widen(pos, region, t: SectionTable, joints: dict, shoulders: float, hips: float):
    """Shoulder and hip breadth as x scales faded by height, applied to the skin so its faces never change."""
    pos = pos.astype(float).copy()
    joints = {n: j.copy() for n, j in joints.items()}
    y = pos[:, 1]
    torso = region == TORSO
    gap = (t.axilla_y - t.crotch_y) / round((CHAINS["torso"] - shoulder_rows(K)) * 0.75)
    neck_y = joints["neck"][1]
    if shoulders != 1:
        w = _smooth(y, t.axilla_y - 4 * gap, t.axilla_y) * (1 - _smooth(y, t.acromion_y, neck_y))
        pos[torso, 0] *= 1 + (shoulders - 1) * w[torso]
        for s, side, arm in (("l", 1, ARM_L), ("r", -1, ARM_R)):
            dx = (shoulders - 1) * joints[f"shoulder_{s}"][0]
            pos[region == arm, 0] += dx
            for n in ("shoulder", "elbow", "wrist"):
                joints[f"{n}_{s}"][0] += dx
    if hips != 1:
        top = joints["spine"][1]
        w = 1 - _smooth(y, joints["pelvis"][1], top)
        pos[torso, 0] *= 1 + (hips - 1) * w[torso]
        knee = joints["knee_l"][1]
        legs = (region == LEG_L) | (region == LEG_R)
        wl = _smooth(y, knee, t.crotch_y)
        pos[legs, 0] *= 1 + (hips - 1) * wl[legs]
        for s in "lr":
            joints[f"hip_{s}"][0] *= hips
    return pos, joints


def _rest_joints(t: SectionTable) -> dict:
    joints = {n: np.asarray(t.joints[n], float) for n in JOINTS}
    joints["head"] = np.asarray(t.head_center, float)
    return joints


def _smoothed(s, level: int):
    """The lofted skin after SMOOTH[level] Taubin steps; soles stay on the ground and the crotch stitch in place."""
    if not SMOOTH[level]:
        return s
    from manifoldx.modeling.skeleton import taubin

    smoothed = taubin(Mesh(positions=s.positions, faces=s.faces), iterations=SMOOTH[level]).positions
    smoothed[:, 1] = np.maximum(smoothed[:, 1], 0.0)
    # The crotch stitch keeps its lofted place: smoothed, its small faces fold when a hip turns.
    stitch = np.r_[
        np.flatnonzero(s.region == BRIDGE),
        np.asarray(s.ring["torso"][0]),
        np.asarray(s.ring["leg_l"][0]),
        np.asarray(s.ring["leg_r"][0]),
    ]
    smoothed[stitch] = s.positions[stitch] + STITCH_SMOOTH * (smoothed[stitch] - s.positions[stitch])
    return replace(s, positions=smoothed)


def _resolution(level: int, detail: float) -> tuple[int, int]:
    if level:
        return LEVELS[level]
    k = max(6, 2 * round(8 * detail))
    return k, int(np.clip(round(1 / detail), 1, 4))


def _body_table(params: HumanParams) -> SectionTable:
    """The section table `params` describe: variants blended, thinned, scaled to stature, legs and spine set."""
    t = _blended(params)
    t = _thinner(t, params)
    t = _scaled(t, params.stature / t.stature)
    t = _legs(t, params.leg_ratio)
    return _spine_on_torso(t)


def rest_skeleton(params: HumanParams) -> dict:
    """The 19 rest joint positions of the figure `params` build, at any level."""
    t = _body_table(params)
    return _widen(np.zeros((0, 3)), np.zeros(0, int), t, _rest_joints(t), params.shoulders, params.hips)[1]


def build_human(params: HumanParams, outfit=None, level: int = 0) -> Figure:
    """The figure `params` describes at `level` (0 near, 1 mid, 2 far), feet on y = 0, facing +Z."""
    params.validate()
    if level not in LEVELS:
        raise ValueError(f"level: {level!r} is not one of {tuple(LEVELS)}")
    t = _body_table(params)
    k, step = _resolution(level, params.detail)
    s = skin(t, k=k, step=step)
    pairs, weights = _weights(s, _rest_joints(t))  # from the lofted rings, before smoothing moves them
    s = _smoothed(s, level)
    pos, joints = _widen(s.positions, s.region, t, _rest_joints(t), params.shoulders, params.hips)
    if level == 2:  # far figures skin rigidly: each vertex follows its heavier joint
        pairs = np.where((weights >= 0.5)[:, None], pairs, pairs[:, ::-1])
        weights = np.ones_like(weights)
    colors = np.broadcast_to(np.array(SKIN_TONES[params.skin], np.float32), pos.shape).copy()
    mesh = Mesh(positions=pos.astype(np.float32), faces=s.faces, colors=colors, joints=pairs, weights=weights)
    parts = {}
    if outfit is not None:
        from manifoldx.modeling.assets.human_dress import dress

        mesh, parts = dress(mesh, s, joints, outfit, level, SKIN_TONES[params.skin])
    return Figure(params, level, mesh, joints, dict(PARENTS), dict(s.ring), parts)


