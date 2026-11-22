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
from manifoldx.modeling.assets.human_skin import ARM_L, ARM_R, LEG_L, LEG_R, LEVELS, TORSO, skin
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
        """A plausible person: stature around 1.63 m or 1.75 m by build, the rest within the middle of its range."""
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
            age=str(g.choice(spec["age"]["choices"])),
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


def _resolution(level: int, detail: float) -> tuple[int, int]:
    if level:
        return LEVELS[level]
    k = max(6, 2 * round(8 * detail))
    return k, int(np.clip(round(1 / detail), 1, 4))


def build_human(params: HumanParams, outfit=None, level: int = 0) -> Figure:
    """The figure `params` describes at `level` (0 near, 1 mid, 2 far), feet on y = 0, facing +Z."""
    params.validate()
    if level not in LEVELS:
        raise ValueError(f"level: {level!r} is not one of {tuple(LEVELS)}")
    t = _blended(params)
    t = _thinner(t, params)
    t = _scaled(t, params.stature / t.stature)
    t = _legs(t, params.leg_ratio)
    t = _spine_on_torso(t)
    k, step = _resolution(level, params.detail)
    s = skin(t, k=k, step=step)
    joints = {n: np.asarray(t.joints[n], float) for n in JOINTS}
    joints["head"] = np.asarray(t.head_center, float)
    pos, joints = _widen(s.positions, s.region, t, joints, params.shoulders, params.hips)
    colors = np.broadcast_to(np.array(SKIN_TONES[params.skin], np.float32), pos.shape).copy()
    mesh = Mesh(positions=pos.astype(np.float32), faces=s.faces, colors=colors)
    return Figure(params, level, mesh, joints, dict(PARENTS), dict(s.ring))


