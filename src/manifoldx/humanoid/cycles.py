"""Idle, walk, run and cross cycles for figures from ``build_human``, as joint angles and skinning matrices.

Angles follow ``humanoid.solve``: each joint's local rotation is Rz(abduct * side + lean) . Rx(flex * flex_sign)
. Ry(twist * side), in degrees, applied in its parent's frame. Zero is the limbs straight down, the spine up and
the foot along +Z. ``rest_angles`` reads the figure's own rest pose (the table's A-pose, an elder's lean) in that
convention, and ``pose_matrices`` turns any angles into the figure's skinning matrices, the identity at rest.
A cycle's legs are solved so a planted ankle stays put in world space while the root advances by speed.
"""

from __future__ import annotations

import math
import numpy as np

from manifoldx.humanoid import _quat as q
from manifoldx.modeling.assets.human import PARENTS, HumanParams, rest_skeleton
from manifoldx.modeling.assets.human_sections import JOINTS

KINDS = ("idle", "walk", "run", "cross")
KEY_PHASES = 8  # level 2 snaps its phase to this many poses per cycle
REACH = 0.995  # of the leg's length a planted leg may stretch
CROSS_TURN = 45.0  # degrees the neck turns each way while crossing

_SPINE = ("pelvis", "spine", "chest", "neck", "head")
_CHILD = {"spine": "chest", "chest": "neck", "neck": "head"}  # the pelvis keeps an upright frame
for _s in "lr":
    _CHILD |= {f"shoulder_{_s}": f"elbow_{_s}", f"elbow_{_s}": f"wrist_{_s}", f"hip_{_s}": f"knee_{_s}",
               f"knee_{_s}": f"ankle_{_s}", f"ankle_{_s}": f"toe_{_s}"}


def _side(name: str) -> int:
    return -1 if name.endswith("_r") else 1


def _flex_sign(name: str) -> int:
    return 1 if name in _SPINE or name.startswith("knee") else -1


def _zero_direction(name: str) -> np.ndarray:
    if name in _SPINE:
        return np.array([0.0, 1.0, 0.0])
    if name.startswith(("ankle", "toe")):
        return np.array([0.0, 0.0, 1.0])
    return np.array([0.0, -1.0, 0.0])


def _local(name: str, a: dict) -> np.ndarray:
    side = _side(name)
    r = q.axis_angle((0, 0, 1), a.get("abduct", 0.0) * side + a.get("lean", 0.0))
    r = q.mul(r, q.axis_angle((1, 0, 0), a.get("flex", 0.0) * _flex_sign(name)))
    return q.mul(r, q.axis_angle((0, 1, 0), a.get("twist", 0.0) * side))


def _inverse(r) -> np.ndarray:
    return np.array([-r[0], -r[1], -r[2], r[3]])


def _swing(u, v) -> np.ndarray:
    """The shortest rotation taking unit u to unit v."""
    r = np.r_[np.cross(u, v), 1.0 + np.dot(u, v)]
    return r / np.linalg.norm(r)


def _angles_of(name: str, r) -> dict:
    """flex, abduct and twist whose local rotation (Rz . Rx . Ry) is the quaternion r."""
    m = q.to_matrix(r)
    flex = math.degrees(math.asin(float(np.clip(m[2, 1], -1, 1))))
    twist = math.degrees(math.atan2(-m[2, 0], m[2, 2]))
    abduct = math.degrees(math.atan2(-m[0, 1], m[1, 1]))
    side = _side(name)
    return {"flex": flex * _flex_sign(name), "abduct": abduct * side, "twist": twist * side}


def _rest_from_joints(joints: dict) -> tuple[dict, dict]:
    """Rest angles and world rotations: each bone's rest direction in its parent's rest frame."""
    angles, world = {}, {}
    for name in JOINTS:
        parent = PARENTS[name]
        frame = world[parent] if parent else np.array([0.0, 0.0, 0.0, 1.0])
        if name in _CHILD:
            bone = np.asarray(joints[_CHILD[name]], float) - np.asarray(joints[name], float)
            local = _swing(_zero_direction(name), q.rotate(_inverse(frame), bone / np.linalg.norm(bone)))
        else:  # head, wrist, toe: no bone of their own, they carry on their parent's
            local = np.array([0.0, 0.0, 0.0, 1.0])
        angles[name] = _angles_of(name, local)
        world[name] = q.mul(frame, _local(name, angles[name]))
    return angles, world


def rest_angles(figure) -> dict[str, dict[str, float]]:
    """The figure's rest pose in the solve convention: the A-pose of its table, plus the elder lean."""
    return {n: dict(a) for n, a in _rest_from_joints(figure.joints)[0].items()}


def _forward(joints: dict, rest_world: dict, angles: dict, offset) -> tuple[dict, dict]:
    pos, rot = {}, {}
    for name in JOINTS:
        parent = PARENTS[name]
        local = _local(name, angles[name])
        if parent is None:
            pos[name] = np.asarray(joints[name], float) + offset
            rot[name] = local
        else:
            turn = q.mul(rot[parent], _inverse(rest_world[parent]))
            pos[name] = pos[parent] + q.rotate(turn, np.asarray(joints[name], float) - joints[parent])
            rot[name] = q.mul(rot[parent], local)
    return pos, rot


def pose_matrices(figure, angles: dict, offset=None) -> np.ndarray:
    """(19, 4, 4) float32 skinning matrices in JOINTS order; joints missing from `angles` keep their rest.

    `offset` moves the pelvis (``root_offset``); the caller's transform carries the figure along the ground.
    """
    rest, rest_world = _rest_from_joints(figure.joints)
    full = {n: angles.get(n, rest[n]) for n in JOINTS}
    offset = np.zeros(3) if offset is None else np.asarray(offset, float)
    pos, rot = _forward(figure.joints, rest_world, full, offset)
    mats = np.tile(np.eye(4), (len(JOINTS), 1, 1))
    for i, name in enumerate(JOINTS):
        r = q.to_matrix(q.mul(rot[name], _inverse(rest_world[name])))
        mats[i, :3, :3] = r
        mats[i, :3, 3] = pos[name] - r @ np.asarray(figure.joints[name], float)
    return mats.astype(np.float32)


# ------------------------------------------------------------------------------------------------ the cycles


_CACHE: dict = {}


def _cached(key, make):
    if key not in _CACHE:
        if len(_CACHE) > 1024:
            _CACHE.clear()
        _CACHE[key] = make()
    return _CACHE[key]


def _rest(params: HumanParams):
    """Rest joints, angles and world rotations of the figure `params` build (keyed by repr: params are unhashable)."""
    return _cached(("rest", repr(params)), lambda: (lambda j: (j, *_rest_from_joints(j)))(rest_skeleton(params)))


def _stride(kind: str, speed: float, params: HumanParams) -> float:
    """Two steps, in metres: 0.83 statures at 1.4 m/s, growing with sqrt(speed / 1.4) to 1.2 statures."""
    s = params.stature * min(1.2, 0.83 * math.sqrt(max(speed, 0.1) / 1.4))
    return s * (0.85 if params.age == "elder" else 1.0)


def cadence(kind: str, speed: float, params: HumanParams) -> float:
    """Cycles per second: speed over stride; idle breathes at 0.25 Hz."""
    _check(kind)
    if kind == "idle":
        return 0.25
    return speed / _stride(kind, speed, params)


def _check(kind: str):
    if kind not in KINDS:
        raise ValueError(f"kind: {kind!r} is not one of {KINDS}")


def _gait(kind: str, speed: float, params: HumanParams) -> dict:
    run = kind == "run"
    if kind == "idle":
        return {"plant": 1.0, "reach": 0.0, "lift": 0.0, "bob": 0.0}
    plant = 0.15 if run else 0.25
    return {
        "plant": plant,
        "reach": plant * _stride(kind, speed, params) / 2,
        "lift": (0.30 if run else 0.07) * params.stature / 1.7,
        "bob": 0.05 if run else 0.02,
    }


def _foot(g: dict, u: float) -> tuple[float, float]:
    """Ankle z relative to its rest and lift above it, at `u` of the cycle since this foot was planted."""
    a, p = g["reach"], g["plant"]
    if u < p:
        return a * (1 - 2 * u / p), 0.0
    # Swinging, the foot covers a stride in world space along a smoothstep, so it lifts off and lands at rest;
    # it clears the ground early in the swing, so the knee straightens before the foot reaches forward.
    s, stride = (u - p) / (1 - p), 2 * a / p
    ahead = stride * s * s * (3 - 2 * s) - (1 - p) * stride * s
    return -a + ahead, g["lift"] * math.sin(math.pi * s**0.7)


def _centre(joints, s: str, g: dict) -> float:
    """Moving, the step is centred under the hip rather than at the ankle's rest, which sits behind it."""
    return 0.0 if g["reach"] == 0 else float(joints[f"hip_{s}"][2] - joints[f"ankle_{s}"][2])


def _bob(kind: str, phase: float, g: dict) -> float:
    """Highest at mid-stance, twice per cycle."""
    return g["bob"] * 0.5 * (1 + math.cos(4 * math.pi * (phase - g["plant"] / 2)))


def _sway(kind: str, phase: float) -> float:
    if kind == "idle":  # a 2 cm weight shift to the left and back once a cycle
        return 0.02 * 0.5 * (1 - math.cos(2 * math.pi * phase))
    return 0.01 * math.cos(2 * math.pi * phase)  # over the planted foot


def _drop(kind: str, speed: float, params: HumanParams) -> float:
    return _cached(("drop", kind, speed, repr(params)), lambda: _reach_drop(kind, speed, params))


def _reach_drop(kind: str, speed: float, params: HumanParams) -> float:
    """How far the pelvis sits below rest so every planted ankle stays within REACH of the leg's length."""
    joints, _, _ = _rest(params)
    g = _gait(kind, speed, params)
    need = 0.0
    for s, start in (("l", 0.0), ("r", 0.5)):
        hip, knee, ankle = (np.asarray(joints[f"{n}_{s}"]) for n in ("hip", "knee", "ankle"))
        reach = REACH * (np.linalg.norm(knee - hip) + np.linalg.norm(ankle - knee))
        for u in np.linspace(0, g["plant"], 17) if g["plant"] < 1 else np.linspace(0, 1, 33):
            phase = (start + u) % 1.0
            dz, _ = _foot(g, u)
            dx = ankle[0] - hip[0] - _sway(kind, phase)
            dz = ankle[2] - hip[2] + _centre(joints, s, g) + dz
            height = math.sqrt(max(reach**2 - dx**2 - dz**2, 1e-6))
            need = max(need, (hip[1] - ankle[1]) - height + _bob(kind, phase, g))
    return need


def root_offset(kind: str, phase: float, speed: float, params: HumanParams) -> np.ndarray:
    """(3,) pelvis offset from rest: bob, sway, and the drop that lets the planted legs reach."""
    _check(kind)
    g = _gait(kind, speed, params)
    y = _bob(kind, phase, g) - _drop(kind, speed, params)
    return np.array([_sway(kind, phase), y, 0.0])


def _leg(s: str, kind: str, phase: float, speed: float, params, pelvis: dict, offset) -> dict:
    """Hip, knee and ankle angles placing the ankle where the gait wants it, the foot kept flat."""
    joints, rest, world = _rest(params)
    g = _gait(kind, speed, params)
    u = (phase - (0.0 if s == "l" else 0.5)) % 1.0
    dz, lift = _foot(g, u)
    target = np.asarray(joints[f"ankle_{s}"], float) + (0.0, lift, _centre(joints, s, g) + dz)

    rot_pelvis = _local("pelvis", pelvis)
    turn = q.mul(rot_pelvis, _inverse(world["pelvis"]))
    hip = np.asarray(joints["pelvis"], float) + offset + q.rotate(turn, joints[f"hip_{s}"] - joints["pelvis"])
    t = q.rotate(_inverse(rot_pelvis), target - hip)
    gamma = math.atan2(t[0], -t[1])  # the leg plane's tilt about z
    c, sn = math.cos(-gamma), math.sin(-gamma)
    y, z = sn * t[0] + c * t[1], t[2]
    l1 = float(np.linalg.norm(joints[f"knee_{s}"] - joints[f"hip_{s}"]))
    l2 = float(np.linalg.norm(joints[f"ankle_{s}"] - joints[f"knee_{s}"]))
    d = float(np.clip(math.hypot(y, z), abs(l1 - l2) + 1e-6, (l1 + l2) * 0.9999))
    beta = math.acos(float(np.clip((d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1, 1)))
    alpha = math.atan2(l2 * math.sin(beta), l1 + l2 * math.cos(beta))
    theta = math.atan2(-z, -y) - alpha  # the thigh's rotation about x
    side = _side(f"hip_{s}")
    hip_a = {"flex": -math.degrees(theta), "abduct": math.degrees(gamma) * side, "twist": 0.0}
    knee_a = {"flex": math.degrees(beta), "abduct": 0.0, "twist": 0.0}
    # The foot keeps its rest turn relative to the pelvis: flat on the ground while the pelvis is upright.
    shin = q.mul(rot_pelvis, q.mul(_local(f"hip_{s}", hip_a), _local(f"knee_{s}", knee_a)))
    foot = q.mul(turn, world[f"ankle_{s}"])
    return {f"hip_{s}": hip_a, f"knee_{s}": knee_a, f"ankle_{s}": _angles_of(f"ankle_{s}", q.mul(_inverse(shin), foot))}


def _plus(a: dict, **d) -> dict:
    return {k: a.get(k, 0.0) + d.get(k, 0.0) for k in ("flex", "abduct", "twist")}


def cycle_angles(kind: str, phase: float, speed: float, params: HumanParams) -> dict[str, dict[str, float]]:
    """Every joint's angles at `phase` of the `kind` cycle at `speed` m/s; at phase 0 the left foot lands."""
    _check(kind)
    _, rest, _ = _rest(params)
    out = {n: dict(a) for n, a in rest.items()}
    w = 2 * math.pi * phase
    run = kind == "run"
    heavy = 8.0 * max(0.0, params.weight - 1.0) / 0.5  # heavy arms ride wider
    if kind == "idle":
        out["chest"] = _plus(rest["chest"], flex=1.5 * math.sin(w))
        for s in "lr":
            out[f"shoulder_{s}"] = _plus(rest[f"shoulder_{s}"], abduct=heavy, flex=1.0 * math.sin(w))
            out[f"elbow_{s}"] = _plus(rest[f"elbow_{s}"], flex=8.0)
    else:
        swing = math.cos(w)  # +1 when the left leg reaches forward
        out["pelvis"] = _plus(rest["pelvis"], twist=-5.0 * swing)
        out["spine"] = _plus(rest["spine"], twist=5.0 * swing, flex=10.0 if run else 0.0)
        arm, elbow = (45.0, 90.0) if run else (20.0, 22.5)
        for s, sign in (("l", -1), ("r", 1)):
            out[f"shoulder_{s}"] = _plus(rest[f"shoulder_{s}"], flex=sign * arm * swing, abduct=heavy)
            bend = elbow if run else elbow + 7.5 * sign * swing  # 15 behind, 30 in front
            out[f"elbow_{s}"] = _plus(rest[f"elbow_{s}"], flex=bend)
    if kind == "cross":
        out["neck"] = _plus(rest["neck"], twist=CROSS_TURN * math.sin(w))
    offset = root_offset(kind, phase, speed, params)
    for s in "lr":
        out |= _leg(s, kind, phase, speed, params, out["pelvis"], offset)
    return out


# --------------------------------------------------------------------------------------- levels in motion


def update_every(level: int) -> int:
    """Frames between pose updates: level 0 every frame, 1 every second, 2 every frame on snapped phases."""
    return {0: 1, 1: 2, 2: 1}[level]


def snap_phase(phase: float, level: int) -> float:
    """Level 2 holds one of KEY_PHASES poses per cycle, so its matrices can be shared and cached."""
    return math.floor(phase * KEY_PHASES) / KEY_PHASES if level == 2 else phase
