"""Three people on the plaza: proportions, poses, skinning and screen space."""

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.humanoid import (
    JOINTS,
    Pose,
    Proportions,
    facing,
    solve,
    spawn_mannequin,
)
from manifoldx.resources import DirectionalLight, StandardMaterial, cube, plane, sphere
from PIL import Image

W, H = 480, 270
engine = mx.Engine("plaza", width=W, height=H)
engine.set_sun(DirectionalLight(color="#fff4e0", intensity=3.0, direction=(-0.5, -1, -0.35)))
engine.spawn(
    Mesh(plane(40, 40)),
    Material(StandardMaterial(color="#c8b89a", roughness=0.9)),
    Transform(rot=Transform.rotation(x=-np.pi / 2)),
)
engine.spawn(
    Mesh(cube(4, 1, 4)),
    Material(StandardMaterial(color="#8a9bb0", roughness=0.6)),
    Transform(pos=(0, 0.5, 0)),
)

wood = Material(StandardMaterial(color="#6b4a2f", roughness=0.8))
BENCH = (-2.8, 0, 4.2)
engine.spawn(Mesh(cube(1.9, 0.08, 0.55)), wood, Transform(pos=(BENCH[0], 0.45, BENCH[2])))
engine.spawn(
    Mesh(cube(1.9, 0.55, 0.08)),
    wood,
    Transform(pos=(BENCH[0], 0.76, BENCH[2] - 0.30)),
)
for side in (-1, 1):
    engine.spawn(
        Mesh(cube(0.08, 0.44, 0.50)),
        wood,
        Transform(pos=(BENCH[0] + side * 0.80, 0.22, BENCH[2])),
    )

woman = Proportions.measured("female")
man = Proportions.measured("male")

seated_fig = solve("seated", woman, height=1.65)
seated = spawn_mannequin(engine, seated_fig, at=BENCH, color="#d1495b")

STAND = (0.1, 0, 3.6)
standing_fig = solve(
    {"base": "stand", "neck": {"lean": 6}}, man, height=1.78, yaw=facing(STAND, (3, 0))
)
standing = spawn_mannequin(engine, standing_fig, at=STAND, color="#2a6f97")

engine.camera.set_pose(position=(-4.2, 2.3, 8.2), target=(0, 0.9, 3.8))
Image.fromarray(engine.render_frame(supersample=1))
UNIT = sphere(1.0, 12)


def rotation(q):
    """The 3x3 of an (x, y, z, w) quaternion, as a Transform reads it."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def frame(pos, rot):
    m = np.eye(4)
    m[:3, :3] = rotation(rot)
    m[:3, 3] = pos
    return m


def skin_matrices(rest, posed):
    """One rest-to-posed matrix per joint, in JOINTS order."""
    return np.array(
        [
            frame(posed.joints[name], posed.rotations[name])
            @ np.linalg.inv(frame(rest.joints[name], rest.rotations[name]))
            for name in JOINTS
        ]
    )


def body(rest):
    """One skinned mesh from the volumes of a neutral PosedFigure."""
    positions, normals, faces, joints = [], [], [], []
    offset = 0
    for part in rest.parts:
        positions.append(UNIT["positions"] * part.radii + part.center)
        normal = UNIT["normals"] / part.radii
        normals.append(normal / np.linalg.norm(normal, axis=1, keepdims=True))
        faces.append(UNIT["indices"].reshape(-1, 3) + offset)
        joints.append(np.full(len(UNIT["positions"]), JOINTS.index(part.joint)))
        offset += len(UNIT["positions"])
    return {
        "positions": np.concatenate(positions).astype(np.float32),
        "normals": np.concatenate(normals).astype(np.float32),
        "indices": np.concatenate(faces).reshape(-1).astype(np.uint32),
        "joints": np.concatenate(joints).astype(np.uint32),
    }


WAVE = (2.7, 0, 4.0)
wave_pose = {
    "base": "stand",
    "shoulder_r": {"abduct": 150, "flex": -15},
    "elbow_r": {"flex": 35},
    "wrist_r": {"flex": 10},
}
rest = solve(Pose(), man, height=1.78)
waving_fig = solve(wave_pose, man, height=1.78, yaw=facing(WAVE, (0, 0)))
waving = engine.spawn(
    Mesh(body(rest)),
    Material(StandardMaterial(color="#e9a23b", roughness=0.6)),
    Transform(pos=WAVE),
)
waving.set_joints(skin_matrices(rest, waving_fig))

engine.camera.set_pose(position=(2.0, 1.3, 6.5), target=(2.7, 0.95, 4.0))
shot = Image.fromarray(engine.render_frame(supersample=1))
engine.camera.set_pose(position=(-4.2, 2.3, 8.2), target=(0, 0.9, 3.8))
shot
Image.fromarray(engine.render_frame(supersample=1))
labels = engine.render_frame(supersample=1, pass_="ids", groups=[seated, standing, [waving.index]])
hand = waving_fig.joints["wrist_r"] + np.array(WAVE)
xy, depth = engine.camera.project([hand], W, H)
x, y = np.round(xy[0]).astype(int)
print(f"wrist pixel ({x}, {y}), {depth[0]:.1f} m away, label {labels[y, x]}")

colors = np.array([[20, 20, 30], [209, 73, 91], [42, 111, 151], [233, 162, 59]], dtype=np.uint8)
Image.fromarray(colors[labels])
