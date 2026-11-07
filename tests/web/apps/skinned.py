"""Skinned entities in frame code: three figures (body and two arms, one joint each) and a box.
Figure 0 is posed at setup. Every frame after the first, one engine.set_joints poses the crowd; from frame 10
figure 1 leaves it after one set_joints with J = 2, so its joint 2 keeps its frame-9 matrix.
Frame 30 destroys figure 2; frame 31 spawns a new figure into its slot, posed from frame 40.
SKINNED_N in the environment spawns that many figures. `--box` poses the box (no joints) and
`--shapes` passes (N, 3, 3, 3) matrices: both raise as the desktop does."""

import math
import os
import sys

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.engine import EntityHandle
from manifoldx.modeling import Mesh as Shape
from manifoldx.resources import PhongMaterial, cube

BOX_POSE = "--box" in sys.argv
BAD_SHAPES = "--shapes" in sys.argv
N = int(os.environ.get("SKINNED_N", "3"))

# Parts: (size, centre). Joint i moves part i.
PARTS = [
    ((0.8, 1.0, 0.4), (0.0, 0.0, 0.0)),  # 0 body
    ((0.2, 0.8, 0.2), (-0.55, 0.1, 0.0)),  # 1 left arm, swings at the shoulder
    ((0.2, 0.8, 0.2), (0.55, 0.1, 0.0)),  # 2 right arm
]


def figure() -> Shape:
    boxes = [Shape.box(*size).translate(centre).recompute_normals() for size, centre in PARTS]
    offsets = np.cumsum([0] + [len(b.positions) for b in boxes[:-1]])
    return Shape(
        positions=np.concatenate([b.positions for b in boxes]),
        faces=np.concatenate([b.faces + o for b, o in zip(boxes, offsets)]).astype(np.uint32),
        normals=np.concatenate([b.normals for b in boxes]),
        joints=np.concatenate([np.full(len(b.positions), i) for i, b in enumerate(boxes)]),
    )


def translation(x: float, y: float, z: float):
    return np.array(
        [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]],
        np.float32,
    )


def about(px: float, py: float, pz: float, axis: str, angle: float):
    """(4, 4) float32 rotation by `angle` about the x or y axis through (px, py, pz)."""
    c = math.cos(angle)
    s = math.sin(angle)
    if axis == "x":
        r = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
            np.float32,
        )
    else:
        r = np.array(
            [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
            np.float32,
        )
    return translation(px, py, pz) @ r @ translation(-px, -py, -pz)


def pose_of(t: float, phase: float):
    """A figure's (3, 4, 4) pose at time t: the body turns, the arms swing opposite."""
    body = about(0.0, 0.0, 0.0, "y", 0.4 * t + phase)
    swing = 0.9 * math.sin(2.0 * t + phase)
    return np.stack(
        [
            body,
            body @ about(-0.55, 0.45, 0.0, "x", swing),
            body @ about(0.55, 0.45, 0.0, "x", -swing),
        ]
    )


class Figures:
    crowd: list[EntityHandle]
    lone: EntityHandle | None
    late: EntityHandle | None
    box: EntityHandle | None
    frame: int

    def __init__(self):
        self.crowd = []
        self.lone = None
        self.late = None
        self.box = None
        self.frame = 0


engine = mx.Engine("Skinned", width=320, height=200)
engine.set_origin(100.0, -50.0)
FIG = figure().to_geometry()
BOX = cube(1.0, 1.0, 1.0)
SKIN = PhongMaterial(color="#e07a3a")
GREY = PhongMaterial(color="#888888")


def spawn_figure(i: int) -> EntityHandle:
    return engine.spawn(Mesh(FIG), Material(SKIN), Transform(pos=(100.0 + 2.0 * i, 0.0, -50.0)))


figures = Figures()
figures.crowd = [spawn_figure(i) for i in range(N)]
figures.box = engine.spawn(Mesh(BOX), Material(GREY), Transform(pos=(100.0, 0.0, -48.0)))
figures.crowd[0].set_joints(pose_of(1.5, 0.3))


def pose(query, dt: float):
    f = figures.frame
    t = engine.elapsed
    if f == 10:
        lone = figures.crowd[1]
        lone.set_joints(np.stack([about(0.0, 0.0, 0.0, "x", 0.7), about(0.0, 0.0, 0.0, "y", 0.2)]))
        figures.lone = lone
        figures.crowd = [h for h in figures.crowd if h.index != lone.index]
    if f == 30:
        gone = figures.crowd[1]
        figures.crowd = [h for h in figures.crowd if h.index != gone.index]
        gone.destroy()
    if f == 31:
        figures.late = spawn_figure(7)
    if f == 40 and figures.late is not None:
        figures.crowd = figures.crowd + [figures.late]
    if BOX_POSE and f == 5 and figures.box is not None:
        figures.box.set_joints(np.stack([np.eye(4, dtype=np.float32)]))
    ix = np.array([h.index for h in figures.crowd])
    if BAD_SHAPES and f == 5:
        bad = np.stack(
            [np.stack([np.eye(3, dtype=np.float32) for j in range(3)]) for h in figures.crowd]
        )
        engine.set_joints(ix, bad)
    if f > 0:  # frame 0 shows the pose setup gave
        poses = [pose_of(t, 0.5 * h.index) for h in figures.crowd]
        engine.set_joints(ix, np.stack(poses))
    figures.frame = f + 1


engine.system(pose)
engine.run()
