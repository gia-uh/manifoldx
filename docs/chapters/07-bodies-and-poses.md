# Bodies and poses

The plaza has a floor, a fountain and a bench, but nobody on it. The walkers of
chapter 2 were boxes on a path; a visitor should now see people: a woman
sitting on the bench, a man standing by the fountain, and a third person waving
from the far side of the square. The caption that follows the wave needs the
pixel where the hand is, not only its place in metres.

## Proportions

A human is not one shape. What makes a figure read as a person is proportion:
how many heads tall it is, where the crotch, waist and shoulders sit, how thick
a forearm is. ManifoldX keeps those measurements as fractions of stature, so
one table works at any height. `Proportions.measured("female")` uses ANSUR II
means; `Proportions.styled(body, preset)` re-proportions the same table to a
canon such as `"heroic"`, `"anime"` or `"child"`, and `heads` reports the total
height in head heights.

```python {run}
import numpy as np

from manifoldx.humanoid import Proportions

woman = Proportions.measured("female")
heroic = Proportions.styled("male", "heroic")
print(f"measured: {woman.heads:.2f} heads; heroic male: {heroic.heads:.2f}")
```

The measured figure is 7.69 heads tall; the heroic male is 8.5. The numbers are
the point: the solver puts the joints where these ratios say, not at guessed
positions.

## The skeleton

`solve` does not start from a mesh. It starts from a `Joint` tree built by
`build_skeleton`: each `Joint` has a parent, a rest offset, a side (+1 left, -1
right) and a `flex_sign` that makes a positive `flex` bend the joint the
anatomical way. On each joint, `build_parts` puts one or more `PartSpec`s: a
kind (`torso`, `limb`, `joint`, `head`, `nose`), a centre in the joint's frame,
and three radii. The knee, for example, is a small sphere at the joint and a
longer ellipsoid down the calf.

```python {run continue}
from manifoldx.humanoid.skeleton import build_parts, build_skeleton

skeleton = build_skeleton(woman.dims)
specs = build_parts(woman.dims)
knee = skeleton["knee_l"]
print(
    f"{type(knee).__name__}: parent {knee.parent}, "
    f"offset {knee.offset[1]:.3f} m, flex {knee.flex_sign}"
)
print([f"{type(s).__name__}({s.kind})" for s in specs["knee_l"]])
```

The offset is negative because the knee hangs below the hip; a seated pose
turns that offset forward. The two `PartSpec`s are the volumes the mannequin
draws.

## Poses

A pose is a set of angles, not positions. `Pose.parse` takes a library name
(`"stand"`, `"seated"`, `"t"`), a dict with a `base` and overrides, or a
`Pose`. Angles are in degrees: `flex` bends forward, `abduct` lifts a limb away
from the midline, `twist` turns it about the bone, and `lean` tilts the pelvis,
spine or neck sideways. A plural name such as `shoulders` or `knees` sets both
sides; `shoulder_r` sets one. `angles_for` reads back what one joint got.

```python {run continue}
from manifoldx.humanoid import Pose

pose = Pose.parse({"base": "stand", "shoulder_r": {"abduct": 40}, "elbow_r": {"flex": 30}})
print(pose.angles_for("shoulder_r"))
```

Now the solver has everything it needs. `solve(pose, proportions, height=...,
yaw=...)` walks the joint tree, composes the rotations, scales the fractions to
metres and grounds the result: the lowest point of the figure sits at y=0
whatever the pose. It returns a `PosedFigure` with `joints` and `rotations` in
world space, derived `points` such as `head`, `nose` and `hand_l`/`hand_r`, the
rounded `parts`, and `bounds`. `yaw` turns the figure; `facing(origin, target)`
computes the yaw that makes it look at a point.

```python {run continue}
from manifoldx.humanoid import solve

fig = solve(pose, woman, height=1.65, yaw=25)
print(f"posed height {fig.bounds[1][1]:.2f} m, {len(fig.parts)} volumes")
print(f"head at {np.round(fig.points['head'], 2)}")
```

To draw it, `spawn_mannequin` takes the `PosedFigure` and spawns one entity per
`Part`: a shared unit sphere with the part's centre, rotation and radii, plus a
darker nose so the gaze reads. It returns the entity ids, which the id pass
uses later.

```python {run continue}
import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.humanoid import spawn_mannequin
from manifoldx.resources import DirectionalLight, StandardMaterial, plane
from PIL import Image

engine = mx.Engine("figure", width=320, height=320)
engine.set_sun(DirectionalLight(color="#fff4e0", intensity=3.0, direction=(-0.4, -1, -0.3)))
engine.spawn(
    Mesh(plane(6, 6)),
    Material(StandardMaterial(color="#c8b89a", roughness=0.9)),
    Transform(rot=Transform.rotation(x=-np.pi / 2)),
)
spawn_mannequin(engine, fig, color="#d1495b")
engine.camera.set_pose(position=(2.2, 1.2, 3.0), target=(0, 0.9, 0))
Image.fromarray(engine.render_frame(supersample=1))
```

## The people in the plaza

The finished program puts the measurements and poses to work. It builds the
plaza's floor and basin, a wooden bench (a seat, a back and two side panels),
then the people: two mannequins and a skinned figure. The seated woman uses the
library's `"seated"` pose at 1.65 m; the standing man uses `"stand"` with a
slight neck lean at 1.78 m and faces across the square.

```python {run export=examples/people_posed.py}
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
```

The bench's seat is at 0.45 m, which is where the seated figure's pelvis lands;
the back sits behind it, away from the camera. The standing man's
`facing(STAND, (3, 0))` turns him toward the fountain side.

## One mesh, many joints

Each mannequin is 24 entities, and a pose that changes every frame would mean
writing 24 transforms. That is fine for three people and hopeless for a crowd.
A skinned mesh carries the whole body in one geometry. Each vertex has a joint
number, and `EntityHandle.set_joints(matrices)` gives one 4x4 matrix per joint.
The vertex shader moves position and normal by `matrices[joint]`, so the
vertices are uploaded once and under a kilobyte of matrices changes per pose.
Behind `set_joints`, the engine's `JointStore` keeps a block of `MAX_JOINTS`
(32) matrices per entity in one storage buffer, with a CPU mirror; the
humanoid's 15 joints fit comfortably. `Engine.set_joints` poses many skinned
entities in one call.

The humanoid package gives volumes, not a baked skinned mesh, so the chapter
builds one. A `Part` already says which joint moves it and what ellipsoid it
is; we place a unit sphere at every part in a neutral pose, tag its vertices
with the joint's index, and take the rest-to-posed matrix of each joint from two
`PosedFigure`s, one made with an empty `Pose()`. The 3x3 of a quaternion is the
only conversion the solver does not hand us.

```python {run continue export=examples/people_posed.py}
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
```

```python {run continue export=examples/people_posed.py}


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
```

The waving figure is now one entity. `rest` stands at the origin, facing +Z;
the skinned mesh is built there. `waving_fig` is the same man with `shoulder_r`
abducted 150 degrees, the elbow bent 35 and the wrist turned 10;
`skin_matrices` maps every joint from the rest pose to that pose, and
`set_joints` uploads them. The geometry never changes.

```python {run continue export=examples/people_posed.py}
Image.fromarray(engine.render_frame(supersample=1))
```

## Where is the hand?

`Camera.project(points, width, height)` returns pixel coordinates and depth
for world points, with the origin at the top left, using the same view and
projection matrices the renderer uses. Give it the waving figure's right wrist
and it says where on the frame the hand is.

But a pixel is not an answer by itself: the fountain, the bench or another
person may be in front of or behind the wrist.
`render_frame(pass_="ids", groups=...)` draws the scene once more with every
mesh flat in its group's colour and returns an integer label map: 0 for
background, 1 for the first group, 2 for the second, and so on. Non-mesh
entities are hidden and no systems run, so the labels describe the same state
as the last beauty frame. The pass is how a click, a caption or a test knows
which person owns a pixel.

```python {run continue export=examples/people_posed.py}
labels = engine.render_frame(supersample=1, pass_="ids", groups=[seated, standing, [waving.index]])
hand = waving_fig.joints["wrist_r"] + np.array(WAVE)
xy, depth = engine.camera.project([hand], W, H)
x, y = np.round(xy[0]).astype(int)
print(f"wrist pixel ({x}, {y}), {depth[0]:.1f} m away, label {labels[y, x]}")

colors = np.array([[20, 20, 30], [209, 73, 91], [42, 111, 151], [233, 162, 59]], dtype=np.uint8)
Image.fromarray(colors[labels])
```

The wrist lands at (303, 95), 8.2 m away, and the label there is 3: the third
group, the waving man. The caption can now follow the hand, and the same two
calls answer "who is under this pixel?" for any person in the plaza.

The next chapter measures the plaza itself: charts, labels and fields drawn in
the same frame, so the town can report what happens on it.
