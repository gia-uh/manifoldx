"""Festival crowd: twenty thousand people steering away from each other on the GPU."""

import numpy as np

import manifoldx as mx
from manifoldx.components import Component, Material, Mesh, Transform
from manifoldx.compute import Compute, ReadsWrites, Uniform
from manifoldx.compute.shader import dot, length, sqrt, u32, vec3, vec4
from manifoldx.resources import DirectionalLight, PhongMaterial, cube, plane
from PIL import Image

CROWD = 20_000


class Velocity(Component):
    """Velocity in xyz; in w, the room this person keeps around them."""

    motion: vec4


engine = mx.Engine("Festival crowd", width=480, height=270)

rng = np.random.default_rng(11)
angle = rng.uniform(0, 2 * np.pi, CROWD)
edge = 20.0 * (1.0 + 0.12 * np.sin(3 * angle) + 0.07 * np.sin(7 * angle))
radius = np.sqrt(rng.uniform(3.0**2, edge**2, CROWD))
pos = np.zeros((CROWD, 3), dtype=np.float32)
pos[:, 0] = np.cos(angle) * radius
pos[:, 2] = np.sin(angle) * radius
pos[:, 1] = 0.85
motion = np.zeros((CROWD, 4), dtype=np.float32)
motion[:, 3] = rng.uniform(0.45, 0.7, CROWD)
height = rng.uniform(0.92, 1.08, CROWD).astype(np.float32)
scale = np.stack([height, height, height], axis=1)

colours = ["#e2683c", "#3c7de2", "#d9b64a", "#3fb37a"]
share = CROWD // len(colours)
watched = []
for k, colour in enumerate(colours):
    lo = k * share
    hi = CROWD if k == len(colours) - 1 else (k + 1) * share
    watched.append(
        engine.spawn(
            Mesh(cube(0.38, 1.7, 0.38)),
            Material(PhongMaterial(colour)),
            Transform(pos=pos[lo:hi], scale=scale[lo:hi]),
            Velocity(motion=motion[lo:hi]),
            n=hi - lo,
        )
    )

engine.spawn(
    Mesh(plane(64, 64)),
    Material(PhongMaterial("#757d68")),
    Transform(pos=(0, -0.01, 0), rot=Transform.rotation(x=-np.pi / 2)),
)
engine.spawn(
    Mesh(plane(40, 40)),
    Material(PhongMaterial("#b8ac96")),
    Transform(pos=(0, 0, 0), rot=Transform.rotation(x=-np.pi / 2)),
)
engine.spawn(
    Mesh(cube(4, 1, 4)),
    Material(PhongMaterial("#8a9bb0")),
    Transform(pos=(0, 0.5, 0)),
)


class CrowdKernel(Compute):
    """Move every person away from their neighbours, one thread per person."""

    transforms: ReadsWrites[Transform]
    people: ReadsWrites[Velocity]

    edge: Uniform[float] = 31.0
    fountain: Uniform[float] = 3.0
    strength: Uniform[float] = 8.0
    max_speed: Uniform[float] = 1.4
    dt: Uniform[float] = "frame_dt"
    n: Uniform[float] = CROWD

    workgroup_size = 64
    dispatch = CROWD

    def main(self, i: int) -> None:
        if i >= u32(self.n):
            return
        pos: vec3 = self.transforms[i].pos
        motion: vec4 = self.people[i].motion
        reach: float = motion.w

        push: vec3 = vec3(0.0, 0.0, 0.0)
        for j in range(u32(self.n)):
            if i == j:
                continue
            away: vec3 = pos - self.transforms[j].pos
            d2: float = dot(away, away)
            if d2 >= reach * reach:
                continue
            if d2 < 1e-6:
                continue
            d: float = sqrt(d2)
            push += away / d * (1.0 - d / reach)

        flat: vec3 = vec3(pos.x, 0.0, pos.z)
        rad: float = length(flat)
        wall: vec3 = vec3(0.0, 0.0, 0.0)
        if rad > self.edge:
            wall = flat / rad * ((self.edge - rad) * 4.0)

        vel: vec3 = vec3(motion.x, motion.y, motion.z)
        vel = vel + (push * self.strength + wall) * self.dt
        speed: float = length(vel)
        if speed > self.max_speed:
            vel = vel / speed * self.max_speed

        self.people[i].motion = vec4(vel.x, vel.y, vel.z, motion.w)
        next: vec3 = pos + vel * self.dt
        nflat: vec3 = vec3(next.x, 0.0, next.z)
        nrad: float = length(nflat)
        if nrad < self.fountain and nrad > 0.001:
            next = vec3(nflat.x / nrad * self.fountain, next.y, nflat.z / nrad * self.fountain)
        self.transforms[i].pos = next


engine.compute(CrowdKernel)
engine.camera.set_pose(position=(0, 42, 18), target=(0, 0.6, 0))
engine.set_sun(DirectionalLight(color="#fff2dd", intensity=3.0, direction=(-0.35, -1, -0.4)))

before = [w.transform.pos.copy() for w in watched]
frame = engine.render_frame(supersample=1)
moved = [np.linalg.norm(w.transform.pos - p) for w, p in zip(watched, before)]
print("four people moved, in metres:", np.round(moved, 3))
Image.fromarray(frame)
