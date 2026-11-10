"""Dongfeng BOX, modelled from its photos: a lofted body, glass, wheels, in a studio.

Run: uv run python demos/dongfeng_box/box.py [photo|poster]   (renders stills; needs profile.json from profile.py)
"""

import json
import sys
from pathlib import Path

import numpy as np

import manifoldx as mx
from manifoldx.components import Material, Mesh, Transform
from manifoldx.modeling.mesh import Mesh as MeshData
from manifoldx.modeling.primitives import cylinder
from manifoldx.resources import DirectionalLight, PointLight, StandardMaterial

HERE = Path(__file__).parent
P = json.load(open(HERE / "profile.json"))
MM = 0.001
L, W, H = (d * MM for d in P["dims"])
TYRE_R = P["tyre_d"] / 2 * MM
PAINT = "#8fb6d3"
STATIONS, RING = 160, 72


def plan(x):
    """Half-width factor along the car seen from above: rounder at the nose than at the tail."""
    u = np.clip(2 * x / L, -1, 1)
    p = np.where(u < 0, 6.0, 7.0)
    return np.clip(1 - np.abs(u) ** p, 0, 1) ** (1 / p)


def loft(xs, top, bottom, half_width, n=8.0):
    """A closed body from cross-sections: at each x, a rounded section from bottom to top whose width at each height
    comes from half_width(y). The ends are capped with a fan."""
    th = np.linspace(0, 2 * np.pi, 16 * RING, endpoint=False)
    s, c = np.sin(th), np.cos(th)
    e = 2 / n
    rings = []
    for x, t, b in zip(xs, top, bottom):
        mid, h = (t + b) / 2, (t - b) / 2
        y = mid + h * np.sign(s) * np.abs(s) ** e
        z = np.sign(c) * half_width(y) * plan(x) * np.abs(c) ** e
        # Resample the dense ring by arc length, so flat sides get as many points as the rounded corners.
        seg = np.hypot(np.diff(y, append=y[0]), np.diff(z, append=z[0]))
        arc = np.concatenate([[0], np.cumsum(seg)])
        at = np.linspace(0, arc[-1], RING, endpoint=False)
        yy = np.interp(at, arc, np.append(y, y[0]))
        zz = np.interp(at, arc, np.append(z, z[0]))
        rings.append(np.column_stack([np.full(RING, x), yy, zz]))
    pos = np.vstack(rings + [[[xs[0], (top[0] + bottom[0]) / 2, 0]], [[xs[-1], (top[-1] + bottom[-1]) / 2, 0]]])
    faces = []
    for i in range(len(xs) - 1):
        for j in range(RING):
            a, b = i * RING + j, i * RING + (j + 1) % RING
            faces += [[a, a + RING, b], [b, a + RING, b + RING]]
    f0, f1 = len(pos) - 2, len(pos) - 1
    last = (len(xs) - 1) * RING
    for j in range(RING):
        faces += [[f0, j, (j + 1) % RING], [f1, last + (j + 1) % RING, last + j]]
    return MeshData(positions=pos.astype(np.float32), faces=np.array(faces, dtype=np.uint32))


def body():
    sp = P["side"]
    xs_mm = np.array(sp["x"])
    pick = np.linspace(0, len(xs_mm) - 1, STATIONS).astype(int)
    xs = xs_mm[pick] * MM
    top = np.array(sp["top"])[pick] * MM
    bottom = np.array(sp["bottom"])[pick] * MM
    fy = np.array(P["front"]["y"]) * MM
    fw = np.array(P["front"]["half_width"]) * MM
    order = np.argsort(fy)
    hw = lambda y: np.interp(y, fy[order], fw[order])
    return loft(xs, top, bottom, hw)


def submesh(mesh, keep, uv_of):
    """The faces where keep is True, as their own mesh, with UVs from uv_of(positions)."""
    faces = mesh.faces[keep]
    used, inverse = np.unique(faces, return_inverse=True)
    pos = mesh.positions[used]
    return MeshData(positions=pos, faces=inverse.reshape(-1, 3).astype(np.uint32), uvs=uv_of(pos).astype(np.float32))


def body_parts():
    """The body split by which way each face looks, each part with UVs in its texture's frame."""
    mesh = body()
    sp, fp = P["side"], P["front"]
    v0, v1, v2 = (mesh.positions[mesh.faces[:, k]] for k in range(3))
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    if fn[np.argmax(mesh.positions[mesh.faces].mean(axis=1)[:, 1]), 1] < 0:
        fn = -fn  # the loft winds inwards; only the facing matters here
    ax = np.argmax(np.abs(fn), axis=1)
    s = sp["scale_mm_per_px"]
    side_uv = lambda p: np.column_stack([np.interp(p[:, 0] / MM, sp["anchors_mm"], sp["anchors_px"]) / 1400,
                                         (sp["ground"] - p[:, 1] / MM / s) / 933])
    front_uv = lambda p: np.column_stack([(700 + p[:, 2] / MM / fp["scale_x"]) / 1400,
                                          (860 - p[:, 1] / MM / fp["scale_y"]) / 933])
    rear_uv = lambda p: np.column_stack([0.5 + p[:, 2] / W, 1 - p[:, 1] / H])
    top_uv = lambda p: np.column_stack([(p[:, 0] + L / 2) / L, 0.5 + p[:, 2] / W])
    return {
        "side": submesh(mesh, ax == 2, side_uv),
        "front": submesh(mesh, (ax == 0) & (fn[:, 0] < 0), front_uv),
        "rear": submesh(mesh, (ax == 0) & (fn[:, 0] > 0), rear_uv),
        "top": submesh(mesh, (ax == 1) & (fn[:, 1] > 0), top_uv),
        "under": submesh(mesh, (ax == 1) & (fn[:, 1] < 0), top_uv),
    }


def revolve(profile, segments=64):
    """Spin a (radius, axial) polyline around the z axis into a surface of revolution."""
    prof = np.asarray(profile, dtype=np.float32)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    pos = np.array([[r * np.cos(t), r * np.sin(t), a] for t in ang for r, a in prof], dtype=np.float32)
    n = len(prof)
    faces = []
    for i in range(segments):
        k = (i + 1) % segments
        for j in range(n - 1):
            a0, a1, b0, b1 = i * n + j, i * n + j + 1, k * n + j, k * n + j + 1
            faces += [[a0, b0, a1], [a1, b0, b1]]
    return MeshData(positions=pos, faces=np.array(faces, dtype=np.uint32))


def disc(radius, segments=64):
    """A flat disc facing +z, with planar UVs spanning the unit square, for a photo of the wheel face."""
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    pos = np.vstack([[[0, 0, 0]], np.column_stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(segments)])])
    uvs = 0.5 + 0.5 * pos[:, :2] / radius * [1, -1]
    faces = [[0, 1 + i, 1 + (i + 1) % segments] for i in range(segments)]
    return MeshData(positions=pos.astype(np.float32), faces=np.array(faces, dtype=np.uint32),
                    uvs=uvs.astype(np.float32))


RIM_R = 17 * 25.4 / 2 * MM


def wheel_meshes():
    """A tyre spun from its cross-section (sidewalls bulge, the tread is rounded), and the wheel's face."""
    w, r0, r1 = 0.205 / 2, RIM_R, TYRE_R
    t = np.linspace(-np.pi / 2, np.pi / 2, 9)
    shoulder = 0.035
    outer = [(r1 - shoulder + shoulder * np.cos(a), (w - shoulder) * np.sign(a) + shoulder * np.sin(a)) for a in t]
    profile = [(r0, w * 0.85)] + [(r0 + (r1 - r0) * 0.5, w)] + outer[::-1] + [(r0 + (r1 - r0) * 0.5, -w), (r0, -w * 0.85)]
    return revolve(profile), disc(RIM_R * 0.98)


def quat_x(deg):
    a = np.radians(deg) / 2
    return (np.sin(a), 0.0, 0.0, np.cos(a))


def build(engine, variant):
    out = HERE / "out"
    tex = lambda name: mx.load_texture(engine, out / f"tex-{name}.png")
    maps = {"side": f"side-{variant}", "front": f"front-{variant}", "rear": "rear", "top": "top", "under": "top"}
    for part, mesh in body_parts().items():
        mat = StandardMaterial(color="#ffffff", roughness=0.3, metallic=0.2, albedo_map=tex(maps[part]))
        engine.spawn(Mesh(mesh.to_geometry()), Material(mat), Transform(pos=(0, 0, 0)), n=1)
    tyre, rim = wheel_meshes()
    wheel_tex = tex("wheel")
    track = W / 2 - 0.205 / 2 - 0.06
    for w in P["side"]["wheels"]:
        for side in (-1, 1):
            pos = (w["x"] * MM, TYRE_R, side * track)
            engine.spawn(Mesh(tyre.to_geometry()), Material(StandardMaterial(color="#151515", roughness=0.85)),
                         Transform(pos=pos), n=1)
            face = (w["x"] * MM, TYRE_R, side * (track + 0.205 * 0.4))
            mat = StandardMaterial(color="#ffffff", roughness=0.35, metallic=0.5, albedo_map=wheel_tex)
            engine.spawn(Mesh(rim.to_geometry()), Material(mat),
                         Transform(pos=face, rot=(0.0, 0.0, 0.0, 1.0) if side > 0 else (0.0, 1.0, 0.0, 0.0)), n=1)
    floor = cylinder(6.0, 0.02, 96)
    engine.spawn(Mesh(floor.to_geometry()), Material(StandardMaterial(color="#c9c9c9", roughness=0.9)),
                 Transform(pos=(0, -0.01, 0)), n=1)


VIEWS = {  # name: (camera position, target), roughly where each photo was shot from
    "04-side": ((0.0, 0.75, 7.2), (0.0, 0.72, 0.0)),
    "03-front": ((-6.2, 0.95, 0.0), (0.0, 0.70, 0.0)),
    "02-three-quarter": ((-4.6, 1.25, 4.6), (0.0, 0.65, 0.0)),
}


def main(out, variant):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    engine = mx.Engine("Dongfeng BOX", width=1400, height=933)
    engine.background_color = (0.62, 0.62, 0.63)
    engine.on("startup")(lambda _: build(engine, variant))
    engine.set_lights([PointLight(position=(-3.0, 4.0, 4.0), color="#ffffff", intensity=40),
                       PointLight(position=(3.0, 3.5, -3.0), color="#ffffff", intensity=25)])
    engine.set_environment("studio")
    from PIL import Image
    for name, (pos, target) in VIEWS.items():
        engine.camera.position = np.array(pos, dtype=np.float32)
        engine.camera.target = np.array(target, dtype=np.float32)
        engine.camera.fov = 30
        Image.fromarray(engine.render_frame(supersample=2)).save(out / f"render-{variant}-{name}.png")
        print("rendered", name)


if __name__ == "__main__":
    main(HERE / "out", sys.argv[1] if len(sys.argv) > 1 else "photo")
