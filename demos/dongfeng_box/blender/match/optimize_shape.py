"""Fit the body's shape parameters to the photos' silhouettes by hill climbing, under the car's constraints.

Run: uv run --with trimesh --with scipy --with pillow --with numpy python optimize_shape.py GLB REPORT OUT.json

The search runs on a fast stand-in for the Blender body: a smooth interpolating surface through the same surface
points (Catmull-Rom along the car and around each section), rasterised with each photo's camera. Wheels and
mirrors do not move, so their silhouette is drawn once per view. box_blender.py then builds the real body from the
result, and register.py measures it: the stand-in steers the search, the full harness decides.

Constraints, all by construction or as penalties:
  - symmetric: only the right half of each section is a parameter, mirrored;
  - spec sheet: length fixed by the sections' x, roof crown 1570 mm and half-width 905 mm held within 5 mm;
  - each section's points stay in order (underside < belt < top, roof inside the side);
  - the end sections stay wide (END_HALF), so the plan of the nose and tail comes from the end rounding;
  - smooth: the second difference of every profile along the car is penalised;
  - fair: the change of turning angle along the car and around every section of the interpolated surface
    (shape.fairness), against the hand-drawn shape's, so the inside of a panel cannot ripple while its outline
    fits. The silhouette sees only the outline, and a search on it alone buys IoU with a wavy hood.
Cost: mean over the three photos of (1 - IoU above the sills), plus the penalties. Every few hundred steps the
two registered cameras are re-registered to the current shape (the side camera stays measured).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import minimize

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import regions  # noqa: E402
import register  # noqa: E402
import shape  # noqa: E402

SCALE = 0.25  # 700 px wide is enough for a silhouette and keeps one step near 0.1 s
W, H = int(register.calib.W * SCALE), int(register.calib.H * SCALE)
COLS = ("top", "bottom", "half", "belt", "roof")
FACTOR_KEYS = ("floor", "tuck", "rocker", "belt_in", "glass_mid", "roof_side", "roof_mid", "hood_drop", "hood_roof",
               "round_f", "round_r", "round_p")
FACTOR_LO = np.array([0.3, 0.6, 0.8, 0.8, 0.1, 0.5, 0.1, 30, 0.5, 0, 0, 1.0])
FACTOR_HI = np.array([0.9, 1.0, 1.0, 1.0, 0.9, 0.98, 0.8, 250, 0.98, 300, 300, 6.0])
FACTOR_STEP = np.array([0.04, 0.02, 0.01, 0.01, 0.05, 0.03, 0.05, 15, 0.03, 20, 20, 0.3])
END_ZONE, END_HALF = 150.0, 720.0  # mm: sections this close to an end keep at least this half-width
FAIR_W = 0.01  # cost of bending as unevenly as the hand-drawn shape, per direction


def project(params, pts):
    cx, cy, cz, yaw, pitch, roll, f = params
    right, up, fwd = register.calib.basis(yaw, pitch, roll)
    d = pts - np.array([cx, cy, cz])
    z = d @ fwd
    return np.column_stack([W / 2 + f * SCALE * (d @ right) / z, H / 2 - f * SCALE * (d @ up) / z])


def raster(params, tris):
    uv = project(params, tris.reshape(-1, 3)).reshape(-1, 3, 2)
    img = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(img)
    for t in uv:
        d.polygon([tuple(p) for p in t], fill=1)
    return np.asarray(img)


def catmull(P, n, closed):
    """Interpolate the rows of P (points along axis 0) with n samples per span, through every input point."""
    pts = np.concatenate([P[-1:], P, P[:2]]) if closed else np.concatenate([P[:1], P, P[-1:]])
    spans = len(P) if closed else len(P) - 1
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    out = []
    for i in range(spans):
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    if not closed:
        out.append(P[-1:])
    return np.concatenate(out)


def grid(sections, k):
    """The stand-in surface's points: rings interpolated along the car and around each section."""
    T = shape.targets(sections, k)  # (S, R, 3)
    T = np.stack([catmull(T[:, j], 3, False) for j in range(T.shape[1])], axis=1)  # along the car
    return np.stack([catmull(T[i], 2, True) for i in range(T.shape[0])], axis=0)  # around each ring


def surface(sections, k):
    """Triangles of the stand-in body, its ends closed with fans."""
    T = grid(sections, k)
    S, R, _ = T.shape
    quads = np.concatenate([T[:-1, :, None], np.roll(T[:-1], -1, 1)[:, :, None], np.roll(T[1:], -1, 1)[:, :, None],
                            T[1:, :, None]], axis=2).reshape(-1, 4, 3)
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    caps = []
    for ring in (T[0], T[-1]):
        c = ring.mean(0)
        caps.append(np.stack([np.repeat(c[None], R, 0), ring, np.roll(ring, -1, 0)], axis=1))
    return np.concatenate([tris] + caps)


def static_tris(glb):
    """Wheels and mirrors from the Blender GLB: the parts the shape search does not move."""
    import trimesh

    scene = trimesh.load(glb, force="scene")
    out = []
    for node in scene.graph.nodes_geometry:
        if not (node.startswith("Wheel_") or node.split("_")[0] in ("Cube.009", "Cube.010", "Cube.011", "Cube.012")):
            continue
        tf, g = scene.graph[node]
        v = trimesh.transform_points(scene.geometry[g].vertices, tf)
        v = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])
        out.append(v[scene.geometry[g].faces])
    return np.concatenate(out)


class Problem:
    def __init__(self, glb, report):
        self.sections0, self.k0 = shape.load()
        rep = json.load(open(report))["views"]
        self.static = static_tris(glb)
        self.views = {}
        for photo, mode in register.VIEWS.items():
            _, pmask = regions.car_mask(register.PHOTOS / f"{photo}.webp", width=W)
            params = np.array(rep[photo]["params"], float)
            self.views[photo] = {"mask": pmask, "params": params, "fixed": mode == "fixed"}
            self.redraw_static(photo)
        self.x = np.array([s[0] for s in self.sections0], float)
        # References for smoothness: the hand-drawn shape, not the start, so a wavy start is not taken as normal.
        rows0 = np.array([s[1:] for s in shape.SECTIONS], float)
        self.bend0 = [max(float((self.bend(c) ** 2).sum()), 1e-9) for c in rows0.T]
        self.fair0 = shape.fairness(grid(shape.SECTIONS, shape.FACTORS))

    def bend(self, col):
        """How much a profile's slope changes from one section to the next."""
        return np.diff(np.diff(col) / np.diff(self.x))

    def redraw_static(self, photo):
        v = self.views[photo]
        v["static"] = raster(v["params"], self.static)
        v["cut"] = int(register.sill_cut(v["params"]) * SCALE / register.SCALE)

    def unpack(self, p):
        S = len(self.x)
        rows = p[: S * 5].reshape(S, 5)
        sections = [(x, *r) for x, r in zip(self.x, rows)]
        k = dict(self.k0)
        k.update(dict(zip(FACTOR_KEYS, p[S * 5 :])))
        return sections, k

    def pack(self):
        rows = np.array([s[1:] for s in self.sections0], float).ravel()
        return np.concatenate([rows, [self.k0[key] for key in FACTOR_KEYS]])

    def penalty(self, p):
        S = len(self.x)
        rows = p[: S * 5].reshape(S, 5)
        top, bottom, half, belt, roof = rows.T
        pen = 0.0
        pen += max(0, abs(top.max() - 1570) - 5) / 20 + max(0, abs(half.max() - 905) - 5) / 20
        pen += np.clip(bottom + 80 - belt, 0, None).sum() / 50 + np.clip(belt + 15 - top, 0, None).sum() / 50
        pen += np.clip(roof - 0.97 * half, 0, None).sum() / 50 + np.clip(150 - bottom, 0, None).sum() / 50
        # No silhouette sees the plan of the nose or the tail: left free, the search narrows the end sections into
        # a snout. The ends stay wide (a car's front and back are broad faces) and the end rounding curves them.
        ends = np.minimum(self.x - self.x[0], self.x[-1] - self.x) < END_ZONE
        pen += np.clip(END_HALF - half[ends], 0, None).sum() / 50
        for col, ref in zip(rows.T, self.bend0):  # smoothness: slope change along the car, against the start's
            pen += 0.02 * float((self.bend(col) ** 2).sum()) / ref
        # No folds: neighbouring quads of the interpolated surface whose normals turn by more than 60 degrees mean
        # it creases or crosses itself, which the exact booleans in Blender cannot cut. The start's minimum is 0.87.
        sections, k = self.unpack(p)
        T = grid(sections, k)
        n = np.cross(T[1:, :] - T[:-1, :], np.roll(T[:-1], -1, 1) - T[:-1])
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        folds = ((n[1:] * n[:-1]).sum(-1) < 0.5).sum() + ((n * np.roll(n, -1, 1)).sum(-1) < 0.5).sum()
        pen += float(folds)
        pen += FAIR_W * sum(a / b for a, b in zip(shape.fairness(T), self.fair0))
        f = p[S * 5 :]
        pen += float((np.clip(FACTOR_LO - f, 0, None) / (FACTOR_HI - FACTOR_LO)
                      + np.clip(f - FACTOR_HI, 0, None) / (FACTOR_HI - FACTOR_LO)).sum()) * 5
        return pen

    def ious(self, p):
        sections, k = self.unpack(p)
        tris = surface(sections, k)
        out = {}
        for photo, v in self.views.items():
            m = raster(v["params"], tris) | v["static"]
            c = v["cut"]
            a, b = m[:c], v["mask"][:c]
            out[photo] = (a & b).sum() / max((a | b).sum(), 1)
        return out

    def cost(self, p):
        return float(np.mean([1 - i for i in self.ious(p).values()])) + self.penalty(p)

    def reregister(self, p):
        sections, k = self.unpack(p)
        tris = surface(sections, k)
        for photo, v in self.views.items():
            if v["fixed"]:
                continue
            scale = np.array([0.2, 0.2, 0.1, 2.0, 1.0, 1.0, 200.0])
            base = v["params"].copy()

            def c(s):
                prm = base + s * scale
                m = raster(prm, tris) | raster(prm, self.static)
                cut = int(register.sill_cut(prm) * SCALE / register.SCALE)
                a, b = m[:cut], v["mask"][:cut]
                return 1 - (a & b).sum() / max((a | b).sum(), 1)

            r = minimize(c, np.zeros(7), method="Powell", options={"xtol": 0.02, "ftol": 1e-4, "maxfev": 150})
            v["params"] = base + r.x * scale
            self.redraw_static(photo)


def main():
    glb, report, out = sys.argv[1:4]
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
    rng = np.random.default_rng(7)
    prob = Problem(glb, report)
    p = prob.pack()
    S = len(prob.x)
    step = np.concatenate([np.full(S * 5, 25.0), FACTOR_STEP])
    best = prob.cost(p)
    start = dict(prob.ious(p))
    print(f"start cost {best:.4f} ious {[round(v, 4) for v in start.values()]}", flush=True)
    t0, accepted = time.time(), 0
    for it in range(1, steps + 1):
        idx = rng.choice(len(p), size=rng.integers(1, 4), replace=False)
        trial = p.copy()
        trial[idx] += rng.normal(0, 1, len(idx)) * step[idx]
        c = prob.cost(trial)
        if c < best:
            p, best = trial, c
            step[idx] *= 1.15
            accepted += 1
        else:
            step[idx] *= 0.97
        step = np.clip(step, np.concatenate([np.full(S * 5, 2.0), FACTOR_STEP / 20]), None)
        if it % 500 == 0:
            prob.reregister(p)
            best = prob.cost(p)
            ious = prob.ious(p)
            print(f"step {it} cost {best:.4f} ious {[round(v, 4) for v in ious.values()]} accepted {accepted} "
                  f"{time.time() - t0:.0f}s", flush=True)
    sections, k = prob.unpack(p)
    json.dump({"sections": [[round(float(v), 1) for v in s] for s in sections],
               "factors": {key: round(float(k[key]), 4) for key in FACTOR_KEYS},
               "start_ious": {a: round(float(b), 4) for a, b in start.items()},
               "end_ious": {a: round(float(b), 4) for a, b in prob.ious(p).items()},
               "cameras": {a: [round(float(x), 4) for x in v["params"]] for a, v in prob.views.items()}},
              open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
