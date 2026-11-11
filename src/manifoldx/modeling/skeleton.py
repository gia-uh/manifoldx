"""A skin around a skeleton: one connected solid grown from a tree of joints, fitted to drawings.

A skeleton is joints (points in metres) joined by bones. Each bone is a round cone between the radii of its
two joints, the cones melt into each other with a smooth minimum, and the solid is the zero level of that
field, meshed by surface nets. Joints that share bones give one closed, connected mesh by construction: a
figure's head, neck, shoulders and arms come out as one piece, with no part to attach.

Fitted to drawings, the skeleton gives the topology and the depth, and the drawings give the outline: the
field is intersected with the visual hull of the views (each view's mask swept along its line of sight), and
:func:`fit_radii` reads each joint's radius from how wide the drawings are across its bones.

A view here is ``(right, up, mask, frame)``: the view's right and up unit vectors in the model's axes, its
mask (row 0 at the top) and ``frame = (u_min, u_max, v_min, v_max)``, the metres the mask spans along them.
"""

from __future__ import annotations

import numpy as np

from manifoldx.modeling.mesh import Mesh


def round_cone_sdf(points, a, b, ra, rb) -> np.ndarray:
    """Signed distance (approximate) to a capsule from ``a`` with radius ``ra`` to ``b`` with radius ``rb``."""
    points = np.asarray(points, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    nearest = a + t[:, None] * ab
    return np.linalg.norm(points - nearest, axis=1) - (ra + t * (rb - ra))


def _smin(a, b, k):
    """A smooth minimum that rounds the join of two fields over a width ``k`` (the plain minimum at 0)."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def skeleton_sdf(points, joints, radii, bones, blend=0.04) -> np.ndarray:
    """The skeleton's field at ``points``: its bones' round cones joined by a smooth minimum of width ``blend``."""
    joints, radii = np.asarray(joints, float), np.asarray(radii, float)
    field = None
    for i, j in bones:
        cone = round_cone_sdf(points, joints[i], joints[j], radii[i], radii[j])
        field = cone if field is None else _smin(field, cone, blend)
    return field


def signed_distance(mask, limit=40) -> np.ndarray:
    """Signed distance in pixels to a mask's edge, negative inside, by a chamfer pass up to ``limit`` pixels."""
    mask = np.asarray(mask, bool)

    def distance(m):
        d = np.where(m, 0.0, float(limit))
        for _ in range(limit):
            p = np.pad(d, 1, constant_values=float(limit))
            near = np.minimum.reduce([p[1:-1, :-2] + 1, p[1:-1, 2:] + 1, p[:-2, 1:-1] + 1, p[2:, 1:-1] + 1,
                                      p[:-2, :-2] + 1.414, p[:-2, 2:] + 1.414,
                                      p[2:, :-2] + 1.414, p[2:, 2:] + 1.414])
            step = np.minimum(d, near)
            if np.array_equal(step, d):
                break
            d = step
        return d

    return distance(mask) - distance(~mask)


def hull_sdf(points, views) -> np.ndarray:
    """The visual hull's field at ``points``: the largest of the views' signed distances, in metres.

    Each view's mask is swept along its line of sight; a point is inside the hull when every view draws it.
    """
    points = np.asarray(points, float)
    field = np.full(len(points), -np.inf)
    for right, up, mask, frame in views:
        sdf = signed_distance(mask) if np.asarray(mask).dtype == bool else np.asarray(mask, float)
        h, w = sdf.shape
        u0, u1, v0, v1 = frame
        u, v = points @ np.asarray(right, float), points @ np.asarray(up, float)
        x = np.clip((u - u0) / (u1 - u0) * w - 0.5, 0, w - 1.001)
        y = np.clip((v1 - v) / (v1 - v0) * h - 0.5, 0, h - 1.001)
        c0, r0 = x.astype(int), y.astype(int)
        fx, fy = x - c0, y - r0
        d = ((sdf[r0, c0] * (1 - fx) + sdf[r0, c0 + 1] * fx) * (1 - fy)
             + (sdf[r0 + 1, c0] * (1 - fx) + sdf[r0 + 1, c0 + 1] * fx) * fy) * (u1 - u0) / w
        outside = (u < u0) | (u > u1) | (v < v0) | (v > v1)
        d[outside] = np.maximum(d[outside], 0.05 * (u1 - u0))
        field = np.maximum(field, d)
    return field


def surface_nets(field, origin, step) -> Mesh:
    """A closed mesh of the zero level of a sampled field (negative inside), one vertex per crossed cell.

    ``field`` is sampled on a regular grid of ``step`` metres whose first sample sits at ``origin``. Each cell
    the surface crosses gets one vertex, at the mean of its edges' zero crossings, and each grid edge the
    surface crosses joins the four cells around it with a quad (two triangles).
    """
    field = np.asarray(field, float)
    inside = field < 0
    nx, ny, nz = field.shape
    corners = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if sum(abs(np.subtract(corners[a], corners[b]))) == 1]
    values = np.stack([field[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k] for i, j, k in corners])
    below = values < 0
    active = below.any(0) & ~below.all(0)
    total = np.zeros(active.shape + (3,))
    count = np.zeros(active.shape)
    for a, b in edges:
        fa, fb = values[a], values[b]
        crossed = (fa < 0) != (fb < 0)
        t = np.where(crossed, fa / np.where(crossed, fa - fb, 1.0), 0.0)
        pa, pb = np.array(corners[a], float), np.array(corners[b], float)
        total += crossed[..., None] * (pa + t[..., None] * (pb - pa))
        count += crossed
    index = -np.ones(active.shape, np.int64)
    cells = np.argwhere(active)
    index[active] = np.arange(len(cells))
    positions = np.asarray(origin, float) + step * (cells + total[active] / count[active][:, None])
    quads = []
    for axis in range(3):
        a1, a2 = [x for x in range(3) if x != axis]
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        changed = inside[tuple(lo)] != inside[tuple(hi)]
        flip = inside[tuple(lo)][changed]
        at = np.argwhere(changed)
        keep = ((at[:, a1] > 0) & (at[:, a2] > 0)
                & (at[:, a1] < field.shape[a1] - 1) & (at[:, a2] < field.shape[a2] - 1))
        at, flip = at[keep], flip[keep]

        def cell(d1, d2):
            c = at.copy()
            c[:, a1] -= d1
            c[:, a2] -= d2
            return index[c[:, 0], c[:, 1], c[:, 2]]

        quad = np.stack([cell(0, 0), cell(1, 0), cell(1, 1), cell(0, 1)], 1)
        quad[flip] = quad[flip][:, ::-1]
        if axis == 1:                                       # the y axis's cyclic order runs the other way
            quad = quad[:, ::-1]
        quads.append(quad)
    quad = np.vstack(quads)
    quad = quad[(quad >= 0).all(1)]
    faces = np.vstack([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]])
    volume = np.einsum("ij,ij->i", positions[faces[:, 0]],
                       np.cross(positions[faces[:, 1]], positions[faces[:, 2]])).sum()
    if volume < 0:
        faces = faces[:, ::-1]
    return Mesh(positions=positions.astype(np.float32), faces=faces.astype(np.uint32)).recompute_normals()


def taubin(mesh: Mesh, iterations=10, lam=0.5, mu=-0.53) -> Mesh:
    """Smooth a mesh without shrinking it: a Laplacian step in, then a slightly larger one back, per iteration."""
    p = np.asarray(mesh.positions, float).copy()
    f = np.asarray(mesh.faces, np.int64)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    degree = np.bincount(e.ravel(), minlength=len(p)).astype(float)
    for _ in range(iterations):
        for weight in (lam, mu):
            total = np.zeros_like(p)
            np.add.at(total, e[:, 0], p[e[:, 1]])
            np.add.at(total, e[:, 1], p[e[:, 0]])
            p += weight * (total / np.maximum(degree, 1)[:, None] - p)
    return Mesh(positions=p.astype(np.float32), faces=mesh.faces).recompute_normals()


def _across(view, joints, bones, i):
    """How far the view's mask reaches on each side of joint ``i``, across its bones, in metres (or None)."""
    right, up, mask, (u0, u1, v0, v1) = view
    right, up = np.asarray(right, float), np.asarray(up, float)
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    directions = []
    for a, b in bones:
        if i in (a, b):
            d = joints[b if a == i else a] - joints[i]
            d2 = np.array([d @ right, d @ up])
            if np.linalg.norm(d2) > 1e-9:
                directions.append(d2 / np.linalg.norm(d2) * (1 if a == i else -1))
    if not directions:
        return None
    along = np.mean(directions, 0)
    if np.linalg.norm(along) < 1e-6:
        along = directions[0]
    along = along / np.linalg.norm(along)
    across = np.array([-along[1], along[0]])
    x = (joints[i] @ right - u0) / (u1 - u0) * w
    y = (v1 - joints[i] @ up) / (v1 - v0) * h
    reach = []
    for sign in (1, -1):
        n = 0
        while True:
            col, row = int(x + sign * across[0] * n), int(y - sign * across[1] * n)
            if not (0 <= row < h and 0 <= col < w) or not mask[row, col]:
                break
            n += 1
        reach.append(n)
    if min(reach) == 0:                                     # the joint is off the drawing in this view
        return None
    return float(np.mean(reach)) * (u1 - u0) / w


def fit_radii(joints, bones, views, default=0.03) -> np.ndarray:
    """Each joint's radius: the mean, over the views that draw it, of the drawing's half-width across its bones."""
    joints = np.asarray(joints, float)
    radii = []
    for i in range(len(joints)):
        widths = [w for w in (_across(view, joints, bones, i) for view in views) if w is not None]
        radii.append(float(np.mean(widths)) if widths else default)
    return np.array(radii)


def skin(joints, radii, bones, step=0.01, blend=0.04, hull=(), round=0.0, smooth=0) -> Mesh:
    """The mesh around a skeleton, optionally cut to the visual hull of ``hull`` views and smoothed.

    ``blend`` is how wide the joins between bones melt; ``round`` rounds the edges the hull cuts (a smooth
    maximum of that width); ``smooth`` is how many Taubin iterations the mesh gets. ``step`` is the grid's
    spacing in metres, the finest detail the skin keeps.
    """
    joints, radii = np.asarray(joints, float), np.asarray(radii, float)
    pad = float(radii.max()) + 2 * step + blend + round
    lo, hi = joints.min(0) - pad, joints.max(0) + pad
    shape = np.ceil((hi - lo) / step).astype(int) + 1
    axes = [lo[i] + step * np.arange(shape[i]) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    hull = [(right, up, signed_distance(mask), frame) for right, up, mask, frame in hull]
    chunks = []
    for points in np.array_split(grid, max(1, len(grid) // 200000)):
        field = skeleton_sdf(points, joints, radii, bones, blend)
        if hull:
            field = -_smin(-field, -hull_sdf(points, hull), round)
        chunks.append(field)
    mesh = surface_nets(np.concatenate(chunks).reshape(tuple(shape)), lo, step)
    return taubin(mesh, smooth) if smooth else mesh
