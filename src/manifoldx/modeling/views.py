"""Solids and texture atlases built from orthographic views.

Axes are the engine's: x right, y up, z toward the viewer of the front view.
A view is named for the side of the object it shows: ``"front"`` looks along
``-z``, ``"back"`` along ``+z``, ``"left"`` (the object's left side) along
``+x``, ``"right"`` along ``-x``, ``"top"`` along ``-y`` and ``"bottom"``
along ``+y``. Images are numpy arrays with row 0 at the top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manifoldx.modeling._poly import _triangulate
from manifoldx.modeling.mesh import Mesh

# name -> (toward the viewer, image right, image up), all in object axes.
_VIEWS = {
    "front": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "back": ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "left": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "right": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "top": ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    "bottom": ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
}
_VIEW_NAMES = tuple(_VIEWS)


@dataclass
class Island:
    """One view's rectangle in an atlas."""

    view: str                               # the view's name
    rect: tuple[int, int, int, int]         # x0, y0, x1, y1 in atlas pixels (y down)
    frame: tuple[float, float, float, float]  # u_min, u_max, v_min, v_max in metres


@dataclass
class Atlas:
    """A square texture atlas with one island per view. Holds a UV per mesh vertex."""

    size: tuple[int, int]
    islands: list[Island]
    uvs: dict[str, np.ndarray]              # part name -> float32 (N, 2)
    meshes: dict[str, Mesh]                 # part name -> mesh split by island, with uvs
    island_of: dict[str, np.ndarray]        # part name -> int (M,) island per face
    texel: float                            # nominal metres per pixel


def extrude(polygon, depth, centre=0.0):
    """A closed prism from a 2-D outline in the XY plane, extruded along z.

    ``depth`` is the length along z and ``centre`` the z of its middle, so the
    solid spans ``centre - depth / 2`` to ``centre + depth / 2``. The outline may
    be concave; faces are counter-clockwise seen from outside.
    """
    q = np.asarray(polygon, dtype=float)[:, :2]
    if len(q) < 3:
        raise ValueError("extrude needs at least three outline points")
    area = np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
    if area < 0:
        q = q[::-1]
    n = len(q)
    lo = float(centre) - float(depth) / 2.0
    hi = float(centre) + float(depth) / 2.0
    positions = np.vstack([
        np.column_stack([q, np.full(n, hi)]),
        np.column_stack([q, np.full(n, lo)]),
    ]).astype(np.float32)

    tris = _triangulate(q)
    faces = [tri for tri in tris]                                  # front face at +z
    faces += [(n + c, n + b, n + a) for a, b, c in tris]           # back face at -z
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, n + j, j), (i, n + i, n + j)]                # wall, outward
    return Mesh(positions=positions, faces=np.asarray(faces, dtype=np.uint32)).recompute_normals()


def two_view_solid(front, side, height_range, n=48, power=2.0, segments=24,
                   x_range=(-1.0, 1.0), z_range=(-1.0, 1.0)):
    """A solid lofted from a front mask and a left-side mask.

    Both masks are boolean arrays with the same rows; row 0 is the top. For each
    of ``n`` heights the front mask gives the x interval and the side mask the z
    interval, and the section is a superellipse of exponent ``power`` (2 is an
    ellipse) inscribed in that rectangle. ``height_range=(y_bottom, y_top)``,
    ``x_range=(x_min, x_max)`` and ``z_range=(z_min, z_max)`` give the metres the
    masks cover: the front's first column is ``x_min``, the side's first column
    is ``z_max`` (the object faces the front-view camera, on the image's left)
    and its last column is ``z_min``. A row's span is the first to the last
    drawn pixel, so a hole left by a part drawn on top does not split the
    section. Heights where either mask is empty are skipped.
    """
    n, segments = int(n), int(segments)
    if n < 2 or segments < 3:
        raise ValueError("two_view_solid needs n >= 2 and segments >= 3")
    if float(power) <= 0.0:
        raise ValueError("power must be positive")
    front = np.asarray(front, dtype=bool)
    side = np.asarray(side, dtype=bool)
    if front.shape != side.shape:
        raise ValueError("front and side masks must have the same shape")
    height, width = front.shape
    if height == 0 or width == 0:
        raise ValueError("the masks are empty")
    y_bottom, y_top = float(height_range[0]), float(height_range[1])
    x_min, x_max = float(x_range[0]), float(x_range[1])
    z_min, z_max = float(z_range[0]), float(z_range[1])
    if y_top == y_bottom:
        raise ValueError("height_range must have a non-zero height")

    def x_at(pixel):
        return x_min + (x_max - x_min) * pixel / width

    def z_at(pixel):
        return z_max + (z_min - z_max) * pixel / width

    rings, ys = [], []
    for y in np.linspace(y_top, y_bottom, n):
        row = int(np.clip(np.floor((y_top - y) / (y_top - y_bottom) * height), 0, height - 1))
        cols = np.flatnonzero(front[row])
        zcols = np.flatnonzero(side[row])
        if cols.size == 0 or zcols.size == 0:
            continue
        x_lo, x_hi = sorted((x_at(cols[0]), x_at(cols[-1] + 1)))
        z_lo, z_hi = sorted((z_at(zcols[0]), z_at(zcols[-1] + 1)))
        if x_hi - x_lo <= 1e-12 or z_hi - z_lo <= 1e-12:
            continue
        if ys and abs(y - ys[-1]) < 1e-12:
            continue
        ys.append(y)
        rings.append((y, (x_lo + x_hi) / 2.0, (x_hi - x_lo) / 2.0,
                      (z_lo + z_hi) / 2.0, (z_hi - z_lo) / 2.0))

    if len(rings) < 2:
        raise ValueError("front and side have no height where both are drawn")
    rings.sort(key=lambda ring: ring[0])  # bottom first, so the side walls face outwards

    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    exponent = 2.0 / float(power)
    sx = np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** exponent
    sz = np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** exponent

    positions = []
    for y, cx, a, cz, b in rings:
        positions.append(np.column_stack([cx + a * sx, np.full(segments, y), cz + b * sz]))
    positions = np.vstack(positions)
    lower_c = np.array([[rings[0][1], rings[0][0], rings[0][3]]])
    upper_c = np.array([[rings[-1][1], rings[-1][0], rings[-1][3]]])
    n_ring = len(positions)
    positions = np.vstack([positions, lower_c, upper_c]).astype(np.float32)
    bottom_centre, top_centre = n_ring, n_ring + 1

    faces = []
    for s in range(len(rings) - 1):
        base, nxt = s * segments, (s + 1) * segments
        for i in range(segments):
            j = (i + 1) % segments
            faces += [(base + i, nxt + i, nxt + j), (base + i, nxt + j, base + j)]
    last = (len(rings) - 1) * segments
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((bottom_centre, i, j))
        faces.append((top_centre, last + j, last + i))
    return Mesh(positions=positions, faces=np.asarray(faces, dtype=np.uint32)).recompute_normals()


def _coverage(px, py, clip):
    """Pixel centres inside one projected triangle, with barycentric weights.

    ``px`` and ``py`` are the triangle's corners in continuous pixel
    coordinates (pixel ``i`` covers ``[i, i + 1)``). ``clip`` is the half-open
    pixel rectangle ``(x0, y0, x1, y1)`` to search. Returns
    ``(row0, col0, l0, l1, l2, inside, order)``, where ``order`` maps the three
    weights to the triangle's original corners, or None when nothing is hit.
    """
    x0, y0, x1, y1 = clip
    order = np.array([0, 1, 2])
    den = (py[1] - py[2]) * (px[0] - px[2]) + (px[2] - px[1]) * (py[0] - py[2])
    if abs(den) < 1e-12:
        return None
    if den < 0.0:
        order = np.array([0, 2, 1])
        px, py = px[order], py[order]
        den = -den

    col0 = max(x0, int(np.floor(px.min() - 0.5)))
    col1 = min(x1 - 1, int(np.ceil(px.max() - 0.5)))
    row0 = max(y0, int(np.floor(py.min() - 0.5)))
    row1 = min(y1 - 1, int(np.ceil(py.max() - 0.5)))
    if col0 > col1 or row0 > row1:
        return None

    xs = np.arange(col0, col1 + 1, dtype=float) + 0.5
    ys = (np.arange(row0, row1 + 1, dtype=float) + 0.5)[:, None]
    inv = 1.0 / den
    l0 = ((py[1] - py[2]) * (xs - px[2]) + (px[2] - px[1]) * (ys - py[2])) * inv
    l1 = ((py[2] - py[0]) * (xs - px[2]) + (px[0] - px[2]) * (ys - py[2])) * inv
    l2 = 1.0 - l0 - l1
    inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
    if not inside.any():
        return None
    return row0, col0, l0, l1, l2, inside, order


def _project(pos, view):
    _, right, up = _VIEWS[view]
    return pos @ np.asarray(right), pos @ np.asarray(up)


def silhouette(mesh, view, size, frame):
    """The mask of a mesh's outline in one view.

    ``size`` is ``(width, height)`` in pixels and ``frame`` is
    ``(u_min, u_max, v_min, v_max)``, the metres the image covers, with v up.
    """
    width, height = int(size[0]), int(size[1])
    u_min, u_max, v_min, v_max = (float(v) for v in frame)
    if width <= 0 or height <= 0 or u_min == u_max or v_min == v_max:
        raise ValueError("silhouette needs a positive size and a non-zero frame")
    u, v = _project(mesh.positions.astype(float), view)
    px = (u - u_min) / (u_max - u_min) * width
    py = (v_max - v) / (v_max - v_min) * height
    mask = np.zeros((height, width), dtype=bool)
    for tri in mesh.faces:
        hit = _coverage(px[tri], py[tri], (0, 0, width, height))
        if hit is None:
            continue
        row0, col0, _, _, _, inside, _ = hit
        rows, cols = inside.shape
        mask[row0:row0 + rows, col0:col0 + cols] |= inside
    return mask


def _as_parts(parts):
    if isinstance(parts, Mesh):
        return {"part": parts}
    if hasattr(parts, "items"):
        return dict(parts)
    out = {}
    for i, item in enumerate(parts):
        if isinstance(item, Mesh):
            out[f"part{i}"] = item
        else:
            name, mesh = item
            out[str(name)] = mesh
    return out


def _face_normals(mesh):
    p = mesh.positions.astype(float)
    f = mesh.faces.astype(np.intp)
    v0, v1, v2 = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, length, out=np.tile([0.0, 0.0, 1.0], (len(n), 1)), where=length > 1e-12)


def view_atlas(parts, views, size, texel=None, margin=8):
    """An atlas with one island per view; each face goes to its closest view.

    ``parts`` maps a name to a :class:`Mesh`. A face is sent to the view whose
    direction has the largest dot product with the face normal, and its UV is
    the orthographic projection in that view. The views are laid out in a grid
    in a square ``size=(width, height)`` atlas, with ``margin`` pixels around
    each island. All islands share one metres-per-pixel scale, so texel density
    is even; pass ``texel`` (metres per pixel) to force it. Returns an
    :class:`Atlas` whose ``meshes`` have duplicated vertices where a vertex is
    shared by faces of different islands, so each vertex has one UV.
    """
    parts = _as_parts(parts)
    if not parts:
        raise ValueError("view_atlas needs at least one part")
    views = list(views)
    for name in views:
        if name not in _VIEWS:
            raise ValueError(f"unknown view {name!r}; choose from {', '.join(_VIEW_NAMES)}")
    if not views:
        raise ValueError("view_atlas needs at least one view")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("atlas size must be positive")

    normals = np.array([_VIEWS[name][0] for name in views], dtype=float)
    rights = np.array([_VIEWS[name][1] for name in views], dtype=float)
    ups = np.array([_VIEWS[name][2] for name in views], dtype=float)

    bounds = []
    for name in views:
        us, vs = [], []
        for mesh in parts.values():
            p = mesh.positions.astype(float)
            us.append(p @ rights[len(bounds)])
            vs.append(p @ ups[len(bounds)])
        u, v = np.concatenate(us), np.concatenate(vs)
        bounds.append((float(u.min()), float(u.max()), float(v.min()), float(v.max())))

    count = len(views)
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    cell_w, cell_h = width / cols, height / rows
    inner_w, inner_h = cell_w - 2 * margin, cell_h - 2 * margin
    if inner_w <= 1 or inner_h <= 1:
        raise ValueError("atlas size leaves no room for the islands at this margin")

    extents = []
    for u0, u1, v0, v1 in bounds:
        du, dv = u1 - u0, v1 - v0
        extents.append((max(du, 1e-9), max(dv, 1e-9)))
    if texel is None:
        scale = min(min(inner_w / du, inner_h / dv) for du, dv in extents)
    else:
        scale = 1.0 / float(texel)
        if scale <= 0:
            raise ValueError("texel must be positive")
        need_w = max(du for du, _ in extents) * scale + 2 * margin
        need_h = max(dv for _, dv in extents) * scale + 2 * margin
        if cols * need_w > width + 1e-9 or rows * need_h > height + 1e-9:
            raise ValueError("atlas size is too small for that texel; use a larger size or texel")

    islands = []
    for k, name in enumerate(views):
        u0, u1, v0, v1 = bounds[k]
        du, dv = u1 - u0, v1 - v0
        if du < 1e-9:
            u0, u1 = u0 - 0.5e-6, u0 + 0.5e-6
        if dv < 1e-9:
            v0, v1 = v0 - 0.5e-6, v0 + 0.5e-6
        pw = max(1, int(np.floor((u1 - u0) * scale)))
        ph = max(1, int(np.floor((v1 - v0) * scale)))
        cell_x = (k % cols) * cell_w
        cell_y = (k // cols) * cell_h
        x0 = int(round(cell_x + (cell_w - pw) / 2.0))
        y0 = int(round(cell_y + (cell_h - ph) / 2.0))
        islands.append(Island(view=name, rect=(x0, y0, x0 + pw, y0 + ph),
                              frame=(u0, u1, v0, v1)))

    uvs, meshes, island_of = {}, {}, {}
    for name, mesh in parts.items():
        p = mesh.positions.astype(float)
        f = mesh.faces.astype(np.intp)
        face_island = np.argmax(_face_normals(mesh) @ normals.T, axis=1) if len(f) else np.zeros(0, int)
        island_of[name] = face_island.astype(np.int32)
        if len(f) == 0:
            uvs[name] = np.zeros((0, 2), dtype=np.float32)
            meshes[name] = Mesh(positions=mesh.positions.astype(np.float32),
                                faces=mesh.faces.astype(np.uint32),
                                uvs=np.zeros((len(mesh.positions), 2), dtype=np.float32))
            continue
        corner_vertex = f.reshape(-1)
        corner_island = np.repeat(face_island, 3)
        keys = corner_vertex * count + corner_island
        unique, inverse = np.unique(keys, return_inverse=True)
        vertex = unique // count
        island = unique % count
        new_pos = p[vertex]
        u = np.einsum("ij,ij->i", new_pos, rights[island])
        v = np.einsum("ij,ij->i", new_pos, ups[island])
        px = np.empty(len(vertex))
        py = np.empty(len(vertex))
        for k, isl in enumerate(islands):
            sel = island == k
            if not sel.any():
                continue
            u0, u1, v0, v1 = isl.frame
            x0, y0, x1, y1 = isl.rect
            px[sel] = x0 + (u[sel] - u0) / (u1 - u0) * (x1 - x0)
            py[sel] = y0 + (v1 - v[sel]) / (v1 - v0) * (y1 - y0)
        uv = np.clip(np.stack([px / width, py / height], axis=1), 0.0, 1.0).astype(np.float32)
        faces = inverse.reshape(-1, 3).astype(np.uint32)
        uvs[name] = uv
        meshes[name] = Mesh(positions=new_pos.astype(np.float32), faces=faces, uvs=uv)

    return Atlas(size=(width, height), islands=islands, uvs=uvs, meshes=meshes,
                 island_of=island_of, texel=1.0 / scale)


def template(parts, atlas, colors, island):
    """Draw one island's faces in flat colours, nearest face winning.

    ``colors`` maps a part (or material) name to RGB and its order gives the
    region index: the first name is region 1, the second 2, and 0 is the white
    background. ``island`` is an island index, an :class:`Island`, or a view
    name. Returns ``(img, regions)``: an RGB image of the atlas (uint8) and an
    int map of the material index, both with the atlas's size.
    """
    if isinstance(island, str):
        names = [item.view for item in atlas.islands]
        if island not in names:
            raise ValueError(f"the atlas has no island for view {island!r}")
        index = names.index(island)
    elif isinstance(island, Island):
        index = atlas.islands.index(island)
    else:
        index = int(island)
    isl = atlas.islands[index]
    width, height = atlas.size
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    regions = np.zeros((height, width), dtype=np.int32)
    depth = np.full((height, width), -np.inf)
    u0, u1, v0, v1 = isl.frame
    x0, y0, x1, y1 = isl.rect
    _, right, up = _VIEWS[isl.view]
    normal = np.asarray(_VIEWS[isl.view][0])
    right, up = np.asarray(right), np.asarray(up)
    colour_names = list(colors)
    parts = _as_parts(parts)

    for name, mesh in parts.items():
        if name not in atlas.island_of:
            raise KeyError(f"the atlas has no part {name!r}")
        if name not in colors:
            raise KeyError(f"colors has no entry for part {name!r}")
        take = atlas.island_of[name] == index
        if not take.any():
            continue
        rgb = np.asarray(colors[name], dtype=np.uint8).reshape(3)
        region = colour_names.index(name) + 1
        pos = mesh.positions.astype(float)
        u, v, z = pos @ right, pos @ up, pos @ normal
        px = x0 + (u - u0) / (u1 - u0) * (x1 - x0)
        py = y0 + (v1 - v) / (v1 - v0) * (y1 - y0)
        for tri in mesh.faces[take].astype(np.intp):
            hit = _coverage(px[tri], py[tri], (x0, y0, x1, y1))
            if hit is None:
                continue
            row0, col0, l0, l1, l2, inside, order = hit
            rows, cols = inside.shape
            sub_depth = depth[row0:row0 + rows, col0:col0 + cols]
            zz = z[tri[order]]
            face_z = l0 * zz[0] + l1 * zz[1] + l2 * zz[2]
            hit_mask = inside & (face_z > sub_depth)
            if not hit_mask.any():
                continue
            sub_depth[hit_mask] = face_z[hit_mask]
            img[row0:row0 + rows, col0:col0 + cols][hit_mask] = rgb
            regions[row0:row0 + rows, col0:col0 + cols][hit_mask] = region

    return img, regions
