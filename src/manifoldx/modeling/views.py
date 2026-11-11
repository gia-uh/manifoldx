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

    Both masks are boolean arrays with the same rows (their widths may differ); row 0 is the top. For each
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
    if front.shape[0] != side.shape[0]:
        raise ValueError("front and side masks must have the same rows")
    height, width = front.shape
    side_width = side.shape[1]
    if height == 0 or width == 0 or side_width == 0:
        raise ValueError("the masks are empty")
    y_bottom, y_top = float(height_range[0]), float(height_range[1])
    x_min, x_max = float(x_range[0]), float(x_range[1])
    z_min, z_max = float(z_range[0]), float(z_range[1])
    if y_top == y_bottom:
        raise ValueError("height_range must have a non-zero height")

    def x_at(pixel):
        return x_min + (x_max - x_min) * pixel / width

    def z_at(pixel):
        return z_max + (z_min - z_max) * pixel / side_width

    # The rings go where the part is: over the rows drawn in both views, so a short part of a tall object
    # (a seat, a foot) gets all n of them, ending exactly at its first and last row.
    both = np.flatnonzero(front.any(axis=1) & side.any(axis=1))
    if both.size == 0:
        raise ValueError("front and side have no height where both are drawn")
    row_y = lambda r: y_top - (y_top - y_bottom) * (r + 0.5) / height  # noqa: E731
    rings, ys = [], []
    for y in np.linspace(row_y(both[0]), row_y(both[-1]), n):
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


def row_centres(mask) -> np.ndarray:
    """Each row's drawn span centre in pixel columns, NaN where the row is empty."""
    work = np.asarray(mask, dtype=bool)
    out = np.full(work.shape[0], np.nan)
    for row, drawn in enumerate(work):
        cols = np.flatnonzero(drawn)
        if cols.size:
            out[row] = (cols[0] + cols[-1] + 1) / 2.0
    return out


def fill_rows(mask, other, reach: int, mode: str = "narrow") -> np.ndarray:
    """A mask with the rows only the other view draws filled in, near a drawn row.

    A part seen from two views sometimes has a row drawn in one and not the other: a hidden end, or two
    blueprints that disagree on what the part is. Each row of ``mask`` that is empty but drawn in ``other``
    takes the span of ``mask``'s nearest drawn row, and ``mode`` says how:

    - ``"narrow"``: narrowed in proportion as ``other`` narrows between the two rows, so a sleeve that hangs
      lower in one view keeps its taper;
    - ``"copy"``: ``other``'s own span at that row.

    Rows further than ``reach`` (pixels) from a drawn row of ``mask`` are left alone.
    """
    if mode not in ("narrow", "copy"):
        raise ValueError("mode must be 'narrow' or 'copy'")
    work = np.asarray(mask, dtype=bool)
    other = np.asarray(other, dtype=bool)
    if work.shape[0] != other.shape[0]:
        raise ValueError("the two masks must have the same rows")

    def span(m, row):
        cols = np.flatnonzero(m[row])
        return (int(cols[0]), int(cols[-1]) + 1) if cols.size else None

    have = np.flatnonzero(work.any(axis=1))
    if have.size == 0:
        return work.copy()
    filled = work.copy()
    for row in np.flatnonzero(other.any(axis=1) & ~work.any(axis=1)):
        near = int(have[np.argmin(np.abs(have - row))])
        if abs(near - row) > reach:
            continue
        here, there = span(work, near), span(other, row)
        if there is None:
            continue
        if mode == "copy" or here is None:
            filled[row, there[0]:there[1]] = True
            continue
        before = span(other, near) or there
        ratio = (there[1] - there[0]) / max(1, before[1] - before[0])
        centre = (here[0] + here[1]) / 2.0
        half = (here[1] - here[0]) / 2.0 * min(1.0, ratio)
        filled[row, int(round(centre - half)):int(round(centre + half)) + 1] = True
    return filled


def lathe_solid(front, height_range, x_range, centre=None, samples=30, segments=16):
    """A solid of revolution from one view's profile, turned about a vertical axis.

    ``front`` is a boolean mask with row 0 at the top, and ``height_range=(y_bottom, y_top)`` and
    ``x_range=(x_min, x_max)`` say what metres the mask covers, as for :func:`two_view_solid`. The profile is
    the mask's outer edge on either side of ``centre`` (an x in metres; the mask's own span centre by default),
    so a rectangle becomes a closed cylinder and a vase comes out round. ``samples`` rings are spread over the
    rows the mask draws; consecutive rings of the same radius are merged, and flat caps close the ends.
    """
    front = np.asarray(front, dtype=bool)
    if front.ndim != 2 or not front.any():
        raise ValueError("lathe_solid needs a drawn mask")
    samples, segments = int(samples), int(segments)
    if samples < 2 or segments < 3:
        raise ValueError("lathe_solid needs samples >= 2 and segments >= 3")
    height, width = front.shape
    y_bottom, y_top = float(height_range[0]), float(height_range[1])
    x_min, x_max = float(x_range[0]), float(x_range[1])
    if y_top == y_bottom:
        raise ValueError("height_range must have a non-zero height")
    x_at = lambda pixel: x_min + (x_max - x_min) * pixel / width  # noqa: E731

    drawn = np.flatnonzero(front.any(axis=1))
    if drawn.size < 1:
        raise ValueError("lathe_solid needs a drawn mask")
    cols = np.flatnonzero(front.any(axis=0))
    if centre is None:
        centre = x_at((cols[0] + cols[-1] + 1) / 2.0)
    centre = float(centre)
    row_y = lambda r: y_top - (y_top - y_bottom) * (r + 0.5) / height  # noqa: E731

    raw = []
    for y in np.linspace(row_y(drawn[0]), row_y(drawn[-1]), samples):
        row = int(np.clip(np.floor((y_top - y) / (y_top - y_bottom) * height), 0, height - 1))
        spans = np.flatnonzero(front[row])
        if spans.size == 0:
            continue
        lo, hi = sorted((x_at(spans[0]), x_at(spans[-1] + 1)))
        radius = max(hi - centre, centre - lo)
        if radius > 1e-12:
            raw.append((y, radius))
    # A run of equal radii keeps both ends, so a cylinder gets its bottom and its top rather than one ring.
    rings = []
    i = 0
    while i < len(raw):
        j = i
        while j + 1 < len(raw) and abs(raw[j + 1][1] - raw[i][1]) <= 1e-12:
            j += 1
        rings.append(raw[i])
        if j > i:
            rings.append(raw[j])
        i = j + 1
    if len(rings) < 2:
        raise ValueError("lathe_solid needs a profile with a height")

    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    positions = []
    for y, radius in rings:
        positions.append(np.column_stack([centre + radius * np.cos(theta), np.full(segments, y),
                                          radius * np.sin(theta)]))
    positions = np.vstack(positions)
    n_ring = len(positions)
    bottom = np.array([[centre, rings[0][0], 0.0]])
    top = np.array([[centre, rings[-1][0], 0.0]])
    positions = np.vstack([positions, bottom, top]).astype(np.float32)
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
    faces = np.asarray(faces, dtype=np.int64)
    p = positions
    volume = np.einsum("ij,ij->i", p[faces[:, 0]], np.cross(p[faces[:, 1]], p[faces[:, 2]])).sum()
    if volume < 0:
        faces = faces[:, ::-1]
    return Mesh(positions=positions, faces=faces.astype(np.uint32)).recompute_normals()


def reflect_x(mesh: Mesh) -> Mesh:
    """A mesh mirrored across the plane x = 0, with its faces turned so they still face outwards."""
    positions = np.asarray(mesh.positions, float).copy()
    positions[:, 0] = -positions[:, 0]
    uvs = None if mesh.uvs is None else np.column_stack([1.0 - np.asarray(mesh.uvs, float)[:, 0],
                                                         np.asarray(mesh.uvs, float)[:, 1]])
    return Mesh(positions=positions.astype(np.float32), faces=np.asarray(mesh.faces)[:, ::-1].copy(),
                uvs=None if uvs is None else uvs.astype(np.float32)).recompute_normals()


def _centreline(points, samples, anchor):
    """Centre and half-width of an elongated point set at `samples` steps along its main axis.

    The axis runs from the end nearest `anchor` (when given) or from the top end.
    """
    mean = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - mean, full_matrices=False)
    d = vt[0]
    t = (points - mean) @ d
    ends = mean + np.outer([t.min(), t.max()], d)
    if anchor is not None:
        flip = np.linalg.norm(ends[1] - anchor) < np.linalg.norm(ends[0] - anchor)
    else:
        flip = ends[1][1] > ends[0][1]
    if flip:
        d, t = -d, -t
    perp = (points - mean) @ np.array([-d[1], d[0]])
    edges = np.linspace(t.min(), t.max(), samples + 1)
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, samples - 1)
    centre = np.full((samples, 2), np.nan)
    half = np.full(samples, np.nan)
    for k in range(samples):
        sel = idx == k
        if sel.any():
            centre[k] = points[sel].mean(axis=0)
            half[k] = (perp[sel].max() - perp[sel].min()) / 2.0
    good = ~np.isnan(half)
    ks = np.arange(samples)
    for j in range(2):
        centre[:, j] = np.interp(ks, ks[good], centre[good, j])
    half = np.interp(ks, ks[good], half[good])
    return centre, half


def swept_tube(front, side, height_range, x_range=(-1.0, 1.0), z_range=(-1.0, 1.0), samples=32,
               segments=16, anchor=None, radius="min"):
    """A tube swept along a part's centreline, matched between a front and a left-side mask.

    For a limb that reaches across and forward at once (an outstretched arm, an armrest on its post, an arch),
    where :func:`two_view_solid` fills the whole rectangle of the two spans. The masks and ranges follow
    :func:`two_view_solid`. Each view's centreline is the mask's mean point in `samples` steps along its main
    axis; the two are matched by their position along that axis (from the end nearest `anchor`, an ``(x, y, z)``
    point such as the body's centre, or from the top end), so the tube's point i takes x from the front view, z
    from the side view and y from both. ``radius`` is the two views' smaller half-width there (``"min"``), their
    mean (``"mean"``), or a fixed number of metres — a hanging sleeve or a cloth widens one view or both.
    """
    front = np.asarray(front, dtype=bool)
    side = np.asarray(side, dtype=bool)
    if front.shape[0] != side.shape[0]:
        raise ValueError("front and side masks must have the same rows")
    if not front.any() or not side.any():
        raise ValueError("swept_tube needs the part drawn in both views")
    height = front.shape[0]
    y_bottom, y_top = float(height_range[0]), float(height_range[1])
    x_min, x_max = float(x_range[0]), float(x_range[1])
    z_min, z_max = float(z_range[0]), float(z_range[1])
    rows, cols = np.nonzero(front)
    ys = y_top - (y_top - y_bottom) * (rows + 0.5) / height
    fpts = np.column_stack([x_min + (x_max - x_min) * (cols + 0.5) / front.shape[1], ys])
    rows, cols = np.nonzero(side)
    ys = y_top - (y_top - y_bottom) * (rows + 0.5) / height
    spts = np.column_stack([z_max + (z_min - z_max) * (cols + 0.5) / side.shape[1], ys])
    fa = None if anchor is None else np.array([anchor[0], anchor[1]], float)
    sa = None if anchor is None else np.array([anchor[2], anchor[1]], float)
    fc, fh = _centreline(fpts, samples, fa)
    sc, sh = _centreline(spts, samples, sa)
    # The anchor can sit halfway along the part in one view (a seated figure's body centre, level with the
    # middle of an outstretched arm seen from the side), so the side's centreline may run hand to shoulder
    # while the front's runs shoulder to hand. Both views share the height and see the same thickness at the
    # same point of the part: run the side the way round where those two agree.
    def disagreement(c, h):
        return np.abs(fc[:, 1] - c[:, 1]).mean() + np.abs(fh - h).mean()

    if disagreement(sc[::-1], sh[::-1]) < 0.8 * disagreement(sc, sh):
        sc, sh = sc[::-1], sh[::-1]
    path =np.column_stack([fc[:, 0], (fc[:, 1] + sc[:, 1]) / 2.0, sc[:, 0]])
    if isinstance(radius, str):
        if radius == "min":
            width = np.minimum(fh, sh)
        elif radius == "mean":
            width = (fh + sh) / 2.0
        else:
            raise ValueError("radius must be 'min', 'mean' or a positive number")
    else:
        width = np.full(samples, float(radius))
        if width[0] <= 0.0:
            raise ValueError("radius must be positive")
    kernel = np.ones(3) / 3.0                       # a light smoothing of the path and the width
    for j in range(3):
        path[1:-1, j] = np.convolve(path[:, j], kernel, mode="valid")
    width[1:-1] = np.convolve(width, kernel, mode="valid")

    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    ref = np.array([0.0, 1.0, 0.0]) if abs(tangent[0][1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    normal = np.cross(tangent[0], ref)
    normal /= np.linalg.norm(normal)
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    rings = []
    for i in range(samples):                        # parallel transport keeps the rings from twisting
        normal = normal - tangent[i] * (normal @ tangent[i])
        normal /= max(np.linalg.norm(normal), 1e-12)
        binormal = np.cross(tangent[i], normal)
        rings.append(path[i] + width[i] * (np.outer(np.cos(theta), normal) + np.outer(np.sin(theta), binormal)))
    positions = np.vstack(rings + [path[:1], path[-1:]])
    first, last = samples * segments, samples * segments + 1
    faces = []
    for i in range(samples - 1):
        for k in range(segments):
            a, b = i * segments + k, i * segments + (k + 1) % segments
            faces += [(a, b, b + segments), (a, b + segments, a + segments)]
    for k in range(segments):
        faces.append((first, (k + 1) % segments, k))
        faces.append((last, (samples - 1) * segments + k, (samples - 1) * segments + (k + 1) % segments))
    faces = np.asarray(faces, dtype=np.int64)
    p = positions
    volume = np.einsum("ij,ij->i", p[faces[:, 0]], np.cross(p[faces[:, 1]], p[faces[:, 2]])).sum()
    if volume < 0:                                  # faces outward, whatever way the path runs
        faces = faces[:, ::-1]
    return Mesh(positions=positions.astype(np.float32), faces=faces.astype(np.uint32)).recompute_normals()


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


def template(parts, atlas, colors, island, whole=False):
    """Draw one island's faces in flat colours, nearest face winning.

    ``colors`` maps a part (or material) name to RGB and its order gives the
    region index: the first name is region 1, the second 2, and 0 is the white
    background. ``island`` is an island index, an :class:`Island`, or a view
    name. Returns ``(img, regions)``: an RGB image of the atlas (uint8) and an
    int map of the material index, both with the atlas's size.

    With ``whole=True`` every face in front of the camera is drawn, not only
    the island's own faces, so the template is a full picture of the object in
    that view. A rounded object needs it: its island alone is a ragged band
    that an image model cannot read as a picture, while every face still takes
    its colour from the one view that sees it best.
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
        take = np.ones(len(mesh.faces), bool) if whole else atlas.island_of[name] == index
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
