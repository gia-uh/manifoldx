"""Bake an atlas by projecting orthographic views onto a built asset.

Every UV-mapped piece of the asset is rasterised into the atlas: each texel gets
the three-dimensional point, the normal and the material it belongs to.  Each
view is registered by fitting the asset's projected silhouette box to the
bounding box of the image's non-background pixels.  Every texel is then
projected orthographically into every view, sampled bilinearly, and blended by
how squarely its normal faces that view.  Only the materials the caller names
are baked; the rest keep their palette colours, and texels no view faces keep a
fallback colour.  Last, the atlas gutters are filled from the nearest covered
texel so that mip-mapping does not bleed one island into another.

The module uses numpy only; it never imports Pillow.  A view may be a Pillow
image, an array read by another library, or a plain ``H x W x 3`` numpy array,
because every view goes through :func:`numpy.asarray`.

Views are keyed ``left``, ``right``, ``front``, ``rear``, ``top`` and ``bottom``
and must be orthographic, with a plain background:

* ``left`` has the asset's front at the image's left;
* ``right`` is the same drawing mirrored (the left view is mirrored when
  ``right`` is missing, and vice versa);
* ``front`` has the asset's left side at the image's right;
* ``rear`` has the asset's right side at the image's right;
* ``top`` has the front at the left and the asset's right side up;
* ``bottom`` has the front at the left and the asset's right side down.

The asset is assumed to use metres, Y up, its front at +X and its right at +Z,
which is the convention of :class:`manifoldx.modeling.assets.car.Car`.
"""

from __future__ import annotations

import numpy as np

#: The car materials a bake touches by default.  Glass, lamps, tyres, wheels and
#: the interior keep their palette colours unless the caller asks for them.
BAKE_MATERIALS = frozenset({"paint", "black", "trim", "chrome", "plate", "underbody"})

# The direction from the asset toward the viewer, in world space.
_VIEW_DIRECTIONS = {
    "left": np.array([0.0, 0.0, -1.0]),
    "right": np.array([0.0, 0.0, 1.0]),
    "front": np.array([1.0, 0.0, 0.0]),
    "rear": np.array([-1.0, 0.0, 0.0]),
    "top": np.array([0.0, 1.0, 0.0]),
    "bottom": np.array([0.0, -1.0, 0.0]),
}
_VIEW_NAMES = tuple(_VIEW_DIRECTIONS)

_WEIGHT_POWER = 3.0
# A texel counts as baked when the views that face it add up to this much.
_COVERAGE = 0.05
# How far the nearest-covered-texel fill reaches into the empty atlas.  This is
# enough for the mip levels a 2K atlas uses without colouring the whole sheet.
_GUTTER = 16
_BACKGROUND_TOLERANCE = 24.0


def bake_projection(car, views, size=2048, materials=None, fallback=None):
    """Texture a built asset's atlas from orthographic drawings.

    Parameters
    ----------
    car:
        A built asset: anything with ``parts``, where each part has ``pieces``
        mapping a material name to a :class:`~manifoldx.modeling.mesh.Mesh`
        with ``uvs``.  A :class:`~manifoldx.modeling.assets.car.Car` is the
        intended input.
    views:
        ``{name: image}`` for any of ``left``, ``right``, ``front``, ``rear``,
        ``top`` and ``bottom``.  A missing ``right`` mirrors ``left`` and a
        missing ``left`` mirrors ``right``.  An image is anything
        :func:`numpy.asarray` can read as ``H x W``, ``H x W x 3`` or
        ``H x W x 4``; an alpha channel is used as the foreground mask.
    size:
        Side of the square atlas, in texels.
    materials:
        Names of the materials to bake.  Defaults to :data:`BAKE_MATERIALS`.
        Everything else keeps its palette colour.
    fallback:
        Colour for baked texels no view faces: an ``(3,)`` RGB triple, or a
        ``{material: RGB}`` mapping.  When omitted, a material with baked
        texels keeps the median of what was baked and one with none keeps its
        palette colour.

    Returns
    -------
    ``(size, size, 3)`` uint8 RGB array, laid out like the asset's atlas.
    """
    size = int(size)
    if size < 1:
        raise ValueError("size must be a positive integer")
    if not isinstance(views, dict) or not views:
        raise ValueError("views must be a non-empty mapping of view name to image")

    images = {name: _prepare_image(image) for name, image in views.items() if name in _VIEW_DIRECTIONS}
    unknown = set(views) - set(_VIEW_DIRECTIONS)
    if unknown:
        raise ValueError(f"unknown view(s) {sorted(unknown)}; expected {sorted(_VIEW_NAMES)}")
    if "right" not in images and "left" in images:
        images["right"] = _mirror(images["left"])
    if "left" not in images and "right" in images:
        images["left"] = _mirror(images["right"])
    if not images:
        raise ValueError("no usable views")

    names = _material_names(car)
    if not names:
        raise ValueError("the asset has no UV-mapped pieces to bake")
    id_of = {name: i for i, name in enumerate(names)}
    baked_names = BAKE_MATERIALS if materials is None else materials
    if isinstance(baked_names, str):
        baked_names = (baked_names,)
    baked_ids = [id_of[name] for name in baked_names if name in id_of]

    pos, nrm, mat = _rasterise(car, size, names)
    baked = np.isin(mat, baked_ids)

    xmin, xmax, ymin, ymax, zmin, zmax = _bounds(car)
    acc = np.zeros((size, size, 3), np.float32)
    wsum = np.zeros((size, size), np.float32)

    for name, (rgb, mask) in images.items():
        weight = np.clip(nrm @ _VIEW_DIRECTIONS[name], 0.0, 1.0) ** _WEIGHT_POWER
        weight *= baked
        if not weight.any():
            continue
        xf, yf = _project(name, pos, _view_box(images[name]), xmin, xmax, ymin, ymax, zmin, zmax)
        covered = _bilinear(mask.astype(np.float32), xf, yf)
        weight *= covered
        if not weight.any():
            continue
        acc += _bilinear(rgb, xf, yf) * weight[..., None]
        wsum += weight

    atlas = _palette_atlas(mat, names)
    got = wsum > _COVERAGE
    palette = _palette()

    for index, name in enumerate(names):
        if index not in baked_ids:
            continue
        sel = mat == index
        if not sel.any():
            continue
        coloured = sel & got
        if coloured.any():
            base = np.median(acc[coloured] / wsum[coloured][:, None], axis=0)
        else:
            base = palette.get(name, np.array([128.0, 128.0, 128.0], np.float32))
        explicit = _fallback_for(fallback, name)
        atlas[sel] = base if explicit is None else explicit

    for index in baked_ids:
        sel = (mat == index) & got
        if sel.any():
            atlas[sel] = acc[sel] / wsum[sel][:, None]

    atlas = _fill_gutters(atlas, mat >= 0, _GUTTER)
    return np.clip(atlas, 0.0, 255.0).astype(np.uint8)


# --- images and registration ---------------------------------------------------------------------------------


def _prepare_image(image):
    """RGB floats and a foreground mask, from a Pillow image or an array."""
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] not in (1, 2, 3, 4):
        raise ValueError(f"a view must be H x W x 3 or H x W x 4, not {a.shape}")
    if a.shape[2] in (1, 2):
        a = np.repeat(a[..., :1], 3, axis=2)

    alpha = None
    if a.shape[2] == 4:
        alpha = a[..., 3].astype(np.float32)
        a = a[..., :3]

    rgb = a.astype(np.float32)
    if np.issubdtype(a.dtype, np.floating) and rgb.size and float(np.nanmax(rgb)) <= 1.0:
        rgb = rgb * 255.0

    light = (rgb.min(axis=2) > 200.0) & (rgb.max(axis=2) - rgb.min(axis=2) < 16.0)
    if alpha is not None and float(alpha.max()) > 0.0:
        mask = alpha > 0.5 * float(alpha.max())
        if not mask.any():
            mask = ~light
    else:
        mask = _background_mask(rgb, light)
    if not mask.any():
        raise ValueError("a view has no foreground pixels; give it a plain background")
    return rgb, mask


def _background_mask(rgb, light):
    """Foreground by distance from the commonest colour on the image's border."""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    quantised = np.round(border / 8.0).astype(np.int32)
    values, counts = np.unique(quantised, axis=0, return_counts=True)
    background = values[np.argmax(counts)].astype(np.float32) * 8.0
    mask = np.linalg.norm(rgb - background, axis=2) > _BACKGROUND_TOLERANCE
    if background.min() > 150.0 and background.max() - background.min() < 20.0:
        mask &= ~light
    if not mask.any():  # a drawing with no margin: fall back to the luminance rule
        mask = (rgb.min(axis=2) < 200.0) & ~light
    return mask


def _mirror(prepared):
    rgb, mask = prepared
    return np.ascontiguousarray(rgb[:, ::-1]), np.ascontiguousarray(mask[:, ::-1])


# --- the atlas raster ----------------------------------------------------------------------------------------


def _material_names(car):
    names = {name for part in car.parts for name, mesh in part.pieces.items() if mesh.uvs is not None}
    return sorted(names)


def _rasterise(car, size, names):
    """Per texel: the 3D point, the normal and the material id of the UV island it lies in."""
    pos = np.zeros((size, size, 3), np.float32)
    nrm = np.zeros((size, size, 3), np.float32)
    mat = np.full((size, size), -1, np.int32)
    id_of = {name: i for i, name in enumerate(names)}
    wheel_seen = False

    for part in car.parts:
        # The four wheels share one atlas island; one wheel stands for all of them.
        if part.kind == "wheel":
            if wheel_seen:
                continue
            wheel_seen = True
        for material, mesh in part.pieces.items():
            positions = np.asarray(mesh.positions, np.float32)
            if mesh.uvs is None:
                continue
            uv = np.asarray(mesh.uvs, np.float32) * size
            faces = np.asarray(mesh.faces)
            normals = None if mesh.normals is None else np.asarray(mesh.normals, np.float32)
            material_id = id_of[material]

            for face in faces:
                a, b, c = uv[face]
                x0 = max(int(np.floor(min(a[0], b[0], c[0]))), 0)
                y0 = max(int(np.floor(min(a[1], b[1], c[1]))), 0)
                x1 = min(int(np.ceil(max(a[0], b[0], c[0]))), size - 1)
                y1 = min(int(np.ceil(max(a[1], b[1], c[1]))), size - 1)
                if x1 < x0 or y1 < y0:
                    continue

                xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
                denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
                if abs(denominator) < 1e-9:
                    continue
                l0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / denominator
                l1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / denominator
                l2 = 1.0 - l0 - l1
                inside = (l0 >= -0.01) & (l1 >= -0.01) & (l2 >= -0.01)
                if not inside.any():
                    continue

                yy, xx = np.nonzero(inside)
                yy += y0
                xx += x0
                bary = np.stack([l0[inside], l1[inside], l2[inside]], axis=1)
                pos[yy, xx] = bary @ positions[face]
                if normals is not None:
                    nrm[yy, xx] = bary @ normals[face]
                else:
                    face_normal = np.cross(positions[face[1]] - positions[face[0]],
                                           positions[face[2]] - positions[face[0]])
                    nrm[yy, xx] = face_normal / (np.linalg.norm(face_normal) + 1e-9)
                mat[yy, xx] = material_id

    lengths = np.linalg.norm(nrm, axis=2, keepdims=True)
    np.divide(nrm, lengths + 1e-9, out=nrm)
    return pos, nrm, mat


def _bounds(car):
    """The asset's full box in metres: X front-back, Y ground-roof, Z left-right."""
    vertices = [np.asarray(mesh.positions, np.float32)
                for part in car.parts for mesh in part.pieces.values()]
    if not vertices:
        raise ValueError("the asset has no pieces")
    v = np.concatenate(vertices, axis=0)
    return (float(v[:, 0].min()), float(v[:, 0].max()), float(v[:, 1].min()), float(v[:, 1].max()),
            float(v[:, 2].min()), float(v[:, 2].max()))


def _project(name, pos, box, xmin, xmax, ymin, ymax, zmin, zmax):
    """Project every texel's 3D point into a view, fitted to that image's foreground box."""
    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    across = max(xmax - xmin, 1e-9)
    tall = max(ymax - ymin, 1e-9)
    wide = max(zmax - zmin, 1e-9)
    if name in ("left", "right"):
        u = (xmax - x) / across if name == "left" else (x - xmin) / across
        v = (ymax - y) / tall
    elif name in ("front", "rear"):
        u = (zmax - z) / wide if name == "front" else (z - zmin) / wide
        v = (ymax - y) / tall
    else:  # plan views: the front is at the left in both, the right side up for top and down for bottom
        u = (xmax - x) / across
        v = (zmax - z) / wide if name == "top" else (z - zmin) / wide
    bx0, bx1, by0, by1 = box
    return bx0 + u * (bx1 - bx0), by0 + v * (by1 - by0)


def _view_box(prepared):
    """The image's non-background bounding box as (x0, x1, y0, y1), inclusive."""
    _, mask = prepared
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def _bilinear(image, xf, yf):
    height, width = image.shape[:2]
    xf = np.clip(xf, 0.0, width - 1.001)
    yf = np.clip(yf, 0.0, height - 1.001)
    x0 = xf.astype(np.int32)
    y0 = yf.astype(np.int32)
    dx = xf - x0
    dy = yf - y0
    if image.ndim == 3:
        dx = dx[..., None]
        dy = dy[..., None]
    return (image[y0, x0] * (1.0 - dx) * (1.0 - dy) + image[y0, x0 + 1] * dx * (1.0 - dy)
            + image[y0 + 1, x0] * (1.0 - dx) * dy + image[y0 + 1, x0 + 1] * dx * dy)


# --- colours and gutters --------------------------------------------------------------------------------------


def _palette():
    try:
        from manifoldx.modeling.assets.car import PALETTE
    except ImportError:  # an asset outside the car package still gets a grey fallback
        return {}
    out = {}
    for name, entry in PALETTE.items():
        value = entry[0]
        out[name] = np.array([int(value[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
    return out


def _palette_atlas(mat, names):
    palette = _palette()
    atlas = np.full(mat.shape + (3,), 255.0, np.float32)
    for index, name in enumerate(names):
        colour = palette.get(name, np.array([128.0, 128.0, 128.0], np.float32))
        atlas[mat == index] = colour
    return atlas


def _fallback_for(fallback, material):
    if fallback is None:
        return None
    if isinstance(fallback, dict):
        value = fallback.get(material)
        return None if value is None else _colour(value)
    return _colour(fallback)


def _colour(value):
    colour = np.asarray(value, np.float32).reshape(-1)
    if colour.size != 3:
        raise ValueError("fallback colours must be RGB triples")
    return colour


def _fill_gutters(colour, cover, radius):
    """Dilate covered texels outward, copying the colour of a covered neighbour each step."""
    out = colour.copy()
    filled = cover.copy()
    for _ in range(radius):
        if filled.all():
            break
        newly = np.zeros_like(filled)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            source = _shift(filled, dy, dx, False)
            take = ~filled & ~newly & source
            if take.any():
                out[take] = _shift(out, dy, dx, False)[take]
                newly |= take
        filled |= newly
    return out


def _shift(array, dy, dx, fill):
    """Shift an array by (dy, dx) pixels, leaving the new border at `fill`."""
    out = np.full_like(array, fill)
    height, width = array.shape[:2]
    ys = slice(max(dy, 0), height + min(dy, 0))
    xs = slice(max(dx, 0), width + min(dx, 0))
    yd = slice(max(-dy, 0), height + min(-dy, 0))
    xd = slice(max(-dx, 0), width + min(-dx, 0))
    out[ys, xs] = array[yd, xd]
    return out
