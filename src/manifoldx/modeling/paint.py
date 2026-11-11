"""Painting an image model's answers back onto a view atlas.

Pictures are numpy arrays with row 0 at the top; a region map is an int array
whose 0 is the background and whose other values are material indices.
"""

from __future__ import annotations

import numpy as np

from manifoldx.image import resize

WHITE = 235


def paste_island(atlas_img, island_rect, picture, pad_offset):
    """Paste a model's answer for one island back into the atlas.

    ``picture`` is the model's answer for the padded canvas: it is resized to
    the padded size, cropped back to the island (at ``pad_offset``, as
    :func:`manifoldx.image.pad_to_aspect` returns it) and written into
    ``atlas_img`` at ``island_rect = (x0, y0, x1, y1)``. The padded size is the
    island size plus the padding on both sides, which centred padding makes
    recoverable from the offset. Returns ``atlas_img``.
    """
    x0, y0, x1, y1 = (int(round(value)) for value in island_rect)
    width, height = x1 - x0, y1 - y0
    if width < 1 or height < 1:
        raise ValueError("island_rect must have a positive size")
    pad_x, pad_y = int(pad_offset[0]), int(pad_offset[1])
    padded = resize(np.asarray(picture), (width + 2 * pad_x, height + 2 * pad_y))
    crop = padded[pad_y:pad_y + height, pad_x:pad_x + width]
    if atlas_img.ndim == 3 and crop.ndim == 2:
        channels = 1 if atlas_img.shape[2] == 1 else 3
        crop = np.repeat(crop[..., None], channels, axis=-1)
    if atlas_img.ndim == 3 and atlas_img.shape[2] == 4:
        if crop.shape[2] == 3:
            crop = np.concatenate([crop, np.full(crop.shape[:2] + (1,), 255, crop.dtype)], axis=-1)
    elif atlas_img.ndim == 3 and atlas_img.shape[2] == 3 and crop.ndim == 3 and crop.shape[2] == 4:
        crop = crop[..., :3]
    atlas_img[y0:y1, x0:x1] = crop
    return atlas_img


def kept_outline(template, picture, white=WHITE):
    """IoU of the non-white pixels of a template and a model's answer."""
    a, b = np.asarray(template), np.asarray(picture)
    if a.shape[:2] != b.shape[:2]:
        raise ValueError("template and picture must have the same height and width")
    mask_a = np.any(a[..., :3] < white, axis=-1) if a.ndim == 3 else a < white
    mask_b = np.any(b[..., :3] < white, axis=-1) if b.ndim == 3 else b < white
    union = np.count_nonzero(mask_a | mask_b)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(mask_a & mask_b) / union)


def _colour_table(colors):
    """Region index -> RGB; 0 is unused. Accepts an ordered mapping or a sequence."""
    if hasattr(colors, "values"):
        items = list(colors.items())
        if items and all(isinstance(key, (int, np.integer)) for key, _ in items):
            table = [None] * (max(int(key) for key, _ in items) + 1)
            for key, value in items:
                table[int(key)] = value
            return table
        return [None] + [value for _, value in items]
    return [None] + list(colors)


def clean(picture, regions, colors, tone=None):
    """Repair a painted island: fill white holes and correct an overall tone.

    A pixel left white inside a region becomes that region's flat colour.
    ``colors`` is an ordered mapping of material name to RGB (its order gives
    the region index, as :func:`manifoldx.modeling.views.template` uses it) or
    a sequence indexed by region. With ``tone=(region, rgb)`` the whole picture
    is scaled so the median of that region becomes ``rgb``. Returns a new uint8
    image.
    """
    picture = np.asarray(picture)
    if picture.ndim == 2:
        picture = np.repeat(picture[..., None], 3, axis=-1)
    out = picture[..., :3].astype(float)
    regions = np.asarray(regions)
    table = _colour_table(colors)
    white = np.all(out >= WHITE, axis=-1)
    for index in range(1, len(table)):
        if table[index] is None:
            continue
        hole = (regions == index) & white
        out[hole] = np.asarray(table[index], dtype=float).reshape(3)

    if tone is not None:
        index, target = tone
        selected = regions == int(index)
        if selected.any():
            median = np.median(out[selected], axis=0)
            factor = np.divide(np.asarray(target, dtype=float).reshape(3), median,
                               out=np.ones(3), where=median > 1e-6)
            out = out * factor

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _shift(a, dy, dx, fill):
    """Shift an array by (dy, dx), padding with ``fill``."""
    out = np.full_like(a, fill)
    height, width = a.shape[:2]
    src_y = slice(max(0, -dy), height - max(0, dy))
    dst_y = slice(max(0, dy), height - max(0, -dy))
    src_x = slice(max(0, -dx), width - max(0, dx))
    dst_x = slice(max(0, dx), width - max(0, -dx))
    out[dst_y, dst_x] = a[src_y, src_x]
    return out


def bleed(img, mask, steps=8):
    """Grow the colours of ``mask`` outwards, so a seam filter never meets white.

    Each step copies the average colour of the already-filled neighbouring
    pixels into the unfilled ones next to them. Returns a new image.
    """
    out = np.array(img, copy=True)
    filled = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(steps)):
        total = np.zeros(out.shape, dtype=float)
        count = np.zeros(out.shape[:2], dtype=float)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            near = _shift(filled, dy, dx, False)
            total += _shift(out, dy, dx, 0) * near[..., None]
            count += near
        new = (~filled) & (count > 0)
        if not new.any():
            break
        mean = total[new] / count[new][:, None] if out.ndim == 3 else total[new] / count[new]
        out[new] = np.rint(mean).astype(out.dtype)
        filled |= new
    return out


def _box_blur(a, radius):
    """Mean over a ``(2 * radius + 1) ** 2`` window, with edge padding."""
    radius = int(radius)
    if radius <= 0:
        return np.asarray(a, dtype=float)
    arr = np.asarray(a, dtype=float)
    pad = np.pad(arr, radius, mode="edge")
    integral = np.pad(np.cumsum(np.cumsum(pad, axis=0), axis=1), ((1, 0), (1, 0)))
    k = 2 * radius + 1
    total = (integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k])
    return total / (k * k)


def normal_from_albedo(albedo, strength=6.0, blur=6, line=4.0):
    """A tangent-space normal map (OpenGL, flat = 128, 128, 255) from a picture.

    Darker-than-their-surroundings thin lines become grooves: the local mean
    minus the pixel isolates them, ``line`` scales that relief, a box blur of
    ``blur`` pixels smooths it, and its gradient tilts the normals by
    ``strength``. The result is a uint8 RGB image.
    """
    arr = np.asarray(albedo)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    rgb = arr[..., :3].astype(float)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    relief = np.maximum(_box_blur(lum, blur) - lum, 0.0) * float(line)
    relief = _box_blur(relief, blur)

    d_row, d_col = np.gradient(relief)
    nx = -d_col * float(strength)
    ny = d_row * float(strength)  # rows grow down, the map's green points up
    nz = np.ones_like(nx)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / length, ny / length, nz / length
    encoded = np.stack([nx, ny, nz], axis=-1) * 0.5 + 0.5
    return np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)


def flip_normal_x(normal):
    """Invert a normal map's red channel, for a mirrored island."""
    out = np.array(normal, copy=True)
    out[..., 0] = 255 - out[..., 0]
    return out
