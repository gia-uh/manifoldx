"""Turn a scene's render passes into a drawing, deterministically.

`ink_and_wash` reads the ids, normals, depth and light passes of
`Engine.render_frame` and paints an ink-and-watercolour image: lines where
objects, parts and creases meet, a wash per material shaded in three soft
steps, pigment blotches, pooling at region edges and paper grain. The same
buffers, style and seed give the same image.

Needs scipy: `pip install manifold-gfx[ink]`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("manifoldx.stylize needs scipy: pip install 'manifold-gfx[ink]'") from e

PAPER = np.array([0.975, 0.955, 0.905])


@dataclass(frozen=True)
class Label:
    """What one id label is: the scene object it belongs to and its wash.

    Lines between labels of one object are thinner part lines; lines between
    objects are outlines. `no_ink` objects get no lines at all (clouds, a sun).
    """

    object: int
    material: str
    no_ink: bool = False


def ink_and_wash(buffers: dict, labels: list[Label], style: dict, seed: int = 0) -> np.ndarray:
    """Paint the passes as ink and wash; (h, w, 3) float in 0..1.

    buffers: `ids`, `normals`, `depth` and `light` from `render_frame`.
    labels: one `Label` per id label, in order: labels[i] describes label i + 1.
    style:
        ink: {color, object, part, crease, wobble, far}: line colour, the three
            line widths in pixels, the wobble in pixels, and the distance where
            lines have faded to 30%.
        sky: {gradient: [hex, ...], blotch, blotch_scale}: the background wash,
            top to bottom.
        materials: {name: {color, shadow, gradient?, blotch, blotch_scale,
            streak?}}: each material's lit and shadow colour, an optional
            vertical gradient in place of `color`, blotch strength and size in
            pixels, and streaks along "x" or "y" (water, grain).
    """
    ids, normals, depth, light = (buffers[k] for k in ("ids", "normals", "depth", "light"))
    h, w = ids.shape
    rng = np.random.default_rng(seed)
    names = list(style["materials"])
    index = {n: i for i, n in enumerate(names)}
    unknown = sorted({lb.material for lb in labels} - set(index))
    if unknown:
        raise ValueError(f"labels use materials missing from style['materials']: {', '.join(unknown)}")
    objs = np.array([0] + [lb.object for lb in labels])[ids]
    mats = np.array([-1] + [index[lb.material] for lb in labels])[ids]
    no_ink_obj = np.zeros(max([lb.object for lb in labels], default=0) + 1, bool)
    for lb in labels:
        no_ink_obj[lb.object] |= lb.no_ink
    ink_spec, mspec = style["ink"], style["materials"]
    depth = np.where(np.isfinite(depth), depth, 1e9)

    def noise(scale, stretch=(1, 1)):
        """Zero-mean, unit-variance smooth noise; scale is the feature size in pixels."""
        sy, sx = max(scale * stretch[0], 1), max(scale * stretch[1], 1)
        hh, ww = max(int(h / sy * 3), 4), max(int(w / sx * 3), 4)
        small = ndi.gaussian_filter(rng.standard_normal((hh, ww)), 1.0, mode="wrap")
        big = np.asarray(Image.fromarray(small.astype(np.float32), "F").resize((w, h), Image.BICUBIC))
        return (big - big.mean()) / big.std()

    def fbm(scale, stretch=(1, 1)):
        n = noise(scale, stretch) + 0.5 * noise(scale / 2, stretch) + 0.25 * noise(scale / 4, stretch)
        return n / n.std()

    def warp(img, dy, dx):
        yy, xx = np.mgrid[0:h, 0:w]
        coords = [yy + dy, xx + dx]
        if img.ndim == 2:
            return ndi.map_coordinates(img, coords, order=1, mode="nearest")
        return np.stack([ndi.map_coordinates(img[..., c], coords, order=1, mode="nearest") for c in range(img.shape[2])], -1)

    def gradient(stops, mask):
        """A vertical wash through `stops`, spanning the rows the mask covers."""
        rows = np.where(mask.any(1))[0]
        t = np.clip((np.arange(h) - rows[0]) / max(rows[-1] - rows[0], 1), 0, 1) * (len(stops) - 1)
        cols = np.array([_rgb(c) for c in stops])
        col = np.stack([np.interp(t, np.arange(len(stops)), cols[:, c]) for c in range(3)], -1)
        return np.broadcast_to(col[:, None, :], (h, w, 3))

    # Ink: three line weights, from the object map, the label map and the normals.
    obj_edge = _edges(objs)
    no_ink = ndi.binary_dilation(no_ink_obj[objs], iterations=2)
    obj_edge &= ~no_ink
    part_edge = _edges(ids) & ~obj_edge & ~no_ink
    dx = (normals[:, :-1] * normals[:, 1:]).sum(-1)
    dy = (normals[:-1] * normals[1:]).sum(-1)
    crease = np.zeros(ids.shape, bool)
    crease[:, :-1] |= (dx < 0.86) & (ids[:, :-1] == ids[:, 1:])
    crease[:-1] |= (dy < 0.86) & (ids[:-1] == ids[1:])
    crease[ids == 0] = False

    far = _smooth(ink_spec["far"], 8, depth)  # 1 close, 0 far away
    far = 0.3 + 0.7 * ndi.grey_dilation(far, size=3)  # a line belongs to the nearer side
    wob = ink_spec["wobble"]
    wy = noise(45) * wob + noise(11) * wob * 0.35
    wx = noise(45) * wob + noise(11) * wob * 0.35
    press = np.clip(1.15 + 0.22 * noise(30), 0.82, 1.5)

    def stroke(edge, width):
        sigma = max(width / 2.1, 0.5)
        field = ndi.gaussian_filter(warp(edge.astype(np.float32), wy, wx), sigma)
        peak = 1 / (sigma * np.sqrt(2 * np.pi))  # a straight line's height after the blur
        return _smooth(0.38, 0.72, field / peak * press)

    ink = np.maximum.reduce([
        stroke(obj_edge, ink_spec["object"]) * far,
        stroke(part_edge, ink_spec["part"]) * far * 0.92,
        stroke(crease, ink_spec["crease"]) * far * 0.8,
    ])
    dry = _smooth(-2.2, -0.9, noise(6, (1, 3)))  # dry-brush gaps
    ink = ink * (0.72 + 0.28 * dry)

    # Wash: lit and shadow colour per material, three soft steps from the light pass.
    base = np.zeros((h, w, 3))
    shade = np.zeros((h, w, 3))
    for i, name in enumerate(names):
        m = mats == i
        if not m.any():
            continue
        s = mspec[name]
        base[m] = gradient(s["gradient"], m)[m] if "gradient" in s else _rgb(s["color"])
        shade[m] = _rgb(s["shadow"])
    sky = ids == 0
    if sky.any():
        base[sky] = gradient(style["sky"]["gradient"], sky)[sky]
        shade[sky] = base[sky]

    lum = light
    if (~sky).any():
        lo, hi = np.percentile(lum[~sky], [4, 96])
        lum = np.clip((lum - lo) / max(hi - lo, 1e-6), 0, 1)
    edge_noise = noise(14) * 0.035
    tone = 0.5 * _smooth(0.14, 0.2, lum + edge_noise) + 0.5 * _smooth(0.5, 0.58, lum + edge_noise)
    tone = 0.7 * tone + 0.3 * lum
    tone[sky] = 1
    flat = shade * (1 - tone[..., None]) + base * tone[..., None]

    # Pigment: density moves with the blotches, the streaks and the region edges.
    density = np.ones((h, w))
    for i, name in enumerate([*names, None]):
        s = style["sky"] if name is None else mspec[name]
        m = sky if name is None else mats == i
        if not m.any() or not s.get("blotch"):
            continue
        field = fbm(s["blotch_scale"])
        if s.get("streak") == "x":
            field = 0.45 * field + 0.55 * fbm(5, (1, 14))
        elif s.get("streak") == "y":
            field = 0.45 * field + 0.55 * fbm(5, (14, 1))
        density[m] += s["blotch"] * 0.5 * field[m]

    regions = mats + 1
    regions[sky] = 0
    dist = ndi.distance_transform_edt(~_edges(regions))
    density += 0.28 * np.exp(-dist / 5.0) * (~sky)  # pigment pools at the edge of a wet region
    density = np.clip(density, 0.25, 2.2)

    # Colour runs a little past the line, as a hand wash does.
    my, mx = noise(30) * 0.9, noise(30) * 0.9
    wash = ndi.gaussian_filter(warp(flat, my, mx), (0.7, 0.7, 0))
    dens = warp(density, my, mx)

    grain = ndi.gaussian_filter(rng.standard_normal((h, w)), 0.8)
    grain = (grain - grain.mean()) / grain.std()
    tooth = 0.6 * grain + 0.4 * fbm(3, (1, 2.5))
    transmit = np.clip(wash, 0.001, 1) / PAPER
    painted = PAPER[None, None] * transmit ** (dens + 0.1 * tooth)[..., None]
    painted *= (1 + 0.012 * tooth)[..., None]

    inkc = _rgb(ink_spec["color"])
    return np.clip(painted * (1 - ink[..., None]) + inkc * ink[..., None], 0, 1)


def _rgb(c: str) -> np.ndarray:
    c = c.lstrip("#")
    return np.array([int(c[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _edges(a: np.ndarray) -> np.ndarray:
    out = np.zeros(a.shape, bool)
    out[:, :-1] |= a[:, :-1] != a[:, 1:]
    out[:-1, :] |= a[:-1, :] != a[1:, :]
    return out
