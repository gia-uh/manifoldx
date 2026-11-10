"""Split a photo of the car into colour classes, and score a model's material regions against them.

Classes inside the car's mask: PAINT (the body colour: saturated and blue), DARK (tyres, gloss black, trim, shadow)
and LIGHT (the rest: clear glass showing the cabin, chrome, lit LEDs, highlights). The model side is a material
id image (register.raster with ids). The score is the IoU of paint and of non-paint where both silhouettes
agree, so an outline error is not counted twice: it says whether windows, pillars, lamps and trim sit where the
photo has them. Paint against non-paint is the split that holds in every photo; dark against light does not,
since the same glass is light from the side and dark in the reflections of the other views.
"""

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mask import car_mask  # noqa: E402

PAINT, DARK, LIGHT = 1, 2, 3
PAINT_MATERIALS = ("Paint",)


def classes(path, width):
    """The photo, the car mask, and a class per pixel (0 outside the car)."""
    img, car = car_mask(path, width=width)
    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(int)
    s, v, h = hsv[..., 1], hsv[..., 2], hsv[..., 0]
    paint = car & (s > 45) & (h > 110) & (h < 170) & (v > 35)  # the front photo is backlit: its paint is dark
    dark = car & ~paint & (v < 85)
    cls = np.zeros(car.shape, np.uint8)
    cls[car] = LIGHT
    cls[paint], cls[dark] = PAINT, DARK
    return img, car, cls


def score(cls, idimg, names, cut):
    """IoU of paint and of non-paint between photo classes and model ids, above the sills where both show car."""
    paint_ids = [names.index(n) + 1 for n in PAINT_MATERIALS if n in names]
    both = (cls > 0) & (idimg > 0)
    both[cut:] = False
    mp, pp = np.isin(idimg, paint_ids) & both, (cls == PAINT) & both
    iou = lambda a, b: float((a & b).sum() / max((a | b).sum(), 1))
    return {"paint": round(iou(mp, pp), 4), "non_paint": round(iou(both & ~mp, both & ~pp), 4)}


def overlay(img, cls, idimg, names, cut):
    """Where model and photo disagree: green photo paint the model lacks, magenta model paint the photo lacks."""
    paint_ids = [names.index(n) + 1 for n in PAINT_MATERIALS if n in names]
    both = (cls > 0) & (idimg > 0)
    mp, pp = np.isin(idimg, paint_ids) & both, (cls == PAINT) & both
    ov = (img * 0.45).astype(np.uint8)
    ov[pp & ~mp] = (40, 220, 60)
    ov[mp & ~pp] = (230, 40, 200)
    ov[cut:] = (ov[cut:] * 0.4).astype(np.uint8)
    return ov
