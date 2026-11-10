"""Trace the front lamps off the front photo, where they are lit, into outlines in mm for box_blender.py.

Run: uv run --with scipy --with pillow --with numpy --with opencv-python-headless python lamps.py REPORT.json OUT.json

The lit LEDs are the bright, unsaturated pixels of the lamp band. Each blob is outlined, the outline is cast back
through the front photo's camera onto the plane of the car's nose (x = -1.95 m) and read as (y, z) mm, the frame
box_blender.py's front prisms use. Only the eye on one side is traced; the other is its mirror image.
"""

import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import calib  # noqa: E402

PHOTO = os.environ.get("BOX_PHOTOS", "/home/apiad/Workspace/repos/briefings/dongfeng/modelos/box-430/imagenes")
NOSE_X = -1.95


def to_mm(params, uv):
    """Cast photo pixels back onto the plane x = NOSE_X: (y, z) in mm."""
    cx, cy, cz, yaw, pitch, roll, f = params
    right, up, fwd = calib.basis(yaw, pitch, roll)
    d = fwd[None] + ((uv[:, 0] - calib.W / 2) / f)[:, None] * right[None] - ((uv[:, 1] - calib.H / 2) / f)[:, None] * up[None]
    t = (NOSE_X - cx) / d[:, 0]
    p = np.array([cx, cy, cz])[None] + t[:, None] * d
    return np.column_stack([p[:, 1], p[:, 2]]) * 1000


def outline(mask, eps=3.0):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    return cv2.approxPolyDP(c, eps, True)[:, 0, :].astype(float)


def main():
    report, out = sys.argv[1:3]
    params = json.load(open(report))["views"]["03-dongfeng-box-vista-frontal"]["params"]
    im = np.asarray(Image.open(f"{PHOTO}/03-dongfeng-box-vista-frontal.webp").convert("RGB"))
    hsv = np.asarray(Image.fromarray(im).convert("HSV")).astype(int)
    bright = (hsv[..., 2] > 225) & (hsv[..., 1] < 70)
    bright[:825] = False  # above the bar is the hood's reflection, not a lamp
    bright[1150:] = False
    lab, n = ndimage.label(ndimage.binary_closing(bright, iterations=2))
    sizes = ndimage.sum(bright, lab, range(1, n + 1))
    blobs = [lab == i + 1 for i in np.argsort(-sizes)[:6] if sizes[i] > 400]
    box = lambda m: (np.where(m)[1].min(), np.where(m)[1].max(), np.where(m)[0].min())
    centre = calib.W / 2
    bar = max(blobs, key=lambda m: (box(m)[1] - box(m)[0]) if box(m)[0] < centre < box(m)[1] else 0)
    # The eye on the photo's right: the largest blob right of the bar's end, and the projector inside it.
    right = [m for m in blobs if box(m)[0] > box(bar)[1] - 50]
    ring = max(right, key=lambda m: m.sum())
    # The C is open at the top, so the lens is its convex hull: an almond, close to the real eye.
    hull = cv2.convexHull(np.argwhere(ring)[:, ::-1].astype(np.int32))
    eye = np.zeros(ring.shape, np.uint8)
    cv2.fillPoly(eye, [hull], 1)
    eye = eye.astype(bool)
    projector = [m for m in right if m is not ring and (m & eye).any()]
    thick = 2 * np.median(ndimage.distance_transform_edt(ring)[ring])  # the LED stroke's width
    inner = ndimage.binary_erosion(eye, iterations=max(int(thick), 1))

    def side(mask):
        mm = to_mm(params, outline(mask))
        return [[round(abs(y), 1), round(z, 1)] for y, z in mm]

    bar_mm = to_mm(params, outline(bar))
    result = {
        "nose_x": NOSE_X,
        "bar": [[round(y, 1), round(z, 1)] for y, z in bar_mm],
        "bar_half_length": round(float(np.abs(bar_mm[:, 0]).max()), 1),
        "eye": side(eye),
        "eye_inner": side(inner) if inner.any() else [],
        "projector": side(projector[0]) if projector else [],
    }
    json.dump(result, open(out, "w"), indent=1)
    for k in ("bar", "eye", "eye_inner", "projector"):
        pts = np.array(result[k]) if result[k] else np.zeros((0, 2))
        if len(pts):
            print(k, len(pts), "pts, y", pts[:, 0].min().round(), pts[:, 0].max().round(), "z", pts[:, 1].min().round(), pts[:, 1].max().round())


if __name__ == "__main__":
    main()
