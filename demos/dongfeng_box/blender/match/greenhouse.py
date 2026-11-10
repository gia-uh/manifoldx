"""Trace the greenhouse off the side photo into outlines in mm for box_blender.py: the black band around the
windows (pillars, frames, the black rear quarter) and the window openings inside it.

Run: uv run --with scipy --with pillow --with numpy --with opencv-python-headless python greenhouse.py OUT.json

The photo is split into classes (regions.classes): paint, dark and light. The greenhouse is the largest dark blob
above the belt, holes filled; the windows are the convex hulls of its light blobs, since this car's side glass is
clear and shows the light cabin. Each outline is cast back through the side photo's camera (calib.py, measured from the wheels) onto the
glass, taken as a surface leaning in from y = 0.85 m at the belt to 0.62 m at the roof, and read as (x, z) mm:
the frame box_blender.py's side prisms use.
"""

import json
import os
import sys

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
import calib  # noqa: E402
import regions  # noqa: E402

PHOTO = os.environ.get("BOX_PHOTOS", "/home/apiad/Workspace/repos/briefings/dongfeng/modelos/box-430/imagenes")
SIDE = "04-dongfeng-box-vista-lateral"
BELT_Z, ROOF_Z, BELT_Y, ROOF_Y = 1.12, 1.55, 0.85, 0.62
LOW_Z = 0.98  # nothing of the greenhouse is lower than this; below it the dark is wheels and trim


def camera():
    c = json.load(open(os.path.join(os.path.dirname(__file__), "cameras.json")))[SIDE]
    return c["position"] + [c["yaw"], c["pitch"], c["roll"], c["focal_px"]]


def to_mm(params, uv):
    """Cast photo pixels onto the left side's glass, y = -(BELT_Y + (z - BELT_Z) * slope): (x, z) in mm."""
    cx, cy, cz, yaw, pitch, roll, f = params
    right, up, fwd = calib.basis(yaw, pitch, roll)
    d = fwd[None] + ((uv[:, 0] - calib.W / 2) / f)[:, None] * right[None] - ((uv[:, 1] - calib.H / 2) / f)[:, None] * up[None]
    slope = (ROOF_Y - BELT_Y) / (ROOF_Z - BELT_Z)
    # cy + t dy = -(BELT_Y + (cz + t dz - BELT_Z) slope), linear in t
    t = (-(BELT_Y + (cz - BELT_Z) * slope) - cy) / (d[:, 1] + d[:, 2] * slope)
    p = np.array([cx, cy, cz])[None] + t[:, None] * d
    return np.column_stack([p[:, 0], p[:, 2]]) * 1000


def row_of(params, z):
    """The highest photo row that z (m) reaches along the car's side."""
    pts = np.array([[x, -BELT_Y, z] for x in np.linspace(-2, 2, 21)])
    cx, cy, cz, yaw, pitch, roll, f = params
    right, up, fwd = calib.basis(yaw, pitch, roll)
    d = pts - np.array([cx, cy, cz])
    return int((calib.H / 2 - f * (d @ up) / (d @ fwd)).min())


def outline(mask, eps=4.0):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    return cv2.approxPolyDP(c, eps, True)[:, 0, :].astype(float)


def main():
    out = sys.argv[1]
    params = camera()
    img, car, cls = regions.classes(f"{PHOTO}/{SIDE}.webp", width=calib.W)
    # The band from its dark pixels only: the light ones also hold the highlight along the belt, which is paint.
    dark = car & (cls == regions.DARK)
    dark[row_of(params, LOW_Z):] = False
    dark = ndimage.binary_opening(ndimage.binary_closing(dark, iterations=4), iterations=3)
    lab, n = ndimage.label(dark)
    sizes = ndimage.sum(dark, lab, range(1, n + 1))
    band = ndimage.binary_fill_holes(lab == 1 + int(np.argmax(sizes)))
    # Windows: light blobs well inside the band, from the biggest down while they are a real pane.
    # The cabin behind the glass shows light and dark patches; a closing of 20 px joins the patches of one opening
    # but not across a pillar, which is about 80 px wide here.
    light = ndimage.binary_opening(band & (cls == regions.LIGHT), iterations=3)
    light = ndimage.binary_closing(light, iterations=20) & band
    lab, n = ndimage.label(light)
    sizes = ndimage.sum(light, lab, range(1, n + 1))
    # A pane is the convex hull of its light blob: through the glass the dash and the seats show dark too.
    hull = lambda m: cv2.fillPoly(np.zeros(m.shape, np.uint8), [cv2.convexHull(np.argwhere(m)[:, ::-1].astype(np.int32))], 1)
    panes = [hull(lab == i + 1) for i in np.argsort(-sizes) if sizes[i] > 0.04 * band.sum()]
    res = {"dlo": [[round(a, 1), round(b, 1)] for a, b in to_mm(params, outline(band))],
           "windows": [[[round(a, 1), round(b, 1)] for a, b in to_mm(params, outline(p))] for p in panes]}
    json.dump(res, open(out, "w"), indent=1)
    xs = np.array(res["dlo"])
    print(f"greenhouse x {xs[:, 0].min():.0f}..{xs[:, 0].max():.0f} z {xs[:, 1].min():.0f}..{xs[:, 1].max():.0f} mm, "
          f"{len(res['dlo'])} points; {len(panes)} windows")


if __name__ == "__main__":
    main()
