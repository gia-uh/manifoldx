"""Measure the BOX from its side and front photos: profiles in millimetres, saved to profile.json.

Axes follow manifoldx: x along the car (front is -x), y up (ground is 0), z across.
"""

import json
import sys

import numpy as np
from PIL import Image, ImageDraw

from mask import car_mask

PHOTOS = "/home/apiad/Workspace/repos/briefings/dongfeng/modelos/box-430/imagenes"
LENGTH, WIDTH, HEIGHT = 4030.0, 1810.0, 1570.0  # ficha técnica
WHEEL_PX = (236, 1186)
# The beltline, traced by hand on the side photo: glass and black pillars sit above it, from the cowl to the tail.
BELT_PX = [(385, 457), (800, 450), (1125, 444), (1300, 470), (1340, 500)]
WHEELBASE = 2660.0  # Wikipedia and CarNewsChina; the side photo alone measures 3012 mm, being a perspective shot
TYRE_D = 17 * 25.4 + 2 * 0.50 * 205  # 205/50 R17, in mm


def smooth(v, k=9):
    pad = np.pad(v, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def lower_edge(paint, dark, top, ground, run=45):
    """The body's lower edge in one column: where a thick dark run (a tyre or the shadow) starts below the body.

    Thin dark runs, such as the door trim, do not count.
    """
    r = top + 10
    while r < ground:
        if not paint[r]:
            end = r
            while end < ground and not paint[end]:
                end += 1
            if end - r >= run or end >= ground:
                return r
            r = end
        r += 1
    return ground


def side(path):
    img, mask = car_mask(path)
    hsv = np.asarray(Image.fromarray(img).convert("HSV")).astype(np.float32)
    paint = mask & (hsv[..., 1] > 38) & (hsv[..., 2] > 75)
    dark = mask & ~paint
    rows = np.where(mask.any(axis=1))[0]
    ground = rows[-1]
    # The car's ends: columns with a real run of paint, which drops the backdrop bleeding in at the edges.
    cols = np.where(paint.sum(axis=0) >= 15)[0]
    x0, x1 = cols[0], cols[-1]
    # The photo is a perspective shot, so one scale does not fit both axes: x goes through four anchors (the
    # ends and the axles, overhangs split as the photo splits them), y through the roof height.
    f_px, r_px = WHEEL_PX[0] - x0, x1 - WHEEL_PX[1]
    front_oh = (LENGTH - WHEELBASE) * f_px / (f_px + r_px)
    anchors_px = [x0, WHEEL_PX[0], WHEEL_PX[1], x1]
    anchors_mm = [-LENGTH / 2, -LENGTH / 2 + front_oh, -LENGTH / 2 + front_oh + WHEELBASE, LENGTH / 2]
    to_x = lambda p: np.interp(p, anchors_px, anchors_mm)
    xs = np.arange(x0, x1 + 1)
    top = np.array([np.argmax(mask[:, c]) for c in xs])
    # The roof rails and the antenna poke out of the roof line: a running median drops them.
    padded = np.pad(top, 20, mode="edge")
    top = np.array([np.median(padded[i:i + 41]) for i in range(len(top))])
    # The body's lower edge: the last painted pixel in the column, so the wheel arches come out of the photo.
    # Glass: dark pixels in the upper half of the body.
    band = (ground - (ground - rows[0]) * 0.45)
    gtop = np.full(len(xs), np.nan)
    gbot = np.full(len(xs), np.nan)
    for i, c in enumerate(xs):
        r = np.where(dark[: int(band), c])[0]
        r = r[r > top[i] + 3]
        if len(r) > 6:
            gtop[i], gbot[i] = r[0], r[-1]
    # Below the beltline: the glass band's lowest row, or the roof line where there is no glass.
    start = np.where(np.isnan(gbot), top, np.fmax(gbot, top)).astype(int) + 8
    bot = np.array([lower_edge(paint[:, c], dark[:, c], start[i], ground) for i, c in enumerate(xs)])
    # Door seams and pillars are thin dark columns: a running median drops them.
    k = 15
    padded = np.pad(bot, k // 2, mode="edge")
    bot = np.array([np.median(padded[i:i + k]) for i in range(len(bot))])
    # Wheel centres, placed by hand on the 1400 px side photo: tracing them was less reliable than looking.
    wheels = [{"x": float(to_x(cx))} for cx in WHEEL_PX]
    s = HEIGHT / (ground - np.min(top))  # mm per pixel, vertically
    to_y = lambda r: (ground - r) * s
    # The arches, drawn rather than traced: a circle 45 mm outside each tyre, over the traced sill line.
    xmm = to_x(xs)
    bottom = to_y(bot)
    sill = float(np.median(bottom[(xmm > wheels[0]["x"] + TYRE_D * 0.7) & (xmm < wheels[1]["x"] - TYRE_D * 0.7)]))
    R, Rv = TYRE_D / 2 + 130, TYRE_D / 2 + 85  # the arch is wider than it is tall
    for w in wheels:
        near = np.abs(xmm - w["x"]) < R
        bottom[near] = np.maximum(sill, TYRE_D / 2 + Rv * np.sqrt(1 - ((xmm[near] - w["x"]) / R) ** 2))
    bottom = np.where(np.abs(xmm[:, None] - [w["x"] for w in wheels]).min(axis=1) < R, bottom,
                      np.maximum(np.minimum(bottom, sill + 120), sill * 0.6))
    return img, dict(
        scale_mm_per_px=s, x0=int(x0), ground=int(ground),
        belt=[[float(to_x(px)), float(to_y(py))] for px, py in BELT_PX],
        x=xmm.tolist(), anchors_px=[int(a) for a in anchors_px], anchors_mm=anchors_mm,
        top=smooth(to_y(top), 21).tolist(), bottom=smooth(bottom, 5).tolist(), sill=sill,
        glass_top=to_y(gtop).tolist(), glass_bottom=to_y(gbot).tolist(),
        wheels=wheels, height_photo=float(to_y(rows[0])),
    )


# The front section, traced by hand on the 1400 px front photo (centre x = 700): (row, half-width in px).
# The mask failed here: the lit flanks lose their saturation and the floor shadow is wider than the car.
FRONT_PX = [(860, 500), (820, 515), (780, 535), (650, 555), (500, 548), (380, 522), (300, 462),
            (265, 368), (100, 292), (70, 245)]


def front(path):
    img = np.asarray(Image.open(path).convert("RGB").resize((1400, 933)))
    rows, half = np.array(FRONT_PX, dtype=float).T
    sx = (WIDTH / 2) / half.max()
    sy = HEIGHT / (rows[0] - rows[-1])
    return img, dict(y=((rows[0] - rows) * sy).tolist(), half_width=(half * sx).tolist(), rows=[70, 860],
                     scale_x=sx, scale_y=sy)


if __name__ == "__main__":
    simg, sp = side(f"{PHOTOS}/04-dongfeng-box-vista-lateral.webp")
    fimg, fp = front(f"{PHOTOS}/03-dongfeng-box-vista-frontal.webp")
    json.dump({"side": sp, "front": fp, "dims": [LENGTH, WIDTH, HEIGHT], "tyre_d": TYRE_D},
              open("profile.json", "w"))
    im = Image.fromarray(simg)
    d = ImageDraw.Draw(im)
    s, g = sp["scale_mm_per_px"], sp["ground"]
    px = lambda x, y: (float(np.interp(x, sp["anchors_mm"], sp["anchors_px"])), g - y / s)
    for key, col in (("top", "#ff3b30"), ("bottom", "#ffcc00"), ("glass_top", "#34c759"), ("glass_bottom", "#34c759")):
        pts = [px(x, y) for x, y in zip(sp["x"], sp[key]) if y == y]
        d.point(pts, fill=col)
        d.line(pts, fill=col, width=3) if key in ("top", "bottom") else None
    for w in sp["wheels"]:
        cx, cy = px(w["x"], TYRE_D / 2)
        r = TYRE_D / 2 / s
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline="#00e5ff", width=3)
    d.line([px(x, y) for x, y in sp["belt"]], fill="#34c759", width=4)
    for a in sp["anchors_px"]:
        d.line([(a, g - 40), (a, g + 10)], fill="#ff2d55", width=3)
    im.save(sys.argv[1])
    fi = Image.fromarray(fimg)
    d = ImageDraw.Draw(fi)
    mid = fi.width // 2
    pts = [(mid + h / fp["scale_x"], fp["rows"][1] - y / fp["scale_y"]) for y, h in zip(fp["y"], fp["half_width"])]
    for side in (pts, [(2 * mid - x, y) for x, y in pts]):
        d.line(side, fill="#ff3b30", width=3)
        for x, y in side:
            d.ellipse([x - 6, y - 6, x + 6, y + 6], outline="#ffcc00", width=3)
    fi.save(sys.argv[1].replace("-04", "-03"))
    wb = sp["wheels"][1]["x"] - sp["wheels"][0]["x"]
    print(f"wheelbase {wb:.0f} mm, tyre r {TYRE_D/2:.0f} mm, "
          f"photo height {sp['height_photo']:.0f} vs {HEIGHT:.0f} mm")
