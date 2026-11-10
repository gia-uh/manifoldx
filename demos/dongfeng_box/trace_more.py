"""Profiles for the Voyah Free (traced on its side photo, as the Friday was) and for the Rich 7 Tiger pickup and the
M5 Cargo van, whose catalogue "side" photos are three-quarter shots: those two are drawn from their dimensions alone,
in fractions of length and height. Writes profile-<car>.json in profile.py's format.

Sources for what the fichas lack: Voyah Free wheelbase 2960 mm (evkx.net); Rich 7 wheelbase 3150 mm (Wikipedia,
Dongfeng Rich); M5 long-wheelbase 3000 mm (dongfengforthing.com, CarNewsChina).
"""

import json

import numpy as np

# The BOX's hand-traced front section, as fractions of height and half-width: the shape a car without a front photo
# borrows.
BOX_FRONT = [(0.0, 0.90), (0.051, 0.928), (0.101, 0.964), (0.266, 1.0), (0.456, 0.987), (0.608, 0.941),
             (0.709, 0.832), (0.753, 0.663), (0.962, 0.526), (1.0, 0.441)]

CARS = {
    "voyah": dict(
        dims=(4905.0, 1950.0, 1645.0), wheelbase=2960.0, tyre=(255, 45, 20), photo_h=933, paint="#101114",
        px=dict(ground=718, front=172, rear=1197, wheels=(340, 1020),
                top=[(172, 560), (176, 520), (200, 505), (300, 462), (450, 452), (520, 420), (600, 385), (680, 358),
                     (800, 352), (950, 355), (1050, 368), (1123, 384), (1150, 430), (1175, 470), (1195, 560),
                     (1197, 600)],
                bottom=[(172, 615), (200, 650), (460, 660), (940, 662), (1170, 640), (1197, 600)],
                belt=[(470, 462), (740, 462), (1000, 458), (1080, 445)]),
        front=BOX_FRONT, plan=(6.0, 7.0)),
    "rich7": dict(
        dims=(5268.0, 1940.0, 1866.0), wheelbase=3150.0, tyre=(255, 65, 18), front_overhang=900.0, paint="#9bd83a",
        # (fraction of length from the nose, fraction of height)
        top=[(0.0, 0.50), (0.02, 0.56), (0.22, 0.60), (0.34, 1.0), (0.55, 1.0), (0.57, 0.62), (0.60, 0.60),
             (0.99, 0.60), (1.0, 0.52)],
        bottom=[(0.0, 0.26), (0.06, 0.20), (0.94, 0.20), (1.0, 0.26)],
        belt=[(0.24, 0.62), (0.56, 0.64)], windows=(0.26, 0.555),
        front=[(0.0, 0.93), (0.3, 1.0), (0.6, 1.0), (0.66, 0.88), (0.75, 0.8), (1.0, 0.74)], plan=(12.0, 14.0)),
    "m5": dict(
        dims=(5115.0, 1720.0, 1960.0), wheelbase=3000.0, tyre=(195, 70, 15), front_overhang=880.0, paint="#e8eaec",
        top=[(0.0, 0.45), (0.04, 0.53), (0.10, 0.56), (0.24, 0.98), (0.30, 1.0), (0.97, 1.0), (1.0, 0.95)],
        bottom=[(0.0, 0.17), (0.05, 0.12), (0.96, 0.12), (1.0, 0.16)],
        belt=[(0.10, 0.56), (0.36, 0.56)], windows=(0.13, 0.34),
        front=[(0.0, 0.95), (0.5, 1.0), (0.56, 0.97), (1.0, 0.90)], plan=(10.0, 16.0)),
}


def build(key, c):
    L, W, H = c["dims"]
    w, a, rim = c["tyre"]
    tyre_d = rim * 25.4 + 2 * a / 100 * w
    xs = np.linspace(-L / 2, L / 2, 400)
    if "px" in c:  # traced on the photo, as trace_friday.py does
        p = c["px"]
        f_px, r_px = p["wheels"][0] - p["front"], p["rear"] - p["wheels"][1]
        front_oh = (L - c["wheelbase"]) * f_px / (f_px + r_px)
        anchors_px = [p["front"], p["wheels"][0], p["wheels"][1], p["rear"]]
        s = H / (p["ground"] - min(y for _, y in p["top"]))
        ground = p["ground"]
        to_x = lambda q: np.interp(np.asarray(q, float), anchors_px,
                                   [-L / 2, -L / 2 + front_oh, -L / 2 + front_oh + c["wheelbase"], L / 2])
        to_y = lambda r: (ground - np.asarray(r, float)) * s
        curve = lambda pts: np.interp(xs, to_x([q for q, _ in pts]), to_y([r for _, r in pts]))
        top, bottom = curve(p["top"]), curve(p["bottom"])
        wheel_x = [float(to_x(q)) for q in p["wheels"]]
        belt = [[float(to_x(q)), float(to_y(r))] for q, r in p["belt"]]
        windows = None
    else:  # designed: a synthetic 1400 px "photo" frame, so the textures and UVs work as for a photo
        front_oh = c["front_overhang"]
        s = L / 1200
        ground = round(H / s) + 60
        anchors_px = [100, 100 + front_oh / s, 100 + (front_oh + c["wheelbase"]) / s, 1300]
        frac = lambda pts: np.interp(xs, [-L / 2 + u * L for u, _ in pts], [v * H for _, v in pts])
        top, bottom = frac(c["top"]), frac(c["bottom"])
        wheel_x = [-L / 2 + front_oh, -L / 2 + front_oh + c["wheelbase"]]
        belt = [[-L / 2 + u * L, v * H] for u, v in c["belt"]]
        windows = [-L / 2 + u * L for u in c["windows"]]
    R, Rv = tyre_d / 2 + 110, tyre_d / 2 + 70
    for wx in wheel_x:
        near = np.abs(xs - wx) < R
        bottom[near] = np.maximum(bottom[near], tyre_d / 2 + Rv * np.sqrt(1 - ((xs[near] - wx) / R) ** 2))
    anchors_mm = [-L / 2, -L / 2 + front_oh, -L / 2 + front_oh + c["wheelbase"], L / 2]
    fy, fw = np.array(c["front"]).T
    side = dict(scale_mm_per_px=s, ground=ground, x=xs.tolist(), top=top.tolist(), bottom=bottom.tolist(),
                belt=sorted(belt), wheels=[{"x": x} for x in wheel_x], anchors_px=anchors_px, anchors_mm=anchors_mm,
                photo_h=c.get("photo_h", round(ground + 60)), windows=windows)
    sx = sy = max(W, H) / 1100
    front = dict(y=(fy * H).tolist(), half_width=(fw * W / 2).tolist(), scale_x=sx, scale_y=sy, centre=700,
                 ground=round(H / sy) + 40, photo_h=round(H / sy) + 80)
    json.dump({"side": side, "front": front, "dims": [L, W, H], "tyre_d": tyre_d, "rim_in": rim,
               "plan_p": list(c["plan"]), "paint": c["paint"], "designed": "px" not in c},
              open(f"profile-{key}.json", "w"))
    print(key, f"tyre {tyre_d:.0f} mm, wheelbase {wheel_x[1] - wheel_x[0]:.0f} mm")


if __name__ == "__main__":
    for key, c in CARS.items():
        build(key, c)
