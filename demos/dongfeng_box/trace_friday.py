"""The Friday EV, traced entirely by hand: its photos defeat the BOX's mask (white paint on a pale blue backdrop,
the side shot faces right, the front shot is of a green car). Writes profile-friday.json in profile.py's format.

Pixel coordinates are on the 1400 px wide side (1400x788) and front (1400x788) photos.
"""

import json

import numpy as np

LENGTH, WIDTH, HEIGHT = 4600.0, 1860.0, 1680.0       # ficha técnica
WHEELBASE = 2715.0                                    # techlekh.com, topgear.com.ph
TYRE_D = 19 * 25.4 + 2 * 0.55 * 235                   # 235/55 R19 (zigwheels); the ficha's "155/90" is wrong
GROUND, FRONT_X, REAR_X = 560, 1105, 298
WHEEL_PX = (946, 442)                                 # front, rear
TOP_PX = [(1103, 505), (1105, 480), (1100, 440), (1090, 410), (1050, 390), (1010, 375), (960, 365), (900, 355),
          (860, 345), (790, 295), (720, 270), (660, 266), (560, 265), (470, 268), (420, 275), (385, 290), (330, 325),
          (310, 345), (300, 400), (298, 470)]
BOTTOM_PX = [(1103, 512), (1040, 515), (860, 515), (520, 515), (330, 490), (300, 485)]
BELT_PX = [(850, 350), (790, 345), (640, 340), (430, 330), (400, 300)]
FRONT_PX = [(608, 270), (545, 286), (520, 288), (400, 292), (330, 280), (290, 250), (265, 205), (200, 190),
            (160, 165), (155, 150)]


def main():
    f_px, r_px = FRONT_X - WHEEL_PX[0], WHEEL_PX[1] - REAR_X
    front_oh = (LENGTH - WHEELBASE) * f_px / (f_px + r_px)
    anchors_px = [FRONT_X, WHEEL_PX[0], WHEEL_PX[1], REAR_X]   # front first: this photo faces right
    anchors_mm = [-LENGTH / 2, -LENGTH / 2 + front_oh, -LENGTH / 2 + front_oh + WHEELBASE, LENGTH / 2]
    to_x = lambda p: np.interp(-np.asarray(p, float), [-a for a in anchors_px], anchors_mm)
    s = HEIGHT / (GROUND - min(y for _, y in TOP_PX))
    to_y = lambda r: (GROUND - np.asarray(r, float)) * s
    xs = np.linspace(-LENGTH / 2, LENGTH / 2, 400)
    curve = lambda pts: np.interp(xs, *zip(*sorted(zip(to_x([p for p, _ in pts]), to_y([q for _, q in pts])))))
    top, bottom = curve(TOP_PX), curve(BOTTOM_PX)
    wheels = [{"x": float(to_x(c))} for c in WHEEL_PX]
    R, Rv = TYRE_D / 2 + 110, TYRE_D / 2 + 60
    for w in wheels:
        near = np.abs(xs - w["x"]) < R
        bottom[near] = np.maximum(bottom[near], TYRE_D / 2 + Rv * np.sqrt(1 - ((xs[near] - w["x"]) / R) ** 2))
    rows, half = np.array(FRONT_PX, float).T
    sx, sy = (WIDTH / 2) / half.max(), HEIGHT / (rows[0] - rows[-1])
    side = dict(scale_mm_per_px=s, ground=GROUND, x=xs.tolist(), top=top.tolist(), bottom=bottom.tolist(),
                belt=sorted([[float(to_x(p)), float(to_y(q))] for p, q in BELT_PX]), wheels=wheels,
                anchors_px=anchors_px, anchors_mm=anchors_mm, sill=float(to_y(515)),
                photo_wheelbase=float(abs(WHEEL_PX[0] - WHEEL_PX[1]) * LENGTH / (FRONT_X - REAR_X)))
    front = dict(y=((rows[0] - rows) * sy).tolist(), half_width=(half * sx).tolist(), rows=[150, 608],
                 scale_x=sx, scale_y=sy, centre=700, ground=608)
    json.dump({"side": side, "front": front, "dims": [LENGTH, WIDTH, HEIGHT], "tyre_d": TYRE_D},
              open("profile-friday.json", "w"))
    print(f"height scale {s:.2f} mm/px, photo wheelbase {side['photo_wheelbase']:.0f} vs {WHEELBASE:.0f} mm")


if __name__ == "__main__":
    main()
