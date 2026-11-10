"""Solve each photo's camera from its wheels, the only parts whose 3-D position is known exactly.

Run: uv run --with scipy --with numpy python calib.py > cameras.json

Per wheel, the photo gives the hub centre and the extremes (left, right, top, bottom) of the tyre's outline, read
by hand off full-resolution crops. The model of a wheel is the tyre's outer circle (radius TYRE_R, on the outer
sidewall plane) and the hub cap. A pinhole camera (position, yaw, pitch, roll, focal length in pixels, principal
point at the image centre) is fitted to all of them by least squares, and the residual in pixels is reported: it
says how far the later silhouette comparison can be trusted.

World axes are Blender's as in box_blender.py: metres, X along the car with the front at -X, Y across (+Y is the
car's right), Z up. Both photos show the car's left side (-Y).
"""

import json

import numpy as np

TYRE_R, TRACK_HALF, AXLES = 0.3184, 0.775, (-1.302, 1.358)
OUTER = -(TRACK_HALF + 0.1025)  # outer sidewall plane on the left side
HUB = -(TRACK_HALF + 0.03)  # the rim is dished: the badge sits well inboard of the sidewall
W, H = 2800, 1866

# Pixels in the 2800x1866 photos: hub (u, v), tyre outline left, right, top, bottom.
PHOTOS = {
    "04-dongfeng-box-vista-lateral": {
        "guess": [0.0, -9.0, 0.8, 0.0, 0.0, 0.0, 4500.0],
        "wheels": {
            AXLES[0]: {"hub": (503, 1442), "left": 276, "right": 752, "top": 1200, "bottom": 1651},
            AXLES[1]: {"hub": (2358, 1448), "left": 2110, "right": 2593, "top": 1213, "bottom": 1661},
        },
        # Spec-sheet facts, not the model: the ends of a 4030 mm car on its centre line at bumper height, and the
        # 1570 mm roof crown. They pin the camera's distance, which two wheels in one plane leave loose.
        "extras": [((-2.015, 0.0, 0.55), 0, 120), ((2.015, 0.0, 0.60), 0, 2681), ((0.5, 0.0, 1.57), 1, 612)],
    },
    "02-dongfeng-box-vista-exterior-tres-cuartos": {
        "guess": [-5.0, -4.5, 1.8, 45.0, -12.0, 0.0, 4500.0],
        "wheels": {
            AXLES[0]: {"hub": (1237, 1436), "left": 965, "right": 1550, "top": 1117, "bottom": 1708},
            AXLES[1]: {"hub": (2500, 1038), "left": 2387, "right": 2611, "top": 807, "bottom": 1269},
        },
    },
}


def basis(yaw, pitch, roll):
    """Camera right, up and forward vectors. Yaw 0 looks along +Y; yaw and pitch in degrees, roll about forward."""
    y, p, r = np.radians([yaw, pitch, roll])
    fwd = np.array([np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), np.sin(p)])
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    right, up = right * np.cos(r) + up * np.sin(r), up * np.cos(r) - right * np.sin(r)
    return right, up, fwd


def project(params, pts):
    cx, cy, cz, yaw, pitch, roll, f = params
    right, up, fwd = basis(yaw, pitch, roll)
    d = pts - np.array([cx, cy, cz])
    z = d @ fwd
    return np.column_stack([W / 2 + f * (d @ right) / z, H / 2 - f * (d @ up) / z])


def circle(x):
    """The tyre's outline: both sidewall circles, so a view that sees some tread gets the cylinder's extent."""
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    ring = lambda y: np.column_stack([x + TYRE_R * np.cos(t), np.full_like(t, y), TYRE_R + TYRE_R * np.sin(t)])
    return np.vstack([ring(OUTER), ring(OUTER + 0.205)])


def residuals(params, wheels, extras=()):
    res = [project(params, np.array([pt]))[0][axis] - value for pt, axis, value in extras]
    for x, m in wheels.items():
        uv = project(params, circle(x))
        hub = project(params, np.array([[x, HUB, TYRE_R]]))[0]
        res += [hub[0] - m["hub"][0], hub[1] - m["hub"][1], uv[:, 0].min() - m["left"], uv[:, 0].max() - m["right"],
                uv[:, 1].min() - m["top"], uv[:, 1].max() - m["bottom"]]
    # Two wheels in one plane barely separate focal length from distance, so a weak prior holds the lens near a
    # studio 50-85 mm (about 4500 px on this sensor width) and the horizon level; they cost 1 px per 300 px and 1 deg.
    res += [(params[6] - 4500) / 300, params[5]]
    return np.array(res)


def main():
    from scipy.optimize import least_squares  # only the fit needs scipy; Blender imports the camera math

    out = {}
    for name, spec in PHOTOS.items():
        extras = spec.get("extras", ())
        fit = least_squares(residuals, spec["guess"], args=(spec["wheels"], extras), x_scale="jac")
        r = residuals(fit.x, spec["wheels"], extras)[:-2]
        cx, cy, cz, yaw, pitch, roll, f = fit.x
        out[name] = {"position": [cx, cy, cz], "yaw": yaw, "pitch": pitch, "roll": roll, "focal_px": f,
                     "width": W, "height": H, "rms_px": float(np.sqrt((r ** 2).mean())),
                     "max_px": float(np.abs(r).max())}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
