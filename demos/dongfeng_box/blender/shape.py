"""The BOX body's shape as parameters: one row per section plus a few factors shared by every section.

Used by box_blender.py (to build the Catmull-Clark body) and by match/optimize_shape.py (to fit it to the photos).
The body is symmetric by construction: only the right half of each section is described and mirrored.
Units: mm. Axes: X along the car (front at -X), Y across (+Y right), Z up.
"""

import json
import os

import numpy as np

# x, top of the section, underside, half-width at the widest point of the side, belt height (where the glass
# starts), half-width of the roof at the top of the glass.
SECTIONS = [
    (-2022, 650, 420, 300, 600, 220),
    (-2010, 770, 355, 480, 700, 360),
    (-1990, 865, 298, 620, 790, 480),
    (-1950, 935, 262, 740, 840, 580),
    (-1880, 988, 242, 830, 880, 650),
    (-1780, 1015, 237, 880, 900, 690),
    (-1650, 1035, 236, 903, 968, 750),
    (-1450, 1065, 238, 907, 1000, 755),
    (-1250, 1095, 236, 907, 1035, 745),
    (-1050, 1130, 232, 905, 1105, 730),
    (-850, 1235, 228, 905, 1122, 690),
    (-600, 1390, 226, 905, 1130, 640),
    (-350, 1520, 225, 905, 1136, 610),
    (-150, 1560, 225, 905, 1140, 600),
    (250, 1570, 225, 905, 1146, 600),
    (700, 1568, 226, 905, 1155, 595),
    (1100, 1558, 230, 905, 1166, 585),
    (1360, 1540, 236, 905, 1175, 570),
    (1560, 1505, 245, 900, 1180, 545),
    (1700, 1390, 255, 885, 1150, 560),
    (1840, 1220, 270, 860, 1100, 600),
    (1950, 1080, 290, 820, 1000, 600),
    (2005, 950, 310, 750, 900, 580),
    (2022, 800, 340, 600, 760, 450),
]
# How a section bends between its measured heights and widths: fractions of the half-width or of the height.
FACTORS = {"floor": 0.62, "tuck": 0.86, "rocker": 0.96, "belt_in": 0.94, "glass_mid": 0.45, "roof_side": 0.80,
           "roof_mid": 0.42, "hood_drop": 120.0, "hood_roof": 0.80, "hood_end": -1080.0}


def load():
    """The shape: the defaults, or the fitted ones from the JSON named by BOX_SHAPE."""
    path = os.environ.get("BOX_SHAPE")
    if path and os.path.exists(path):
        d = json.load(open(path))
        return [tuple(r) for r in d["sections"]], {**FACTORS, **d["factors"]}
    return SECTIONS, dict(FACTORS)


def section(top, bottom, half, belt, roof, k):
    """Half a section on the surface, from the bottom centre round the right side (+Y) to the roof centre, (y, z)."""
    body, upper = belt - bottom, top - belt
    lerp = lambda a, b, t: a + (b - a) * t
    return [
        (0, bottom),
        (k["floor"] * half, bottom),
        (k["tuck"] * half, bottom + 0.035 * body),
        (k["rocker"] * half, bottom + 0.16 * body),
        (half, bottom + 0.48 * body),
        (0.985 * half, belt - 0.14 * body),
        (k["belt_in"] * half, belt - 0.01 * body),
        (lerp(0.98 * k["belt_in"] * half, roof, k["glass_mid"]), belt + 0.45 * upper),
        (roof, top - 0.10 * upper),
        (k["roof_side"] * roof, top - 0.025 * upper),
        (k["roof_mid"] * roof, top - 0.004 * upper),
        (0, top),
    ]


def targets(sections, k):
    """The surface points as an (S, R, 3) array in metres, each section's full ring: right side, then left."""
    rings = []
    for x, top, bottom, half_w, belt, roof in sections:
        # The hood is a dome: its edge (the "belt" here) sits well below the crown and well inside the side. The
        # rule ramps in over 250 mm ahead of the cowl, since a sudden switch folds the surface onto itself.
        w = float(np.clip((k["hood_end"] - x) / 250, 0, 1))
        belt += w * (min(belt, top - k["hood_drop"]) - belt)
        roof += w * (min(roof, k["hood_roof"] * half_w) - roof)
        half = section(top, bottom, half_w, belt, roof, k)
        full = half + [(-y, z) for y, z in reversed(half[1:-1])]
        rings.append([(x / 1000, y / 1000, z / 1000) for y, z in full])
    return np.array(rings)
