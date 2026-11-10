"""Door mirror shapes: a mirror as flat outlines from three sides.

A shape is drawn in units of the half-size of its front view: the mirror as seen from ahead of the car, u toward
the car's side and v up. Its layers are that view: the whole mirror with its stalk first ("black"), then the cap
in the body's colour ("paint") and the signal strip ("lit"). The glass is the mirror glass on the other side, in
the same (u, v). The plan is the cap from above, as (u, d) with d forward from the glass in the units of u: it
says how deep the mirror is at each point across it. The anchor is the foot of the stalk, which the car places on
the door.

This one was traced from flat-colour drawings of a real mirror, drawn for the purpose.
"""

# A rounded pod on a stalk that leans in to the door: painted cap, black shell under it, a strip round the end.
POD = {
    "aspect": 1.166,
    "anchor": (-0.715, -0.992),
    "layers": [
        ("black", [
            (0.869, 0.671), (0.754, 0.765), (0.537, 0.857), (0.077, 0.971), (-0.317, 0.998), (-0.536,
            0.888), (-0.638, 0.747), (-0.726, 0.49), (-0.69, -0.065), (-1.0, -0.473), (-0.97, -0.616),
            (-0.839, -0.861), (-0.712, -0.992), (-0.593, -0.969), (-0.347, -0.453), (-0.241, -0.329),
            (0.549, -0.251), (0.849, -0.151), (0.919, -0.051), (1.0, 0.184), (0.987, 0.367)
        ]),
        ("paint", [
            (-0.734, 0.353), (-0.704, 0.053), (-0.531, 0.039), (-0.152, 0.057), (-0.103, 0.137),
            (-0.029, 0.18), (0.694, 0.233), (0.828, 0.347), (0.953, 0.414), (0.879, 0.655), (0.736,
            0.775), (0.287, 0.929), (-0.058, 0.988), (-0.302, 1.0), (-0.484, 0.929), (-0.601, 0.81),
            (-0.704, 0.578)
        ]),
        ("lit", [
            (0.988, 0.359), (0.961, 0.394), (0.854, 0.359), (0.738, 0.253), (0.684, 0.229), (-0.04,
            0.176), (-0.103, 0.135), (-0.142, 0.084), (-0.144, 0.057), (0.568, 0.098), (0.963, 0.149),
            (1.0, 0.184)
        ]),
    ],
    "glass": [
        (-0.536, 0.796), (-0.491, 0.845), (-0.42, 0.87), (-0.119, 0.854), (0.329, 0.762), (0.63, 0.647),
        (0.711, 0.556), (0.759, 0.405), (0.782, 0.142), (0.754, -0.03), (0.658, -0.124), (0.395, -0.176),
        (-0.212, -0.221), (-0.383, -0.174), (-0.458, -0.064), (-0.531, 0.261), (-0.56, 0.567), (-0.559,
        0.715)
    ],
    "plan": [
        (-0.734, 0.466), (-0.732, 0.253), (-0.622, 0.0), (-0.575, 0.048), (-0.443, 0.098), (0.105, 0.145),
        (0.637, 0.153), (0.868, 0.118), (0.946, 0.088), (0.972, 0.048), (1.0, 0.082), (0.902, 0.335),
        (0.753, 0.522), (0.556, 0.652), (0.368, 0.712), (-0.122, 0.765), (-0.461, 0.749), (-0.555, 0.716),
        (-0.64, 0.648)
    ],
}

MIRRORS = {"pod": POD}
